import json

import pytest
from aiohttp import web

from invokeai_discord_bot.bot import GenerationState, InvokeBot
from invokeai_discord_bot.config import Settings
from invokeai_discord_bot.ollama import ENHANCER_SYSTEM_PROMPT, OllamaClient, OllamaError, clean_enhanced_prompt


def test_clean_enhanced_prompt_strips_quotes_thinking_and_newlines():
    raw = '<think>plan</think>\n"A clockwork fox,\n\nlit by candles"\n'
    assert clean_enhanced_prompt(raw) == "A clockwork fox, lit by candles"


@pytest.fixture
async def ollama_server():
    from aiohttp.test_utils import TestServer

    servers = []

    async def start(handler):
        app = web.Application()
        app.router.add_post("/api/chat", handler)
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        return server

    yield start
    for server in servers:
        await server.close()


async def test_enhance_sends_system_prompt_and_original_as_user(ollama_server):
    seen = {}

    async def handler(request):
        seen.update(await request.json())
        return web.json_response({"message": {"role": "assistant", "content": "A detailed fox"}})

    server = await ollama_server(handler)
    client = OllamaClient(str(server.make_url("")), "llama3.1:8b")
    try:
        assert await client.enhance("a fox") == "A detailed fox"
    finally:
        await client.close()
    assert seen["model"] == "llama3.1:8b"
    assert seen["stream"] is False
    assert seen["messages"] == [
        {"role": "system", "content": ENHANCER_SYSTEM_PROMPT},
        {"role": "user", "content": "a fox"},
    ]


async def test_enhance_raises_on_http_error(ollama_server):
    async def handler(request):
        return web.Response(status=404, text="model not found")

    server = await ollama_server(handler)
    client = OllamaClient(str(server.make_url("")), "missing")
    try:
        with pytest.raises(OllamaError):
            await client.enhance("a fox")
    finally:
        await client.close()


def _settings(**ollama):
    return Settings(
        discord_token="x", invokeai_url="http://invoke", invokeai_token=None, queue="default",
        poll_interval=1, timeout=10, max_concurrent_jobs=1, guild_id=None, generation_defaults={},
        **ollama,
    )


class _FakeOllama:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    async def enhance(self, prompt):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.result

    async def close(self):
        pass


async def test_generation_prompt_is_unchanged_when_disabled():
    bot = InvokeBot(_settings())
    assert bot.ollama is None
    assert await bot.generation_prompt("a fox") == "a fox"


async def test_generation_prompt_uses_enhancement_and_falls_back_on_error():
    bot = InvokeBot(_settings(ollama_enabled=True, ollama_model="llama3.1:8b"))
    bot.ollama = _FakeOllama(result="a richly detailed fox")
    assert await bot.generation_prompt("a fox") == "a richly detailed fox"
    bot.ollama = _FakeOllama(error=OllamaError("down"))
    assert await bot.generation_prompt("a fox") == "a fox"


async def test_render_sends_enhanced_prompt_but_displays_original(monkeypatch):
    bot = InvokeBot(_settings(ollama_enabled=True, ollama_model="m"))
    bot.ollama = _FakeOllama(result="ENHANCED")
    captured = {}

    async def resolve_model(*args, **kwargs):
        return {"name": "Model", "base": "sdxl"}

    def build(values, model):
        captured["prompt"] = values["prompt"]
        return {}

    async def generate(*args):
        return b"png", "out.png"

    monkeypatch.setattr(bot.invoke, "resolve_model", resolve_model)
    monkeypatch.setattr(bot.invoke, "generate", generate)
    monkeypatch.setattr("invokeai_discord_bot.bot.build_generation_graph", build)
    state = GenerationState(1, "a fox", "", 1024, 1024, 1, 20, 7)
    embed, _ = await bot.render(state)
    assert captured["prompt"] == "ENHANCED"
    assert embed.description == "a fox"
    assert state.prompt == "a fox"
    await bot.invoke.close()


def test_config_requires_model_when_enabled(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"discord": {"token": "t"}, "ollama": {"enabled": True}}))
    monkeypatch.setenv("BOT_CONFIG", str(path))
    for name in ("DISCORD_TOKEN", "OLLAMA_ENABLED", "OLLAMA_MODEL", "OLLAMA_URL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="ollama.model"):
        Settings.from_env()


def test_config_reads_ollama_section(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "discord": {"token": "t"},
        "ollama": {"enabled": True, "url": "http://gpu-box:11434/", "model": "qwen3:8b", "timeout_seconds": 30},
    }))
    monkeypatch.setenv("BOT_CONFIG", str(path))
    for name in ("DISCORD_TOKEN", "OLLAMA_ENABLED", "OLLAMA_MODEL", "OLLAMA_URL", "OLLAMA_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.from_env()
    assert settings.ollama_enabled is True
    assert settings.ollama_url == "http://gpu-box:11434"
    assert settings.ollama_model == "qwen3:8b"
    assert settings.ollama_timeout == 30
