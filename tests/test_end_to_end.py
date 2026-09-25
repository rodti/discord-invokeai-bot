"""Drive the real /dream command and result buttons end to end.

Discord is replaced by small fakes; InvokeAI and Ollama are local HTTP servers
that mimic their real APIs, and the bot talks to them with its real clients.
"""

import asyncio
import socket

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from invokeai_discord_bot import bot as bot_module
from invokeai_discord_bot.bot import PromptModal, ResultView, TweakView, create_bot
from invokeai_discord_bot.config import Settings

PNG = b"\x89PNG\r\n\x1a\nfake"


# --- fake servers --------------------------------------------------------------

class FakeInvokeAI:
    def __init__(self, base="sdxl"):
        self.base = base
        self.graphs = []

    def app(self):
        app = web.Application()
        app.router.add_get("/api/v2/models/", self.models)
        app.router.add_post("/api/v1/queue/default/enqueue_batch", self.enqueue)
        app.router.add_get("/api/v1/queue/default/i/{item}", self.item)
        app.router.add_get("/api/v1/images/i/{name}/full", self.image)
        app.router.add_get("/openapi.json", self.openapi)
        return app

    async def models(self, request):
        return web.json_response({"models": [
            {"key": "main-1", "hash": "h1", "name": "Test Model", "base": self.base, "type": "main"},
            {"key": "lora-1", "hash": "h2", "name": "Test LoRA", "base": self.base, "type": "lora"},
        ]})

    async def enqueue(self, request):
        self.graphs.append((await request.json())["batch"]["graph"])
        return web.json_response({"item_ids": [len(self.graphs)]})

    async def item(self, request):
        name = f"image-{request.match_info['item']}.png"
        return web.json_response({"status": "completed", "session": {"results": {"n": {"image": {"image_name": name}}}}})

    async def image(self, request):
        return web.Response(body=PNG, content_type="image/png")

    async def openapi(self, request):
        return web.json_response({"components": {"schemas": {"Denoise": {"properties": {"scheduler": {"enum": ["euler", "ddim"]}}}}}})

    def prompts(self):
        return [graph["nodes"]["positive"]["prompt"] for graph in self.graphs]


class FakeOllama:
    def __init__(self, mode="ok"):
        self.mode = mode
        self.received = []

    def app(self):
        app = web.Application()
        app.router.add_post("/api/chat", self.chat)
        return app

    async def chat(self, request):
        body = await request.json()
        prompt = body["messages"][1]["content"]
        self.received.append(prompt)
        if self.mode == "error":
            return web.json_response({"error": "model not found"}, status=404)
        if self.mode == "garbage":
            return web.Response(text="<html>bad gateway</html>", content_type="text/html")
        return web.json_response({"model": body["model"], "done": True,
                                  "message": {"role": "assistant", "content": enhanced(prompt)}})


def enhanced(prompt):
    return f"ENHANCED[{prompt}] with volumetric light and a shallow depth of field"


# --- fake Discord ----------------------------------------------------------------

class FakeMessage:
    def __init__(self, log):
        self.log = log
        self.final = None

    async def edit(self, **kwargs):
        self.log.append(kwargs)
        if "embed" in kwargs or "view" in kwargs:
            self.final = kwargs


class FakeResponse:
    def __init__(self):
        self.modal = None

    async def defer(self, **kwargs):
        pass

    async def send_modal(self, modal):
        self.modal = modal

    async def send_message(self, *args, **kwargs):
        pass


class FakeFollowup:
    def __init__(self, interaction):
        self.interaction = interaction

    async def send(self, content=None, wait=False, **kwargs):
        self.interaction.shown.append({"content": content})
        message = FakeMessage(self.interaction.shown)
        self.interaction.messages.append(message)
        return message


class FakeInteraction:
    def __init__(self, user_id=42):
        self.user = type("User", (), {"id": user_id})()
        self.response = FakeResponse()
        self.followup = FakeFollowup(self)
        self.shown = []
        self.messages = []

    async def edit_original_response(self, **kwargs):
        self.shown.append(kwargs)
        message = FakeMessage(self.shown)
        self.messages.append(message)
        return message

    def result(self):
        final = self.messages[-1].final
        assert final is not None, f"no result was posted: {self.shown}"
        return final

    def everything_shown(self):
        text = []
        for entry in self.shown:
            text.append(str(entry.get("content") or ""))
            embed = entry.get("embed")
            if embed is not None:
                text.append(str(embed.description))
                text.append(str(embed.footer.text))
        return "\n".join(text)


# --- harness ---------------------------------------------------------------------

@pytest.fixture
async def harness(monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(bot_module.asyncio, "sleep", lambda _: real_sleep(0))
    started = []

    async def make(ollama_mode="ok", enabled=True, ollama_url=None, base="sdxl"):
        invoke, ollama = FakeInvokeAI(base), FakeOllama(ollama_mode)
        servers = [TestServer(invoke.app()), TestServer(ollama.app())]
        for server in servers:
            await server.start_server()
        started.extend(servers)
        settings = Settings(
            discord_token="t", invokeai_url=str(servers[0].make_url("")).rstrip("/"), invokeai_token=None,
            queue="default", poll_interval=0.01, timeout=10, max_concurrent_jobs=2, guild_id=None,
            generation_defaults={"negative_prompt": "blurry", "width": 1024, "height": 1024, "seed": -1,
                                 "steps": 30, "cfg_scale": 7.0},
            ollama_enabled=enabled, ollama_url=ollama_url or str(servers[1].make_url("")),
            ollama_model="llama3.1:8b" if enabled else None, ollama_timeout=5,
        )
        bot = create_bot(settings)
        started.append(bot)
        return bot, invoke, ollama

    yield make
    for item in started:
        if isinstance(item, TestServer):
            await item.close()
        else:
            await item.invoke.close()
            if item.ollama is not None:
                await item.ollama.close()


async def dream(bot, prompt, **options):
    interaction = FakeInteraction()
    await bot.tree.get_command("dream").callback(interaction, prompt=prompt, **options)
    return interaction


def button(view, label):
    return next(item for item in view.children if getattr(item, "label", None) == label)


# --- /dream ----------------------------------------------------------------------

@pytest.mark.parametrize("base", ["sdxl", "sd-1", "flux", "flux2"])
async def test_dream_sends_enhanced_prompt_but_shows_original(harness, base):
    bot, invoke, ollama = await harness(base=base)
    interaction = await dream(bot, "a clockwork fox")

    assert ollama.received == ["a clockwork fox"]
    assert invoke.prompts() == [enhanced("a clockwork fox")]
    result = interaction.result()
    assert result["content"] is None
    assert result["embed"].description == "a clockwork fox"
    assert "Test Model" in result["embed"].footer.text
    assert result["attachments"][0].filename.endswith(".png")
    assert isinstance(result["view"], ResultView)
    assert result["view"].state.prompt == "a clockwork fox"
    assert "ENHANCED" not in interaction.everything_shown()


async def test_negative_prompt_is_never_enhanced(harness):
    bot, invoke, ollama = await harness()
    await dream(bot, "a lighthouse", negative_prompt="text, watermark")
    assert invoke.graphs[0]["nodes"]["negative"]["prompt"] == "text, watermark"
    assert ollama.received == ["a lighthouse"]


async def test_disabled_enhancement_never_contacts_ollama(harness):
    bot, invoke, ollama = await harness(enabled=False)
    interaction = await dream(bot, "a clockwork fox")
    assert bot.ollama is None
    assert ollama.received == []
    assert invoke.prompts() == ["a clockwork fox"]
    assert interaction.result()["embed"].description == "a clockwork fox"


@pytest.mark.parametrize("mode", ["error", "garbage"])
async def test_ollama_failures_fall_back_to_the_original_prompt(harness, mode):
    bot, invoke, ollama = await harness(ollama_mode=mode)
    interaction = await dream(bot, "a clockwork fox")
    assert ollama.received == ["a clockwork fox"]
    assert invoke.prompts() == ["a clockwork fox"]
    assert interaction.result()["embed"].description == "a clockwork fox"
    assert "Oops" not in interaction.everything_shown()


async def test_unreachable_ollama_falls_back_to_the_original_prompt(harness):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
    bot, invoke, _ = await harness(ollama_url=dead)
    interaction = await dream(bot, "a clockwork fox")
    assert invoke.prompts() == ["a clockwork fox"]
    assert interaction.result()["embed"].description == "a clockwork fox"


async def test_long_prompt_is_shown_in_full_and_enhanced_from_the_original(harness):
    bot, invoke, ollama = await harness()
    prompt = "a fox " * 250  # 1500 characters, the slash-command maximum
    interaction = await dream(bot, prompt)
    assert ollama.received == [prompt]
    assert interaction.result()["embed"].description == prompt


async def test_progress_bar_is_shown_before_the_result(harness):
    bot, _, _ = await harness()
    interaction = await dream(bot, "a clockwork fox")
    assert "Working…" in interaction.shown[0]["content"]


# --- result buttons ----------------------------------------------------------------

async def test_refresh_re_enhances_the_original_not_the_enhanced_prompt(harness):
    bot, invoke, ollama = await harness()
    first = await dream(bot, "a clockwork fox")
    view = first.result()["view"]

    click = FakeInteraction()
    await button(view, "Refresh").callback(click)
    await button(click.result()["view"], "Refresh").callback(again := FakeInteraction())

    assert ollama.received == ["a clockwork fox"] * 3
    assert invoke.prompts() == [enhanced("a clockwork fox")] * 3
    assert click.result()["embed"].description == "a clockwork fox"
    assert again.result()["embed"].description == "a clockwork fox"
    assert view.state.prompt == "a clockwork fox"


async def test_edit_prompt_shows_the_original_and_enhances_the_new_text(harness):
    bot, invoke, ollama = await harness()
    view = (await dream(bot, "a clockwork fox")).result()["view"]

    opener = FakeInteraction()
    await button(view, "Edit prompt").callback(opener)
    modal = opener.response.modal
    assert isinstance(modal, PromptModal)
    assert modal.prompt.default == "a clockwork fox"

    modal.prompt._value = "a porcelain owl"
    modal.negative._value = "blurry"
    submit = FakeInteraction()
    await modal.on_submit(submit)

    assert ollama.received == ["a clockwork fox", "a porcelain owl"]
    assert invoke.prompts()[-1] == enhanced("a porcelain owl")
    assert submit.result()["embed"].description == "a porcelain owl"
    assert "ENHANCED" not in submit.everything_shown()


async def test_random_prompt_is_enhanced_and_shown_unenhanced(harness, monkeypatch):
    bot, invoke, ollama = await harness()
    view = (await dream(bot, "a clockwork fox")).result()["view"]
    monkeypatch.setattr(bot_module, "random_prompt", lambda: "a floating city, oil painting")

    click = FakeInteraction()
    await button(view, "Random").callback(click)

    assert ollama.received[-1] == "a floating city, oil painting"
    assert invoke.prompts()[-1] == enhanced("a floating city, oil painting")
    assert click.result()["embed"].description == "a floating city, oil painting"


async def test_tweak_generate_re_enhances_the_original(harness):
    bot, invoke, ollama = await harness()
    view = (await dream(bot, "a clockwork fox")).result()["view"]
    models = await bot.invoke.get_models()
    samplers = await bot.invoke.get_samplers()
    tweak = TweakView(view, models, samplers)

    click = FakeInteraction()
    await button(tweak, "Generate").callback(click)

    assert ollama.received == ["a clockwork fox", "a clockwork fox"]
    assert invoke.prompts()[-1] == enhanced("a clockwork fox")
    assert click.result()["embed"].description == "a clockwork fox"


async def test_simultaneous_generations_keep_their_own_prompts(harness):
    bot, invoke, ollama = await harness()
    prompts = [f"prompt number {i}" for i in range(6)]
    interactions = await asyncio.gather(*(dream(bot, prompt) for prompt in prompts))

    assert sorted(ollama.received) == sorted(prompts)
    assert sorted(invoke.prompts()) == sorted(enhanced(prompt) for prompt in prompts)
    for prompt, interaction in zip(prompts, interactions):
        assert interaction.result()["embed"].description == prompt


async def test_bot_close_shuts_both_clients(harness):
    bot, _, _ = await harness()
    await dream(bot, "a clockwork fox")
    await bot.close()
    assert bot.invoke.session.closed
    assert bot.ollama.session.closed
