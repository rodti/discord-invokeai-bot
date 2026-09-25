"""Ollama client behaviour against a local server that mimics Ollama's /api/chat."""

import asyncio
import socket

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from invokeai_discord_bot.ollama import ENHANCER_SYSTEM_PROMPT, OllamaClient, OllamaError


def ollama_reply(content, **extra):
    """A response shaped like Ollama's non-streaming /api/chat reply."""
    return {
        "model": "llama3.1:8b",
        "created_at": "2026-09-25T08:00:00.000000Z",
        "message": {"role": "assistant", "content": content, **extra},
        "done_reason": "stop",
        "done": True,
        "total_duration": 1_234_567_890,
        "load_duration": 12_345_678,
        "prompt_eval_count": 512,
        "eval_count": 128,
    }


@pytest.fixture
async def serve():
    servers = []

    async def start(handler, path="/api/chat"):
        app = web.Application()
        app.router.add_post(path, handler)
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        return str(server.make_url(""))

    yield start
    for server in servers:
        await server.close()


async def enhance(url, prompt="a fox", model="llama3.1:8b", timeout=5):
    client = OllamaClient(url, model, timeout)
    try:
        return await client.enhance(prompt)
    finally:
        await client.close()


# --- request shape -----------------------------------------------------------

async def test_request_matches_ollama_chat_api(serve):
    seen = {}

    async def handler(request):
        seen["path"] = request.path
        seen["content_type"] = request.content_type
        seen["body"] = await request.json()
        return web.json_response(ollama_reply("A fox in a lantern-lit forest"))

    url = await serve(handler)
    assert await enhance(url + "/", "a fox") == "A fox in a lantern-lit forest"
    assert seen["path"] == "/api/chat"  # trailing slash in the configured URL is harmless
    assert seen["content_type"] == "application/json"
    body = seen["body"]
    assert body["model"] == "llama3.1:8b"
    assert body["stream"] is False
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == ENHANCER_SYSTEM_PROMPT
    assert body["messages"][1]["content"] == "a fox"


def test_system_prompt_is_the_requested_text_verbatim():
    assert ENHANCER_SYSTEM_PROMPT.startswith("You are an expert image-generation prompt enhancer.\n\n")
    assert ENHANCER_SYSTEM_PROMPT.endswith(
        "Return ONLY the finished enhanced image-generation prompt as one continuous paragraph."
    )
    assert "- Appropriate rendering characteristics such as realistic materials, subsurface scattering" in ENHANCER_SYSTEM_PROMPT
    assert 'meaningless phrases such as repeated "masterpiece", "best quality", or resolution tokens.' in ENHANCER_SYSTEM_PROMPT
    assert ENHANCER_SYSTEM_PROMPT.count("Do not ") == 8


async def test_unicode_and_long_prompts_are_passed_through_untouched(serve):
    seen = {}

    async def handler(request):
        seen.update(await request.json())
        return web.json_response(ollama_reply("ok"))

    url = await serve(handler)
    prompt = "un café flotante ☕ über den Wolken — 雲の上 " * 40
    await enhance(url, prompt)
    assert seen["messages"][1]["content"] == prompt


# --- response handling -------------------------------------------------------

async def test_thinking_field_from_reasoning_models_is_ignored(serve):
    async def handler(request):
        return web.json_response(ollama_reply("A clockwork fox", thinking="Let me think about foxes..."))

    assert await enhance(await serve(handler)) == "A clockwork fox"


async def test_inline_think_blocks_quotes_and_line_breaks_are_cleaned(serve):
    async def handler(request):
        return web.json_response(ollama_reply('<think>\nplan\n</think>\n\n"A clockwork fox,\nlit by candles."\n'))

    assert await enhance(await serve(handler)) == "A clockwork fox, lit by candles."


async def test_quotes_inside_the_prompt_are_kept(serve):
    async def handler(request):
        return web.json_response(ollama_reply('A shop sign reading "OPEN" above a fox'))

    assert await enhance(await serve(handler)) == 'A shop sign reading "OPEN" above a fox'


@pytest.mark.parametrize(
    "payload",
    [
        ollama_reply(""),
        ollama_reply("   \n  "),
        ollama_reply("<think>only thinking</think>"),
        {"done": True},
        {"message": None},
        {"message": "not an object"},
        {"message": {"role": "assistant"}},
        ["not", "an", "object"],
        None,
    ],
    ids=["empty", "whitespace", "only-thinking", "no-message", "null-message",
         "string-message", "no-content", "list-body", "null-body"],
)
async def test_unusable_responses_raise_ollama_error(serve, payload):
    async def handler(request):
        return web.json_response(payload)

    with pytest.raises(OllamaError):
        await enhance(await serve(handler))


async def test_non_json_body_raises_ollama_error(serve):
    async def handler(request):
        return web.Response(text="<html>proxy error</html>", content_type="text/html")

    with pytest.raises(OllamaError):
        await enhance(await serve(handler))


async def test_model_not_found_raises_ollama_error_with_detail(serve):
    async def handler(request):
        return web.json_response({"error": "model \"nope\" not found, try pulling it first"}, status=404)

    with pytest.raises(OllamaError, match="HTTP 404.*not found"):
        await enhance(await serve(handler), model="nope")


async def test_server_error_raises_ollama_error(serve):
    async def handler(request):
        return web.Response(status=500, text="boom")

    with pytest.raises(OllamaError, match="HTTP 500"):
        await enhance(await serve(handler))


async def test_slow_server_times_out_as_ollama_error(serve):
    async def handler(request):
        await asyncio.sleep(3)
        return web.json_response(ollama_reply("too late"))

    with pytest.raises(OllamaError):
        await enhance(await serve(handler), timeout=0.3)


async def test_unreachable_server_raises_ollama_error():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with pytest.raises(OllamaError, match="Cannot communicate"):
        await enhance(f"http://127.0.0.1:{port}")


async def test_unresolvable_host_raises_ollama_error():
    with pytest.raises(OllamaError):
        await enhance("http://ollama.invalid:11434", timeout=5)


# --- client lifecycle --------------------------------------------------------

async def test_client_reuses_one_session_for_several_requests(serve):
    calls = []

    async def handler(request):
        body = await request.json()
        calls.append(body["messages"][1]["content"])
        return web.json_response(ollama_reply(f"enhanced {body['messages'][1]['content']}"))

    client = OllamaClient(await serve(handler), "m")
    try:
        results = await asyncio.gather(*(client.enhance(f"prompt {i}") for i in range(5)))
        session = client.session
        await client.enhance("again")
        assert client.session is session
    finally:
        await client.close()
    assert results == [f"enhanced prompt {i}" for i in range(5)]
    assert sorted(calls) == sorted([f"prompt {i}" for i in range(5)] + ["again"])
    assert client.session.closed


async def test_close_without_any_request_is_safe():
    await OllamaClient("http://localhost:11434", "m").close()
