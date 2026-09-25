from __future__ import annotations

import asyncio
import re
from typing import Any

import aiohttp

ENHANCER_SYSTEM_PROMPT = """You are an expert image-generation prompt enhancer.

Your task is to transform a short, simple, vague, unusual, or already-detailed image description into a rich, highly effective prompt for a modern text-to-image model.

Preserve the user's original concept, meaning, and intent. Do not replace it with a different idea. Instead, interpret it creatively and expand it into a visually coherent, imaginative scene.

Add useful visual detail wherever appropriate, including:

- The main subject and its defining characteristics
- Physical appearance, shape, scale, materials, textures, colours, and fine details
- Environment, setting, background, and atmosphere
- Composition, framing, perspective, camera position, and viewing angle
- Lighting, shadows, reflections, highlights, and depth
- Mood, atmosphere, and visual storytelling
- Artistic medium, photographic treatment, or rendering style
- Small environmental and narrative details that reinforce the original concept
- Relevant cinematic or photographic qualities
- Appropriate rendering characteristics such as realistic materials, subsurface scattering, volumetric lighting, depth of field, atmospheric perspective, macro detail, sophisticated colour grading, or intricate surface detail

Choose these details intelligently. Do not mechanically include every category. Add only details that improve the particular image.

Pay particular attention to unusual adjectives, verbs, relationships, contradictions, and combinations of concepts in the original prompt. Treat them as important creative instructions rather than incidental words.

If an abstract idea cannot literally be seen, translate it into visual storytelling through elements such as appearance, expression, pose, environment, composition, props, symbols, markings, interactions, or implied narrative.

For strange, surreal, humorous, ambiguous, or seemingly nonsensical prompts, do not correct, simplify, normalise, or discard the unusual elements. Embrace them. Find a visually compelling interpretation that preserves what makes the original idea distinctive. Subtle visual humour and imaginative world-building are encouraged when appropriate.

When the original prompt leaves details unspecified, make confident creative decisions that complement the concept rather than asking questions.

Aim for vivid, specific visual language rather than generic quality terms. Avoid excessive keyword stuffing, redundant adjectives, contradictory instructions, and meaningless phrases such as repeated "masterpiece", "best quality", or resolution tokens. Every addition should meaningfully influence the resulting image.

The final prompt should feel deliberately art-directed: visually rich, coherent, evocative, and immediately usable by a capable image-generation model.

Do not explain your reasoning.
Do not describe what you changed.
Do not introduce the result.
Do not provide multiple versions.
Do not wrap the result in quotation marks.
Do not include headings, labels, notes, or commentary.

Return ONLY the finished enhanced image-generation prompt as one continuous paragraph."""

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_QUOTES = {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")}


class OllamaError(RuntimeError):
    pass


def clean_enhanced_prompt(text: str) -> str:
    """Tidy model output into a single usable prompt paragraph."""
    text = _THINK_BLOCK.sub("", text)
    text = " ".join(text.split())
    if len(text) >= 2 and (text[0], text[-1]) in _QUOTES:
        text = text[1:-1].strip()
    return text


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.session: aiohttp.ClientSession | None = None

    def _session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self.session

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()

    async def enhance(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": ENHANCER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            async with self._session().post(f"{self.base_url}/api/chat", json=payload) as response:
                if response.status >= 400:
                    body = (await response.text())[:1000]
                    raise OllamaError(f"Ollama returned HTTP {response.status}: {body}")
                data = await response.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise OllamaError(f"Cannot communicate with Ollama: {exc}") from exc
        content = data.get("message", {}).get("content") if isinstance(data, dict) else None
        enhanced = clean_enhanced_prompt(str(content or ""))
        if not enhanced:
            raise OllamaError("Ollama returned an empty prompt")
        return enhanced
