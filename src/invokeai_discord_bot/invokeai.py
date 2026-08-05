from __future__ import annotations

import asyncio
from typing import Any

import aiohttp


class InvokeAIError(RuntimeError):
    pass


class InvokeAIClient:
    def __init__(self, base_url: str, token: str | None, queue: str = "default") -> None:
        self.base_url = base_url
        self.queue = queue
        self.token = token
        self.session: aiohttp.ClientSession | None = None

    def _session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.token}"} if self.token else None,
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self.session

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()

    async def get_models(self) -> list[dict[str, Any]]:
        response = await self._json("GET", "/api/v2/models/")
        models = response if isinstance(response, list) else response.get("models", response.get("items", response))
        if isinstance(models, dict):
            models = list(models.values())
        if not isinstance(models, list):
            raise InvokeAIError("InvokeAI returned an unexpected model list")
        return [m for m in models if isinstance(m, dict)]

    async def get_samplers(self) -> list[str]:
        fallback = [
            "ddim", "ddpm", "deis", "euler", "euler_a", "heun", "kdpm_2", "kdpm_2_a",
            "lms", "pndm", "unipc", "dpmpp_2s", "dpmpp_2s_k", "dpmpp_2m", "dpmpp_2m_k",
            "dpmpp_2m_sde", "dpmpp_2m_sde_k", "dpmpp_sde", "dpmpp_sde_k",
        ]
        try:
            schema = await self._json("GET", "/openapi.json")
            found: set[str] = set()
            for definition in schema.get("components", {}).get("schemas", {}).values():
                scheduler = definition.get("properties", {}).get("scheduler", {}) if isinstance(definition, dict) else {}
                found.update(str(value) for value in scheduler.get("enum", []) if value)
            return sorted(found) or fallback
        except InvokeAIError:
            return fallback

    async def resolve_model(self, requested: Any = None, main_only: bool = True) -> dict[str, Any]:
        all_models = await self.get_models()
        main_models = [m for m in all_models if str(m.get("type", "main")).lower() == "main"]
        candidates = main_models if main_only else all_models
        if isinstance(requested, dict):
            key = requested.get("key")
            match = next((m for m in candidates if m.get("key") == key), None)
            return match or requested
        if requested not in (None, ""):
            needle = str(requested).casefold()
            match = next((m for m in candidates if needle in {str(m.get("key", "")).casefold(), str(m.get("name", "")).casefold()}), None)
            if not match:
                raise InvokeAIError(f"Installed main model not found: {requested}")
            return match
        supported = {"sd-1", "sd-2", "sd1", "sd2", "sdxl", "sdxl-refiner", "flux", "flux-1", "flux1", "flux2", "flux-2", "flux.2", "flux2-klein"}
        match = next((m for m in main_models if str(m.get("base", m.get("base_model", ""))).lower() in supported), None)
        if not match:
            raise InvokeAIError("No supported installed main generation model was found")
        return match

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            async with self._session().request(method, self.base_url + path, **kwargs) as response:
                if response.status >= 400:
                    body = (await response.text())[:1000]
                    raise InvokeAIError(f"InvokeAI returned HTTP {response.status}: {body}")
                return await response.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise InvokeAIError(f"Cannot communicate with InvokeAI: {exc}") from exc

    async def generate(self, graph: dict[str, Any], poll_interval: float, timeout: float) -> tuple[bytes, str]:
        payload = {"batch": {"graph": graph, "runs": 1, "origin": "discord-bot"}}
        queued = await self._json("POST", f"/api/v1/queue/{self.queue}/enqueue_batch", json=payload)
        try:
            item_id = queued["item_ids"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise InvokeAIError("InvokeAI did not return a queue item ID") from exc

        async def wait_for_result() -> str:
            while True:
                item = await self._json("GET", f"/api/v1/queue/{self.queue}/i/{item_id}")
                status = item.get("status")
                if status == "completed":
                    image_name = find_image_name(item.get("session", {}).get("results", {}))
                    if not image_name:
                        raise InvokeAIError("Generation completed but returned no image")
                    return image_name
                if status in {"failed", "canceled"}:
                    message = item.get("error_message") or f"generation was {status}"
                    raise InvokeAIError(str(message))
                await asyncio.sleep(poll_interval)

        try:
            image_name = await asyncio.wait_for(wait_for_result(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise InvokeAIError(f"generation timed out after {timeout:g} seconds") from exc

        path = f"/api/v1/images/i/{image_name}/full"
        try:
            async with self._session().get(self.base_url + path) as response:
                if response.status >= 400:
                    raise InvokeAIError(f"image download returned HTTP {response.status}")
                return await response.read(), image_name
        except aiohttp.ClientError as exc:
            raise InvokeAIError(f"cannot download generated image: {exc}") from exc


def find_image_name(value: Any) -> str | None:
    if isinstance(value, dict):
        image = value.get("image")
        if isinstance(image, dict) and isinstance(image.get("image_name"), str):
            return image["image_name"]
        if isinstance(value.get("image_name"), str):
            return value["image_name"]
        for child in value.values():
            found = find_image_name(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_image_name(child)
            if found:
                return found
    return None
