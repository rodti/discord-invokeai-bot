from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


def _integer(name: str, value: object) -> int:
    try:
        return int(os.getenv(name, str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _number(name: str, value: object) -> float:
    try:
        return float(os.getenv(name, str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _section(config: dict[str, object], name: str) -> dict[str, object]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} in BOT_CONFIG must be an object")
    return value


def _value(env_name: str, section: dict[str, object], key: str, default: object = "") -> object:
    env_value = os.getenv(env_name)
    return env_value if env_value not in (None, "") else section.get(key, default)


def _boolean(name: str, value: object) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        raw = value
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "", "none", "null"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(frozen=True)
class Settings:
    discord_token: str
    invokeai_url: str
    invokeai_token: str | None
    queue: str
    poll_interval: float
    timeout: float
    max_concurrent_jobs: int
    guild_id: int | None
    generation_defaults: dict[str, object]
    ollama_enabled: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_model: str | None = None
    ollama_timeout: float = 60

    @classmethod
    def from_env(cls) -> "Settings":
        config_path = Path(os.getenv("BOT_CONFIG", "config.json")).expanduser().resolve()
        config: dict[str, object] = {}
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read BOT_CONFIG at {config_path}: {exc}") from exc
            if not isinstance(config, dict):
                raise ValueError("BOT_CONFIG must contain a JSON object")
        discord_config = _section(config, "discord")
        invoke_config = _section(config, "invokeai")
        bot_config = _section(config, "bot")
        token = _optional_text(_value("DISCORD_TOKEN", discord_config, "token", "")) or ""
        if not token:
            raise ValueError("Discord token is required in config.json or DISCORD_TOKEN")
        guild = _optional_text(_value("DISCORD_GUILD_ID", discord_config, "guild_id", "")) or ""
        generation = {
            "negative_prompt": "",
            "width": 1024,
            "height": 1024,
            "seed": -1,
            "steps": 30,
            "cfg_scale": 7.0,
            "model": None,
            "lora": None,
            "sampler": None,
            "strength": None,
            "upscaling": None,
            "t5_encoder": None,
            "clip_encoder": None,
            "text_encoder": None,
            "vae": None,
        }
        configured_generation = _section(config, "generation")
        generation.update(configured_generation)
        _validate_generation(generation)
        ollama_config = _section(config, "ollama")
        ollama_enabled = _boolean("OLLAMA_ENABLED", ollama_config.get("enabled", False))
        ollama_model = _optional_text(_value("OLLAMA_MODEL", ollama_config, "model", ""))
        if ollama_enabled and not ollama_model:
            raise ValueError("ollama.model is required when Ollama prompt enhancement is enabled")
        return cls(
            discord_token=token,
            invokeai_url=str(_value("INVOKEAI_URL", invoke_config, "url", "http://localhost:9090")).rstrip("/"),
            invokeai_token=_optional_text(_value("INVOKEAI_TOKEN", invoke_config, "token", "")),
            queue=str(_value("INVOKEAI_QUEUE", invoke_config, "queue", "default")),
            poll_interval=_number("POLL_INTERVAL_SECONDS", bot_config.get("poll_interval_seconds", 1)),
            timeout=_number("GENERATION_TIMEOUT_SECONDS", bot_config.get("generation_timeout_seconds", 600)),
            max_concurrent_jobs=_integer("MAX_CONCURRENT_JOBS", bot_config.get("max_concurrent_jobs", 2)),
            guild_id=int(guild) if guild else None,
            generation_defaults=generation,
            ollama_enabled=ollama_enabled,
            ollama_url=str(_value("OLLAMA_URL", ollama_config, "url", "http://localhost:11434")).rstrip("/"),
            ollama_model=ollama_model,
            ollama_timeout=_number("OLLAMA_TIMEOUT_SECONDS", ollama_config.get("timeout_seconds", 60)),
        )


def _validate_generation(values: dict[str, object]) -> None:
    try:
        width, height = int(values["width"]), int(values["height"])
        steps, scale = int(values["steps"]), float(values["cfg_scale"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("generation width, height, steps, and cfg_scale must be numeric") from exc
    if not (256 <= width <= 2048 and 256 <= height <= 2048):
        raise ValueError("default generation width and height must be between 256 and 2048")
    if not (1 <= steps <= 100 and 0 <= scale <= 30):
        raise ValueError("default generation steps must be 1-100 and cfg_scale must be 0-30")
