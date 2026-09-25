import json
from pathlib import Path

import pytest

from invokeai_discord_bot.config import Settings

ENV_NAMES = (
    "BOT_CONFIG", "DISCORD_TOKEN", "DISCORD_GUILD_ID", "INVOKEAI_URL", "INVOKEAI_TOKEN", "INVOKEAI_QUEUE",
    "POLL_INTERVAL_SECONDS", "GENERATION_TIMEOUT_SECONDS", "MAX_CONCURRENT_JOBS",
    "OLLAMA_ENABLED", "OLLAMA_URL", "OLLAMA_MODEL", "OLLAMA_TIMEOUT_SECONDS",
)


@pytest.fixture
def load(tmp_path, monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)

    def _load(config=None, **env):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"discord": {"token": "t"}} if config is None else config))
        monkeypatch.setenv("BOT_CONFIG", str(path))
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return Settings.from_env()

    return _load


def with_ollama(**ollama):
    return {"discord": {"token": "t"}, "ollama": ollama}


def test_ollama_is_off_when_the_section_is_missing(load):
    settings = load()
    assert settings.ollama_enabled is False
    assert settings.ollama_model is None
    assert settings.ollama_url == "http://localhost:11434"
    assert settings.ollama_timeout == 60


def test_disabled_section_does_not_need_a_model(load):
    assert load(with_ollama(enabled=False, model=None)).ollama_enabled is False


def test_full_section_is_read(load):
    settings = load(with_ollama(enabled=True, url="http://gpu-box:11434/", model="qwen3:8b", timeout_seconds=30))
    assert (settings.ollama_enabled, settings.ollama_url, settings.ollama_model, settings.ollama_timeout) == (
        True, "http://gpu-box:11434", "qwen3:8b", 30
    )


@pytest.mark.parametrize("value, expected", [
    (True, True), (False, False), ("true", True), ("True", True), ("yes", True), ("on", True), ("1", True),
    ("false", False), ("no", False), ("off", False), ("0", False), ("", False), (None, False),
])
def test_enabled_accepts_common_spellings(load, value, expected):
    assert load(with_ollama(enabled=value, model="m")).ollama_enabled is expected


@pytest.mark.parametrize("value", ["maybe", "enabled", 2, "tru"])
def test_enabled_rejects_unclear_values(load, value):
    with pytest.raises(ValueError, match="OLLAMA_ENABLED"):
        load(with_ollama(enabled=value, model="m"))


@pytest.mark.parametrize("model", [None, "", "   "])
def test_enabled_without_a_model_is_a_startup_error(load, model):
    with pytest.raises(ValueError, match="ollama.model"):
        load(with_ollama(enabled=True, model=model))


def test_model_name_is_trimmed(load):
    assert load(with_ollama(enabled=True, model="  llama3.1:8b  ")).ollama_model == "llama3.1:8b"


def test_timeout_must_be_a_number(load):
    with pytest.raises(ValueError, match="OLLAMA_TIMEOUT_SECONDS"):
        load(with_ollama(enabled=True, model="m", timeout_seconds="soon"))


def test_section_must_be_an_object(load):
    with pytest.raises(ValueError, match="ollama"):
        load({"discord": {"token": "t"}, "ollama": True})


def test_environment_overrides_the_file(load):
    settings = load(
        with_ollama(enabled=False, url="http://file:11434", model="file-model", timeout_seconds=10),
        OLLAMA_ENABLED="true", OLLAMA_URL="http://env:11434", OLLAMA_MODEL="env-model", OLLAMA_TIMEOUT_SECONDS="45",
    )
    assert (settings.ollama_enabled, settings.ollama_url, settings.ollama_model, settings.ollama_timeout) == (
        True, "http://env:11434", "env-model", 45
    )


def test_environment_can_switch_enhancement_off(load):
    assert load(with_ollama(enabled=True, model="m"), OLLAMA_ENABLED="false").ollama_enabled is False


def test_environment_alone_can_configure_ollama(load):
    settings = load(None, OLLAMA_ENABLED="1", OLLAMA_MODEL="env-model")
    assert settings.ollama_enabled and settings.ollama_model == "env-model"


def test_example_config_loads_and_keeps_enhancement_off(load):
    example = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
    settings = load(example)
    assert settings.ollama_enabled is False
    assert settings.ollama_model == "llama3.1:8b"
    assert settings.ollama_url == "http://localhost:11434"


def test_example_config_works_once_enabled(load):
    example = json.loads((Path(__file__).parents[1] / "config.example.json").read_text())
    example["ollama"]["enabled"] = True
    assert load(example).ollama_enabled is True


def test_existing_settings_are_unaffected(load):
    settings = load({
        "discord": {"token": "t", "guild_id": 123},
        "invokeai": {"url": "http://invoke:9090/", "queue": "q"},
        "bot": {"max_concurrent_jobs": 3},
        "ollama": {"enabled": True, "model": "m"},
    })
    assert (settings.guild_id, settings.invokeai_url, settings.queue, settings.max_concurrent_jobs) == (
        123, "http://invoke:9090", "q", 3
    )
