# Discord InvokeAI bot

A Discord `/dream` bot that generates images with an InvokeAI server. Users do not create or export workflows: the bot builds the required execution graph internally from the selected installed model.

## Setup

1. Create a Discord application in the Developer Portal, add a bot, and invite it with the `bot` and `applications.commands` scopes.
2. Make sure InvokeAI is running and has a supported main model installed.
3. Copy `config.example.json` to `config.json` and set the Discord token and InvokeAI connection.

## Run as native Python

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python run.py
```

On Windows, activate with `.venv\Scripts\activate`. No Docker installation or `workflow.json` is needed.

## Run with Docker

```bash
cp config.example.json config.json
docker compose up --build -d
```

## Models and configuration

Set `generation.model` to an installed InvokeAI main-model name or key. When it is `null`, the bot chooses the first supported installed model. Built-in text-to-image generation supports:

- Stable Diffusion 1.x and 2.x
- SDXL
- FLUX.1
- FLUX.2 Klein

FLUX.1 installations may use separate T5, CLIP, and VAE models. Configure their installed names or keys as `generation.t5_encoder`, `generation.clip_encoder`, and `generation.vae`. FLUX.2 may require separate `generation.text_encoder` and `generation.vae` values. References are resolved against InvokeAI's installed model list.

The core generation settings control negative prompt, dimensions, seed, sampler, steps, and CFG/guidance. Slash-command values override configured defaults. A seed of `-1` chooses a random seed.

Each result includes Refresh, Edit prompt, Random, Tweak, and Delete controls. The Tweak panel loads model and LoRA choices from the connected InvokeAI installation and scheduler choices from InvokeAI's API schema; dropdown pages expose installations with more than 25 options. Refreshing or changing a result creates a new message and preserves the original image; only Delete removes it. Only the requester may use the controls. A yellow face alternates between closed and wide-open mouth frames as it munches through the “Working…” progress bar. It moves quickly at first and slows as it nears the end, so it never jumps back to the start however long a generation takes; very occasionally a ghost gives chase a few cells behind. Result metadata uses icons for model, sampler, seed, dimensions, steps, and CFG/guidance. Discord receives only a generic failure message; full errors are logged to the bot console.

## Prompt enhancement with Ollama (optional)

The bot can pass each prompt through a remote [Ollama](https://ollama.com) server before generating. Ollama rewrites a short or vague prompt into a detailed, art-directed one, and the bot sends that rewritten prompt to InvokeAI. Discord only ever shows the user's original prompt: the result embed and the Edit prompt dialog both use the original text. The enhanced prompt is written to the bot's console log.

```json
"ollama": {
  "enabled": true,
  "url": "http://gpu-box:11434",
  "model": "llama3.1:8b",
  "timeout_seconds": 60
}
```

- `enabled`: turns enhancement on or off. The default is `false`.
- `url`: the Ollama server's base URL.
- `model`: an Ollama model that is already pulled on that server, for example `ollama pull llama3.1:8b`. This is required when enhancement is enabled.
- `timeout_seconds`: how long to wait for Ollama before giving up.

The original prompt is sent as the user message, with a built-in system prompt that tells the model to preserve the idea and return only the enhanced prompt. Each Refresh and Tweak requests a new enhancement, so repeated generations get varied interpretations. If Ollama is unreachable, times out, or returns an error, the bot logs a warning and generates from the original prompt. The negative prompt is never changed.

If Ollama runs on another machine, start it with `OLLAMA_HOST=0.0.0.0` so it listens on the network. It has no authentication, so keep it on a trusted network.

Global slash-command registration can take up to an hour. Set `discord.guild_id` to a development server ID for immediate guild-scoped registration.

## Environment overrides

Connection and runtime settings have optional environment overrides in `.env.example`. Ollama settings can be overridden with `OLLAMA_ENABLED`, `OLLAMA_URL`, `OLLAMA_MODEL`, and `OLLAMA_TIMEOUT_SECONDS`. Environment values take precedence over `config.json`; `.env` is not required.

## Security

Do not commit `.env` or `config.json`, because either may contain tokens. Both are ignored by Git. Prefer a private network or authenticated reverse proxy when InvokeAI is on another machine.
