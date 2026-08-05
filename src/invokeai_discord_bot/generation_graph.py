from __future__ import annotations

import uuid
from typing import Any

class GenerationGraphError(ValueError):
    pass


def _edge(source: str, source_field: str, destination: str, destination_field: str) -> dict[str, Any]:
    return {
        "source": {"node_id": source, "field": source_field},
        "destination": {"node_id": destination, "field": destination_field},
    }


def build_generation_graph(values: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    """Build a standard InvokeAI text-to-image graph for SD 1/2 or SDXL."""
    base = str(model.get("base", model.get("base_model", ""))).lower()
    model_field = _model_field(model)
    common = {"is_intermediate": False, "use_cache": True}
    nodes: dict[str, dict[str, Any]] = {
        "noise": {"id": "noise", "type": "noise", "width": values["width"], "height": values["height"], "seed": values["seed"], **common},
        "denoise": {
            "id": "denoise", "type": "denoise_latents", "steps": values["steps"],
            "cfg_scale": values["cfg_scale"], "scheduler": values.get("sampler") or "euler",
            "denoising_start": 0, "denoising_end": 1, **common,
        },
        "decode": {"id": "decode", "type": "l2i", **common},
    }
    edges = [
        _edge("noise", "noise", "denoise", "noise"),
        _edge("denoise", "latents", "decode", "latents"),
        _edge("loader", "unet", "denoise", "unet"),
        _edge("loader", "vae", "decode", "vae"),
        _edge("positive", "conditioning", "denoise", "positive_conditioning"),
        _edge("negative", "conditioning", "denoise", "negative_conditioning"),
    ]
    if base in {"sdxl", "sdxl-refiner"}:
        nodes.update({
            "loader": {"id": "loader", "type": "sdxl_model_loader", "model": model_field, **common},
            "positive": {"id": "positive", "type": "sdxl_compel_prompt", "prompt": values["prompt"], "style": "", **common},
            "negative": {"id": "negative", "type": "sdxl_compel_prompt", "prompt": values["negative_prompt"], "style": "", **common},
        })
        for prompt_node in ("positive", "negative"):
            edges.extend([
                _edge("loader", "clip", prompt_node, "clip"),
                _edge("loader", "clip2", prompt_node, "clip2"),
            ])
    elif base in {"sd-1", "sd-2", "sd1", "sd2"}:
        nodes.update({
            "loader": {"id": "loader", "type": "main_model_loader", "model": model_field, **common},
            "positive": {"id": "positive", "type": "compel", "prompt": values["prompt"], **common},
            "negative": {"id": "negative", "type": "compel", "prompt": values["negative_prompt"], **common},
        })
        edges.extend([
            _edge("loader", "clip", "positive", "clip"),
            _edge("loader", "clip", "negative", "clip"),
        ])
    elif base in {"flux", "flux-1", "flux1"}:
        loader = {"id": "loader", "type": "flux_model_loader", "model": model_field, **common}
        for setting, field_name in (("t5_encoder", "t5_encoder_model"), ("clip_encoder", "clip_embed_model"), ("vae", "vae_model")):
            if values.get(setting):
                loader[field_name] = _model_field(values[setting])
        nodes = {
            "loader": loader,
            "positive": {"id": "positive", "type": "flux_text_encoder", "prompt": values["prompt"], **common},
            "denoise": {
                "id": "denoise", "type": "flux_denoise", "width": values["width"], "height": values["height"],
                "num_steps": values["steps"], "guidance": values["cfg_scale"], "seed": values["seed"], **common,
            },
            "decode": {"id": "decode", "type": "flux_vae_decode", **common},
        }
        edges = [
            _edge("loader", "transformer", "denoise", "transformer"),
            _edge("loader", "clip", "positive", "clip"),
            _edge("loader", "t5_encoder", "positive", "t5_encoder"),
            _edge("positive", "conditioning", "denoise", "positive_conditioning"),
            _edge("denoise", "latents", "decode", "latents"),
            _edge("loader", "vae", "decode", "vae"),
        ]
    elif base in {"flux2", "flux-2", "flux.2", "flux2-klein"}:
        loader = {"id": "loader", "type": "flux2_model_loader", "model": model_field, **common}
        for setting, field_name in (("text_encoder", "text_encoder_model"), ("vae", "vae_model")):
            if values.get(setting):
                loader[field_name] = _model_field(values[setting])
        nodes = {
            "loader": loader,
            "positive": {"id": "positive", "type": "flux2_text_encoder", "prompt": values["prompt"], **common},
            "denoise": {
                "id": "denoise", "type": "flux2_denoise", "width": values["width"], "height": values["height"],
                "steps": values["steps"], "cfg_scale": values["cfg_scale"], "seed": values["seed"], **common,
            },
            "decode": {"id": "decode", "type": "flux2_vae_decode", **common},
        }
        edges = [
            _edge("loader", "transformer", "denoise", "transformer"),
            _edge("loader", "text_encoder", "positive", "text_encoder"),
            _edge("positive", "conditioning", "denoise", "positive_conditioning"),
            _edge("denoise", "latents", "decode", "latents"),
            _edge("loader", "vae", "decode", "vae"),
        ]
    else:
        raise GenerationGraphError(
            f"Built-in generation does not support model base '{base or 'unknown'}'. "
            "Choose an installed SD 1.x, SD 2.x, SDXL, FLUX.1, or FLUX.2 main model."
        )
    return {"id": str(uuid.uuid4()), "nodes": nodes, "edges": edges}


def _model_field(model: Any) -> Any:
    if isinstance(model, dict):
        normalized = {
            "key": model.get("key"),
            "hash": model.get("hash"),
            "name": model.get("name"),
            "base": model.get("base", model.get("base_model")),
            "type": model.get("type", model.get("model_type")),
        }
        missing = [name for name, value in normalized.items() if value in (None, "")]
        if missing:
            raise GenerationGraphError(
                "InvokeAI model metadata is missing required fields: " + ", ".join(missing)
            )
        return normalized
    return model
