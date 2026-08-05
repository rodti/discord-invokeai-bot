from invokeai_discord_bot.generation_graph import build_generation_graph


VALUES = {
    "prompt": "a lighthouse",
    "negative_prompt": "blurry",
    "width": 1024,
    "height": 1024,
    "seed": 42,
    "steps": 20,
    "cfg_scale": 4.0,
    "sampler": "euler",
}


def model(key: str, base: str):
    return {"key": key, "hash": f"hash-{key}", "name": f"Model {key}", "base": base, "type": "main"}


def test_builds_sdxl_graph():
    graph = build_generation_graph(VALUES, model("m1", "sdxl"))
    assert graph["nodes"]["loader"]["type"] == "sdxl_model_loader"
    assert graph["nodes"]["positive"]["type"] == "sdxl_compel_prompt"
    assert graph["nodes"]["loader"]["model"] == {
        "key": "m1", "hash": "hash-m1", "name": "Model m1", "base": "sdxl", "type": "main"
    }


def test_builds_flux1_graph():
    graph = build_generation_graph(VALUES, model("m2", "flux"))
    assert graph["nodes"]["loader"]["type"] == "flux_model_loader"
    assert graph["nodes"]["denoise"]["type"] == "flux_denoise"
    assert "negative" not in graph["nodes"]


def test_builds_flux2_graph():
    graph = build_generation_graph(VALUES, model("m3", "flux2"))
    assert graph["nodes"]["loader"]["type"] == "flux2_model_loader"
    assert graph["nodes"]["decode"]["type"] == "flux2_vae_decode"
