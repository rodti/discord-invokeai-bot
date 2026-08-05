from invokeai_discord_bot.bot import GenerationState, pacman_frame


def test_progress_copy_is_working():
    frame = pacman_frame(3)
    assert "Working…" in frame
    assert "Generating your dream" not in frame
    assert "😮" in frame


def test_pacman_opens_and_closes_its_mouth():
    assert "😐" in pacman_frame(0)
    assert "😮" in pacman_frame(1)


def test_generation_state_can_be_copied_without_mutating_original():
    import copy

    original = GenerationState(1, "first", "", 1024, 1024, 1, 20, 7, {"sampler": "euler"})
    branch = copy.deepcopy(original)
    branch.prompt = "second"
    branch.extras["sampler"] = "ddim"
    assert original.prompt == "first"
    assert original.extras["sampler"] == "euler"
