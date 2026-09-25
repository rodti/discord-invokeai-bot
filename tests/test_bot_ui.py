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


def test_ghost_follows_pacman_at_a_distance():
    from invokeai_discord_bot.bot import GHOST, GHOST_DISTANCE

    frame = pacman_frame(8, ghost=True)
    bar = frame.split("`")[1]
    assert bar.index("😐") - bar.index(GHOST) == GHOST_DISTANCE
    assert len(pacman_frame(8, ghost=True)) == len(pacman_frame(8))


def test_ghost_is_off_screen_until_pacman_has_a_head_start():
    from invokeai_discord_bot.bot import GHOST, GHOST_DISTANCE

    assert GHOST not in pacman_frame(GHOST_DISTANCE - 1, ghost=True)
    assert GHOST in pacman_frame(GHOST_DISTANCE, ghost=True)


def test_no_ghost_by_default():
    from invokeai_discord_bot.bot import GHOST

    assert all(GHOST not in pacman_frame(i) for i in range(13))
