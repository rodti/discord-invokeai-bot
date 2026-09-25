import asyncio

import discord

from invokeai_discord_bot import bot as bot_module
from invokeai_discord_bot.bot import (
    GHOST,
    GHOST_DISTANCE,
    PACMAN_DOTS,
    animate_progress,
    pacman_frame,
    progress_frame,
    progress_position,
    stop_animation,
)


def bar(frame_text: str) -> str:
    return frame_text.split("`")[1]


def pacman_cell(frame_text: str) -> int:
    text = bar(frame_text)
    for face in ("😐", "😮"):
        if face in text:
            return text.index(face) - 1  # skip the opening bracket
    raise AssertionError(f"no Pac-Man in {text!r}")


# --- the restart bug -------------------------------------------------------

def test_position_never_moves_backwards_over_a_very_long_generation():
    positions = [progress_position(frame) for frame in range(5000)]  # ~2.8 hours
    assert positions[0] == 0
    assert all(later >= earlier for earlier, later in zip(positions, positions[1:]))


def test_position_never_reaches_the_end_of_the_bar():
    assert max(progress_position(frame) for frame in range(5000)) == PACMAN_DOTS - 1


def test_rendered_bar_never_jumps_back_to_the_start():
    cells = [pacman_cell(progress_frame(frame)) for frame in range(500)]
    assert all(later >= earlier for earlier, later in zip(cells, cells[1:]))
    assert cells[-1] == PACMAN_DOTS - 1


def test_pacman_moves_on_the_first_frames_so_the_bar_feels_alive():
    assert [progress_position(frame) for frame in range(4)] == [0, 1, 2, 3]


def test_pacman_slows_down_towards_the_end():
    def frames_to_reach(cell):
        return next(frame for frame in range(10_000) if progress_position(frame) >= cell)

    gaps = [frames_to_reach(cell + 1) - frames_to_reach(cell) for cell in range(1, PACMAN_DOTS - 1)]
    # Whole-cell rounding makes early gaps wobble (1, 1, 2, 1...), so compare
    # the start of the bar with the end rather than demanding strict order.
    assert min(gaps[-3:]) > max(gaps[:3])
    assert gaps[-1] == max(gaps)


def test_mouth_keeps_chomping_while_pacman_waits_near_the_end():
    late = [progress_frame(frame) for frame in range(200, 206)]
    assert len({pacman_cell(text) for text in late}) == 1
    faces = ["😮" in bar(text) for text in late]
    assert faces == [False, True, False, True, False, True]


def test_every_frame_has_the_same_width():
    widths = {len(bar(progress_frame(frame, ghost))) for frame in range(300) for ghost in (False, True)}
    assert len(widths) == 1


def test_negative_and_oversized_positions_are_clamped():
    assert pacman_cell(pacman_frame(-5)) == 0
    assert pacman_cell(pacman_frame(PACMAN_DOTS + 20)) == PACMAN_DOTS


# --- the ghost -------------------------------------------------------------

def test_ghost_stays_the_same_distance_behind_whenever_visible():
    for frame in range(300):
        text = bar(progress_frame(frame, ghost=True))
        cell = pacman_cell(progress_frame(frame, ghost=True))
        if cell < GHOST_DISTANCE:
            assert GHOST not in text
        else:
            assert text.index(GHOST) - 1 == cell - GHOST_DISTANCE


def test_ghost_never_appears_when_not_chasing():
    assert all(GHOST not in progress_frame(frame) for frame in range(300))


def test_ghost_chance_is_occasional(monkeypatch):
    rolls = iter([0.05, 0.5, 0.95, 0.099, 0.1])
    monkeypatch.setattr(bot_module.random, "random", lambda: next(rolls))
    assert [bot_module.ghost_chase() for _ in range(5)] == [True, False, False, True, False]


# --- the animation loop ----------------------------------------------------

class FakeMessage:
    def __init__(self, fail_after=None):
        self.edits = []
        self.fail_after = fail_after

    async def edit(self, content):
        if self.fail_after is not None and len(self.edits) >= self.fail_after:
            raise discord.NotFound(FakeResponse(404), "Unknown Message")
        self.edits.append(content)


class FakeResponse:
    def __init__(self, status):
        self.status = status
        self.reason = "error"


async def run_animation(monkeypatch, message, frames, ghost=False):
    monkeypatch.setattr(bot_module, "ghost_chase", lambda: ghost)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(bot_module.asyncio, "sleep", lambda _: real_sleep(0))
    task = asyncio.create_task(animate_progress(message))
    while len(message.edits) < frames and not task.done():
        await real_sleep(0)
    await stop_animation(task)


async def test_animation_runs_for_hundreds_of_frames_without_restarting(monkeypatch):
    message = FakeMessage()
    await run_animation(monkeypatch, message, 400)
    cells = [pacman_cell(text) for text in message.edits]
    assert len(cells) >= 400
    assert cells[0] == 1  # frame 0 is shown before the animation starts
    assert all(later >= earlier for earlier, later in zip(cells, cells[1:]))


async def test_ghost_decision_holds_for_the_whole_generation(monkeypatch):
    message = FakeMessage()
    await run_animation(monkeypatch, message, 100, ghost=True)
    visible = [GHOST in text for text in message.edits]
    first = visible.index(True)
    assert not any(visible[:first])
    assert all(visible[first:])


async def test_animation_stops_quietly_when_the_message_is_deleted(monkeypatch):
    message = FakeMessage(fail_after=3)
    monkeypatch.setattr(bot_module, "ghost_chase", lambda: False)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(bot_module.asyncio, "sleep", lambda _: real_sleep(0))
    await asyncio.wait_for(animate_progress(message), timeout=1)
    assert len(message.edits) == 3


async def test_stop_animation_cancels_cleanly(monkeypatch):
    message = FakeMessage()
    task = asyncio.create_task(animate_progress(message))
    await asyncio.sleep(0)
    await stop_animation(task)
    assert task.cancelled()
