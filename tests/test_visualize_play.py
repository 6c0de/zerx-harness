from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import visualize_play  # noqa: E402
from visualize_play import (  # noqa: E402
    _clamp_index,
    _clamp_scroll,
    _color_for,
    _load_trace,
    _wrap_reasoning,
)
from zerx.trace import JsonlTraceWriter, TraceMeta, build_trace_step
from zerx.policy import Decision
from zerx.types import Action, ActionName, GameFrame


def test_color_for_returns_a_distinct_rgb_tuple_per_known_color_index():
    colors = {_color_for(i) for i in range(10)}
    assert len(colors) == 10
    for color in colors:
        assert len(color) == 3
        assert all(0 <= channel <= 255 for channel in color)


def test_color_for_falls_back_to_a_default_for_out_of_range_index():
    assert _color_for(99) == _color_for(99)  # deterministic, doesn't raise


def test_clamp_index_stays_within_bounds():
    assert _clamp_index(-1, length=5) == 0
    assert _clamp_index(0, length=5) == 0
    assert _clamp_index(4, length=5) == 4
    assert _clamp_index(5, length=5) == 4
    assert _clamp_index(2, length=5) == 2


def test_clamp_index_handles_empty_buffer():
    assert _clamp_index(3, length=0) == 0


def test_wrap_reasoning_returns_text_as_is_when_it_fits_on_one_line():
    assert _wrap_reasoning("hello world", chars_per_line=20) == ["hello world"]


def test_wrap_reasoning_wraps_at_chars_per_line():
    assert _wrap_reasoning("abcdefghij", chars_per_line=4) == ["abcd", "efgh", "ij"]


def test_wrap_reasoning_never_truncates_long_text():
    # "x" * 100 wraps into 10 lines of 10 chars each -- no cap, no
    # indicator; windowing over this is the caller's (_render's) job now.
    lines = _wrap_reasoning("x" * 100, chars_per_line=10)
    assert len(lines) == 10
    assert all(line == "xxxxxxxxxx" for line in lines)


def test_wrap_reasoning_handles_empty_text():
    assert _wrap_reasoning("", chars_per_line=10) == []


def test_wrap_reasoning_handles_degenerate_chars_per_line():
    # chars_per_line <= 0 is clamped to 1 rather than raising/looping forever.
    assert _wrap_reasoning("ab", chars_per_line=0) == ["a", "b"]


def test_clamp_scroll_clamps_negative_scroll_to_zero():
    assert _clamp_scroll(-5, total_lines=10, visible_lines=3) == 0


def test_clamp_scroll_clamps_past_end_to_last_full_window():
    assert _clamp_scroll(100, total_lines=10, visible_lines=3) == 7


def test_clamp_scroll_stays_at_zero_when_everything_fits_on_screen():
    assert _clamp_scroll(0, total_lines=3, visible_lines=10) == 0
    assert _clamp_scroll(5, total_lines=3, visible_lines=10) == 0


def test_load_trace_parses_meta_and_steps(tmp_path):
    # Uses the real JsonlTraceWriter (not a hand-rolled fixture) so this
    # test actually asserts the writer's output is readable by _load_trace,
    # instead of merely re-testing _load_trace's own parsing logic against
    # a fixture that could silently drift from what the writer produces.
    path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(str(path))
    meta = TraceMeta(game_id="ls20", seed=0, backend="fake", config_hash="abc", started_at="2026-08-06T00:00:00")
    writer.write_meta(meta)
    decision = Decision(action=Action(name=ActionName.ACTION1), source="fallback_deterministic")
    frame = GameFrame(
        grid=((0, 0), (0, 1)),
        legal_actions=frozenset({ActionName.RESET, ActionName.ACTION1}),
        is_game_over=False,
    )
    step = build_trace_step(
        step_index=0, game_id="ls20", frame=frame, decision=decision,
        levels_completed=0, game_state="NOT_FINISHED",
    )
    writer.record(step)

    loaded_meta, steps = _load_trace(str(path))
    assert loaded_meta.game_id == "ls20"
    assert len(steps) == 1
    assert steps[0].action_name == "ACTION1"
    assert steps[0].grid == ((0, 0), (0, 1))
    assert isinstance(steps[0].grid, tuple) and isinstance(steps[0].grid[0], tuple)
