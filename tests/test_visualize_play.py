from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import visualize_play  # noqa: E402
from visualize_play import _clamp_index, _color_for, _load_trace  # noqa: E402
from zerx.trace import TraceMeta, TraceStep


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


def test_load_trace_parses_meta_and_steps(tmp_path):
    path = tmp_path / "trace.jsonl"
    meta = TraceMeta(game_id="ls20", seed=0, backend="fake", config_hash="abc", started_at="2026-08-06T00:00:00")
    step = TraceStep(
        step_index=0, game_id="ls20", grid=((0, 0), (0, 1)), action_name="ACTION1",
        action_x=None, action_y=None, source="fallback_deterministic", repaired=False,
        target_object_label=None, reasoning="no model call needed",
        levels_completed=0, game_state="NOT_FINISHED",
    )
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "meta", **meta.__dict__}) + "\n")
        fh.write(json.dumps({"type": "step", **step.__dict__}) + "\n")

    loaded_meta, steps = _load_trace(str(path))
    assert loaded_meta.game_id == "ls20"
    assert len(steps) == 1
    assert steps[0].action_name == "ACTION1"
    assert steps[0].grid == ((0, 0), (0, 1))
    assert isinstance(steps[0].grid, tuple) and isinstance(steps[0].grid[0], tuple)
