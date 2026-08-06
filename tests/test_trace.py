from __future__ import annotations

import json

from zerx.policy import Decision
from zerx.trace import (
    CompositeTraceRecorder,
    JsonlTraceWriter,
    TraceMeta,
    TraceStep,
    build_trace_step,
    describe_reasoning,
)
from zerx.types import Action, ActionName, GameFrame


def _frame():
    return GameFrame(
        grid=((0, 0), (0, 5)),
        legal_actions=frozenset({ActionName.RESET, ActionName.ACTION6}),
        is_game_over=False,
    )


def test_describe_reasoning_returns_raw_response_when_present():
    decision = Decision(
        action=Action(name=ActionName.ACTION1),
        source="model",
        raw_response='{"action": "ACTION1"}',
    )
    assert describe_reasoning(decision) == '{"action": "ACTION1"}'


def test_describe_reasoning_synthesizes_text_for_known_fallback_sources():
    decision = Decision(action=Action(name=ActionName.ACTION1), source="fallback_deterministic")
    assert describe_reasoning(decision) == (
        "no model or heuristic action available; used the static fallback preference order"
    )


def test_describe_reasoning_falls_back_to_source_name_for_unknown_source():
    decision = Decision(action=Action(name=ActionName.ACTION1), source="some_new_source")
    assert describe_reasoning(decision) == "some_new_source"


def test_build_trace_step_captures_frame_and_decision():
    decision = Decision(
        action=Action(name=ActionName.ACTION6, x=3, y=4),
        source="heuristic",
        target_object_label="obj-0",
    )
    step = build_trace_step(
        step_index=2,
        game_id="ls20",
        frame=_frame(),
        decision=decision,
        levels_completed=1,
        game_state="NOT_FINISHED",
    )
    assert step.step_index == 2
    assert step.game_id == "ls20"
    assert step.grid == ((0, 0), (0, 5))
    assert step.action_name == "ACTION6"
    assert step.action_x == 3
    assert step.action_y == 4
    assert step.source == "heuristic"
    assert step.target_object_label == "obj-0"
    assert step.levels_completed == 1
    assert step.game_state == "NOT_FINISHED"


def test_jsonl_trace_writer_appends_meta_then_steps(tmp_path):
    path = tmp_path / "trace.jsonl"
    writer = JsonlTraceWriter(str(path))
    writer.write_meta(TraceMeta(game_id="ls20", seed=0, backend="fake", config_hash="abc123", started_at="2026-08-06T00:00:00"))
    decision = Decision(action=Action(name=ActionName.ACTION1), source="fallback_deterministic")
    step = build_trace_step(
        step_index=0, game_id="ls20", frame=_frame(), decision=decision,
        levels_completed=0, game_state="NOT_FINISHED",
    )
    writer.record(step)

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    meta_line = json.loads(lines[0])
    assert meta_line["type"] == "meta"
    assert meta_line["game_id"] == "ls20"
    step_line = json.loads(lines[1])
    assert step_line["type"] == "step"
    assert step_line["action_name"] == "ACTION1"


def test_composite_trace_recorder_fans_out_to_every_child():
    class _Spy:
        def __init__(self):
            self.steps = []

        def record(self, step):
            self.steps.append(step)

    spy_a, spy_b = _Spy(), _Spy()
    composite = CompositeTraceRecorder([spy_a, spy_b])
    decision = Decision(action=Action(name=ActionName.ACTION1), source="fallback_deterministic")
    step = build_trace_step(
        step_index=0, game_id="ls20", frame=_frame(), decision=decision,
        levels_completed=0, game_state="NOT_FINISHED",
    )
    composite.record(step)
    assert len(spy_a.steps) == 1
    assert len(spy_b.steps) == 1
    assert spy_a.steps[0] is step
    assert spy_b.steps[0] is step
