"""Tests for `agent/my_agent.py`'s `MyAgent.choose_action` exception
boundary (final whole-branch review Fix 1) and, incidentally, the decision
telemetry now attached via `GameAction.reasoning` (Fix 3b).

`agents.agent.Agent.__init__` (the real upstream base class) only stores
its constructor args and does not touch `arc_env` during `__init__` or
during `choose_action` (only during `take_action`, which `choose_action`
never calls) — so a real `MyAgent` can be constructed with `arc_env=None`
for unit testing `choose_action` directly, without a live game
environment. This module therefore depends on the vendored
`ARC-AGI-3-Agents` framework and the pip-installed `arcengine`, exactly
like `agent/my_agent.py` itself already does.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if str(VENDOR) not in sys.path:
    # `agent/my_agent.py` imports `from agents.agent import Agent`; the
    # vendored framework package only resolves once this is on sys.path
    # (see scripts/play_local.py, which does the same thing).
    sys.path.insert(0, str(VENDOR))

from arcengine import FrameData, GameAction, GameState  # noqa: E402

from agent.my_agent import MyAgent  # noqa: E402


def _make_agent() -> MyAgent:
    return MyAgent(
        card_id="test-card",
        game_id="test-game",
        agent_name="test-agent",
        ROOT_URL="http://example.invalid",
        record=False,
        arc_env=None,
    )


def test_choose_action_survives_shape_mismatch_between_successive_frames():
    """TransitionLedger.finalize diffs the new frame against the previous
    one recorded by begin(); zerx/transitions.py's `_diff` indexes
    `after.grid[y][x]` using `before`'s dimensions, which raises
    `IndexError` if the new frame is a different (narrower) shape. This
    must not escape `choose_action`.
    """
    agent = _make_agent()

    frame1 = FrameData(
        frame=[[[0, 0], [0, 0]]],  # 2x2
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )
    first = agent.choose_action([frame1], frame1)
    assert isinstance(first, GameAction)

    # Narrower second frame -> before's width (2) indexes out of range
    # against after's single-column rows -> IndexError inside finalize().
    frame2 = FrameData(
        frame=[[[9]]],  # 1x1
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )
    second = agent.choose_action([frame1, frame2], frame2)
    assert isinstance(second, GameAction)


def test_choose_action_survives_out_of_range_available_action_id():
    """`available_actions` containing an id with no corresponding
    `GameAction` (e.g. a hypothetical future game exposing id 99) makes
    `GameAction.from_id` raise inside `_to_game_frame`. `choose_action`
    must still return a real, legal `GameAction` (via the outer fallback,
    since the bad id breaks translation before decide() ever runs).
    """
    agent = _make_agent()
    frame = FrameData(
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        available_actions=[99, 1, 2],
    )
    action = agent.choose_action([frame], frame)
    assert isinstance(action, GameAction)
    # The fallback linear-scans available_actions in order and returns the
    # first id that resolves via GameAction.from_id; 99 is skipped, 1 (the
    # next entry) succeeds.
    assert action is GameAction.ACTION1
    # This IS the exception-fallback path (translation broke before decide()
    # ever ran) — its reasoning must say so, not silently inherit whatever a
    # prior normal decision last stamped onto this shared enum singleton.
    assert action.reasoning == {"source": "exception_fallback"}


def test_safe_fallback_overwrites_stale_reasoning_from_a_prior_normal_decision():
    """`GameAction` members are process-wide singletons: `_choose_action_inner`
    sets `.reasoning` on whichever member a normal decision returns, and
    that attribute persists on the singleton afterward. A later
    exception-triggered fallback that happens to return the SAME member
    must overwrite it — otherwise a crash-recovery action gets recorded as
    if it came from the normal decide() pipeline.
    """
    agent = _make_agent()

    # A normal decision that stamps GameAction.ACTION1 with real telemetry.
    GameAction.ACTION1.reasoning = {"source": "heuristic", "repaired": False}

    # Now force the exception-fallback path, landing on that same member.
    frame = FrameData(
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        available_actions=[99, 1],
    )
    action = agent.choose_action([frame], frame)

    assert action is GameAction.ACTION1
    assert action.reasoning == {"source": "exception_fallback"}


def test_choose_action_on_bare_minimal_frame_takes_normal_reset_path():
    """A default-constructed FrameData (state=NOT_PLAYED, no frame data, no
    available_actions) does not need the exception-boundary fallback at
    all: `_to_game_frame` maps NOT_PLAYED to `is_game_over=True`, and
    `zerx.policy.decide()`'s terminal short-circuit returns RESET directly,
    before finalize()'s diff or any model call. Confirm that is really the
    path taken (not a raise recovered by the outer fallback).
    """
    agent = _make_agent()
    frame = FrameData()  # state=NOT_PLAYED, frame=[], available_actions=[]

    action = agent.choose_action([frame], frame)

    assert action is GameAction.RESET
    # Fix 3b: decision telemetry reaches the framework via `.reasoning`.
    # The normal (non-exceptional) path sets it; a fallback-path action
    # would leave it unset, so this also confirms which path was taken.
    assert action.reasoning is not None
    assert action.reasoning["source"] == "reset"


def test_structured_memory_off_by_default_is_a_true_no_op():
    """With no env vars set (structured_memory_on defaults False), the
    structured memory attribute must never advance across several
    choose_action calls -- confirms the flag is a real no-op, not just an
    unused field.
    """
    agent = _make_agent()
    assert agent._structured_memory.step_count == 0

    frame = FrameData(
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )
    agent.choose_action([frame], frame)
    agent.choose_action([frame], frame)

    assert agent._structured_memory.step_count == 0


def test_structured_memory_on_advances_step_count(monkeypatch):
    monkeypatch.setenv("ZERX_STRUCTURED_MEMORY_ON", "true")
    agent = _make_agent()
    assert agent._structured_memory.step_count == 0

    frame = FrameData(
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )
    agent.choose_action([frame], frame)

    assert agent._structured_memory.step_count == 1


def test_trace_recorder_is_none_by_default():
    agent = _make_agent()
    assert agent.trace_recorder is None


def test_trace_recorder_records_once_per_choose_action_call_when_attached():
    agent = _make_agent()

    class _Spy:
        def __init__(self):
            self.steps = []

        def record(self, step):
            self.steps.append(step)

    spy = _Spy()
    agent.trace_recorder = spy

    frame = FrameData(
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )
    agent.choose_action([frame], frame)
    assert len(spy.steps) == 1
    assert spy.steps[0].step_index == 0
    assert spy.steps[0].game_id == agent.game_id

    agent.choose_action([frame, frame], frame)
    assert len(spy.steps) == 2
    assert spy.steps[1].step_index == 1


def test_config_driven_trace_recorder_writes_a_replayable_meta_line(tmp_path, monkeypatch):
    """Finding 1(b): a config-driven trace_recorder (built from
    Config.trace_export_path, e.g. a headless Colab run via
    ZERX_TRACE_EXPORT_PATH) must call write_meta() so the resulting file
    has a meta line -- scripts/visualize_play.py's _load_trace requires
    one to replay a trace at all. Uses a real directory-mode
    JsonlTraceWriter (the actual code under test), not a hand-rolled file.
    """
    monkeypatch.setenv("ZERX_TRACE_EXPORT_PATH", str(tmp_path))
    agent = _make_agent()

    frame = FrameData(
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )
    agent.choose_action([frame], frame)

    matches = list(tmp_path.glob(f"{agent.game_id}-*.jsonl"))
    assert len(matches) == 1
    lines = matches[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    meta_line = json.loads(lines[0])
    assert meta_line["type"] == "meta"
    assert meta_line["game_id"] == agent.game_id


def test_trace_recorder_exception_does_not_desync_agent_state():
    """Finding 2: a dev-only observability sink (trace_recorder) must
    never be able to alter real agent behavior. A raising `.record()`
    must not trip the OUTER exception boundary (choose_action must still
    return a normal decide()-sourced action, not _safe_fallback_action's
    `{"source": "exception_fallback"}`), and `_actions_taken` must still
    increment normally across repeated calls.
    """
    agent = _make_agent()

    class _RaisingRecorder:
        def record(self, step):
            raise RuntimeError("simulated trace sink failure (e.g. disk full)")

    agent.trace_recorder = _RaisingRecorder()

    frame = FrameData(
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )

    first = agent.choose_action([frame], frame)
    assert isinstance(first, GameAction)
    assert first.reasoning != {"source": "exception_fallback"}
    assert "source" in first.reasoning and first.reasoning["source"] != "exception_fallback"
    assert agent._actions_taken == 1

    second = agent.choose_action([frame, frame], frame)
    assert isinstance(second, GameAction)
    assert second.reasoning != {"source": "exception_fallback"}
    assert agent._actions_taken == 2
