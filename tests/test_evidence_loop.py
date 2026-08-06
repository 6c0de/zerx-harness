"""The evidence channel: what the agent's own recent actions did, fed back
into the prompt and into reflection memory.

Before this existed, every prompt showed the board and the legal actions
but never the outcome of anything already tried, and `Config.memory_on`
ran a `lambda prev, ctx: prev` summarizer so the prompt permanently read
"What you've learned so far: (nothing yet)".
"""
from __future__ import annotations

from zerx.config import Config
from zerx.heuristics import DeadSignatureTracker
from zerx.memory import MemoryState
from zerx.model_backend import FakeModelBackend
from zerx.perception import PerceptionResult
from zerx.policy import build_prompt, decide
from zerx.transitions import (
    TransitionLedger,
    TransitionRecord,
    render_transition_history,
    summarize_transitions,
)
from zerx.types import Action, ActionName, GameFrame

LEGAL = frozenset({ActionName.ACTION1, ActionName.ACTION5, ActionName.ACTION6})


def _record(action: Action, changed: int = 0, score_delta: int = 0, **kw) -> TransitionRecord:
    return TransitionRecord(
        step=kw.get("step", 0),
        before_hash="b",
        action=action,
        after_hash="a",
        changed_pixels=changed,
        change_bbox=kw.get("bbox"),
        legal_before=LEGAL,
        legal_after=LEGAL,
        score_delta=score_delta,
        terminal=kw.get("terminal", False),
        repeated_state=kw.get("repeated_state", False),
        change_label=kw.get("change_label"),
    )


def _frame(grid, legal=LEGAL, is_game_over=False, score=0) -> GameFrame:
    return GameFrame(
        grid=tuple(tuple(r) for r in grid),
        legal_actions=legal,
        is_game_over=is_game_over,
        score=score,
    )


def test_history_says_explicitly_when_an_action_did_nothing():
    text = render_transition_history([_record(Action(name=ActionName.ACTION1))])
    assert "ACTION1" in text
    assert "NOTHING" in text


def test_history_reports_a_level_completion_above_pixel_detail():
    text = render_transition_history(
        [_record(Action(name=ActionName.ACTION5), changed=3, score_delta=1)]
    )
    assert "COMPLETED A LEVEL (+1)" in text


def test_history_includes_click_coordinates_and_changed_region():
    text = render_transition_history(
        [_record(Action(name=ActionName.ACTION6, x=7, y=9), changed=12, bbox=(1, 2, 3, 4))]
    )
    assert "ACTION6(x=7, y=9)" in text
    assert "12 cells" in text
    assert "x 1-3" in text and "y 2-4" in text


def test_history_surfaces_an_optional_scene_label():
    text = render_transition_history(
        [_record(Action(name=ActionName.ACTION1), changed=2, change_label="HUD_ONLY")]
    )
    assert "[HUD_ONLY]" in text


def test_history_is_bounded_so_it_cannot_swamp_the_prompt():
    records = [_record(Action(name=ActionName.ACTION1), step=i) for i in range(50)]
    assert len(render_transition_history(records, limit=8).splitlines()) == 8


def test_history_with_nothing_recorded_says_so():
    assert render_transition_history([]) == "(no actions taken yet)"


def test_build_prompt_carries_the_history_and_the_do_not_repeat_instruction():
    prompt = build_prompt(
        PerceptionResult(ascii_grid="0", objects=()),
        MemoryState(),
        recent_transitions=[_record(Action(name=ActionName.ACTION1))],
    )
    assert "What your recent actions actually did" in prompt
    assert "ACTION1 -> changed NOTHING" in prompt
    assert "Do not repeat an action" in prompt


def test_summarizer_separates_effective_actions_from_proven_dead_ends():
    records = [
        _record(Action(name=ActionName.ACTION1)),
        _record(Action(name=ActionName.ACTION1)),
        _record(Action(name=ActionName.ACTION5), changed=20),
    ]
    summary = summarize_transitions(records)
    assert "ACTION5" in summary
    assert "ACTION1 (2x)" in summary
    assert "Did nothing every time tried" in summary


def test_summarizer_never_writes_off_an_action_that_worked_at_least_once():
    """A single no-op observation is weak evidence; permanently marking an
    action dead that the agent later needs is worse than saying nothing.
    """
    records = [
        _record(Action(name=ActionName.ACTION1)),
        _record(Action(name=ActionName.ACTION1), changed=5),
    ]
    summary = summarize_transitions(records)
    assert "Did nothing every time tried" not in summary


def test_summarizer_is_empty_with_no_evidence():
    assert summarize_transitions([]) == ""


def test_memory_on_actually_reaches_the_prompt_now():
    """Regression for ARC-HANDOFF-003's `memory_on`: it defaulted to True
    while controlling nothing, so the summary could never become non-empty.
    """
    records = [_record(Action(name=ActionName.ACTION1)) for _ in range(3)]
    memory = MemoryState()
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'] * 3)
    config = Config(memory_refresh_interval=1)

    for step in range(3):
        _, memory = decide(
            frame=_frame([[0, 0], [0, 0]]),
            history=(),
            memory=memory,
            dead_signatures=DeadSignatureTracker(),
            config=config,
            backend=backend,
            actions_taken=step,
            recent_transitions=records,
        )

    assert memory.summary != ""
    assert "ACTION1" in memory.summary
    assert "(nothing yet)" not in backend.last_prompt
    assert memory.summary in backend.last_prompt


def test_memory_off_leaves_the_summary_empty():
    _, memory = decide(
        frame=_frame([[0, 0], [0, 0]]),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(memory_on=False, memory_refresh_interval=1),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
        recent_transitions=[_record(Action(name=ActionName.ACTION1))],
    )
    assert memory.summary == ""


def test_ledger_attaches_a_classifier_label_when_one_is_supplied():
    ledger = TransitionLedger()
    before = _frame([[0, 0], [0, 0]])
    after = _frame([[0, 0], [0, 5]])
    ledger.begin(before, Action(name=ActionName.ACTION1))
    record = ledger.finalize(after, classifier=lambda b, a: "OBJECT_APPEAR_DISAPPEAR")
    assert record is not None and record.change_label == "OBJECT_APPEAR_DISAPPEAR"


def test_a_failing_classifier_never_breaks_the_evidence_loop():
    ledger = TransitionLedger()
    before = _frame([[0, 0], [0, 0]])
    after = _frame([[0, 0], [0, 5]])
    ledger.begin(before, Action(name=ActionName.ACTION1))

    def boom(b, a):
        raise RuntimeError("scene analysis blew up")

    record = ledger.finalize(after, classifier=boom)
    assert record is not None
    assert record.change_label is None
    assert record.changed_pixels == 1  # the real evidence still landed
