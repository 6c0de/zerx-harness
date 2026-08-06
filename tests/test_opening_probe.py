"""The opening probe: spend the first few actions establishing what each
action does, model-free, so the model's first real decision is made with a
filled-in evidence table instead of a guess at the control scheme.
"""
from __future__ import annotations

from zerx.config import Config
from zerx.heuristics import DeadSignatureTracker
from zerx.memory import MemoryState
from zerx.model_backend import FakeModelBackend
from zerx.policy import _opening_probe, _unprobed_actions, decide
from zerx.transitions import TransitionRecord
from zerx.types import Action, ActionName, GameFrame

LEGAL = frozenset({ActionName.ACTION1, ActionName.ACTION2, ActionName.ACTION5, ActionName.RESET})


def _frame(grid=((0, 0), (0, 0)), legal=LEGAL, is_game_over=False):
    return GameFrame(
        grid=tuple(tuple(r) for r in grid),
        legal_actions=legal,
        is_game_over=is_game_over,
        score=0,
    )


def _record(name: ActionName, x=None, y=None) -> TransitionRecord:
    return TransitionRecord(
        step=0,
        before_hash="b",
        action=Action(name=name, x=x, y=y) if x is not None else Action(name=name),
        after_hash="a",
        changed_pixels=0,
        change_bbox=None,
        legal_before=LEGAL,
        legal_after=LEGAL,
        score_delta=0,
        terminal=False,
        repeated_state=False,
    )


def test_unprobed_lists_only_legal_actions_never_observed():
    assert _unprobed_actions(LEGAL, []) == (
        ActionName.ACTION5,
        ActionName.ACTION1,
        ActionName.ACTION2,
    )
    assert _unprobed_actions(LEGAL, [_record(ActionName.ACTION5)]) == (
        ActionName.ACTION1,
        ActionName.ACTION2,
    )


def test_probe_walks_every_legal_action_exactly_once_then_stops():
    """This is the whole point: after N actions the evidence table has one
    observation per action name, and the probe gets out of the way.
    """
    seen = []
    history = []
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'] * 10)

    for step in range(3):  # exactly one per legal non-RESET action
        decision, _ = decide(
            frame=_frame(),
            history=(),
            memory=MemoryState(),
            dead_signatures=DeadSignatureTracker(),
            config=Config(),
            backend=backend,
            actions_taken=step,
            recent_transitions=tuple(history),
        )
        assert decision.source == "probe"
        seen.append(decision.action.name)
        history.append(_record(decision.action.name))

    assert seen == [ActionName.ACTION5, ActionName.ACTION1, ActionName.ACTION2]
    assert backend.call_count == 0, "the probe must cost no model calls"
    # Next decision hands over to the model.
    decision, _ = decide(
        frame=_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=backend,
        actions_taken=len(seen),
        recent_transitions=tuple(history),
    )
    assert decision.source == "model"


def test_probe_never_overrides_the_terminal_reset_shortcut():
    decision, _ = decide(
        frame=_frame(is_game_over=True),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[]),
        actions_taken=0,
        recent_transitions=(),
    )
    assert decision.source == "reset"
    assert decision.action.name == ActionName.RESET


def test_probe_stops_at_its_action_budget_even_with_actions_left_unprobed():
    """A mid-game change in the legal set must not restart probing late in a
    run -- the budget is what bounds that.
    """
    assert (
        _opening_probe(LEGAL, [], candidates=(), actions_taken=12, probe_actions=12)
        is None
    )
    assert (
        _opening_probe(LEGAL, [], candidates=(), actions_taken=11, probe_actions=12)
        is not None
    )


def test_probe_can_be_turned_off_entirely():
    decision, _ = decide(
        frame=_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(opening_probe_on=False),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
        recent_transitions=(),
    )
    assert decision.source == "model"


def test_action6_is_probed_at_a_real_candidate_not_a_blind_coordinate():
    from zerx.heuristics import ClickCandidate

    legal = frozenset({ActionName.ACTION6, ActionName.RESET})
    candidate = ClickCandidate(x=13, y=21, object_label="obj0", score=0.5)

    probe = _opening_probe(legal, [], (candidate,), actions_taken=0, probe_actions=12)
    assert probe is not None
    action, label = probe
    assert (action.name, action.x, action.y) == (ActionName.ACTION6, 13, 21)
    assert label == "obj0"


def test_action6_probe_is_skipped_when_there_is_nothing_worth_clicking():
    """One arbitrary click out of 4096 cells teaches nothing, so it is not
    worth an action.
    """
    legal = frozenset({ActionName.ACTION6, ActionName.RESET})
    assert _opening_probe(legal, [], (), actions_taken=0, probe_actions=12) is None


def test_probe_records_its_click_target_so_affordance_feedback_still_works():
    from zerx.heuristics import ClickCandidate

    legal = frozenset({ActionName.ACTION6, ActionName.RESET})
    decision, _ = decide(
        frame=_frame(grid=((0, 0), (0, 5)), legal=legal),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[]),
        actions_taken=0,
        recent_transitions=(),
    )
    assert decision.source == "probe"
    assert decision.target_object_label is not None
