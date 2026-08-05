from zerx.exact_state_memory import ExactStateMemory, action_signature
from zerx.types import Action, ActionName


def test_new_pair_has_no_suppression():
    memory = ExactStateMemory()
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_ineffective_outcome_suppresses_exact_pair():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is True


def test_effective_outcome_does_not_suppress():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=True, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_level_delta_outcome_does_not_suppress():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=1)
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_different_action_same_state_not_suppressed():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION2") is False


def test_same_action_different_state_not_suppressed():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-b", "ACTION1") is False


def test_later_disconfirmed_lifts_suppression_and_it_stays_lifted():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is True

    # Same exact (state, action) pair later produces a real change --
    # contradicts "identical state -> identical outcome", so suppression
    # must lift.
    memory.record_outcome("state-a", "ACTION1", visible_change=True, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is False

    # A subsequent no-op observation must NOT re-suppress a disconfirmed pair.
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_reset_clears_all_records():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    memory.reset()
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_attempt_count_increments_on_repeated_recording():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    record = memory.record_for("state-a", "ACTION1")
    assert record.attempt_count == 2


def test_record_for_returns_none_when_absent():
    memory = ExactStateMemory()
    assert memory.record_for("state-a", "ACTION1") is None


def test_action_signature_stable_for_non_action6():
    action = Action(name=ActionName.ACTION3)
    assert action_signature(action) == "ACTION3"


def test_action_signature_distinguishes_action6_coordinates():
    a = Action(name=ActionName.ACTION6, x=10, y=20)
    b = Action(name=ActionName.ACTION6, x=11, y=20)
    assert action_signature(a) != action_signature(b)


def test_action_signature_same_action6_coordinates_match():
    a = Action(name=ActionName.ACTION6, x=10, y=20)
    b = Action(name=ActionName.ACTION6, x=10, y=20)
    assert action_signature(a) == action_signature(b)
