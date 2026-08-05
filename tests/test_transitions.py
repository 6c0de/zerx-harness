from zerx.transitions import TransitionLedger, grid_hash
from zerx.types import Action, ActionName, GameFrame

DEFAULT_LEGAL = frozenset({ActionName.ACTION1, ActionName.ACTION2, ActionName.ACTION5})


def _frame(grid, legal=None, score=0, is_game_over=False):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=legal if legal is not None else DEFAULT_LEGAL,
        is_game_over=is_game_over,
        score=score,
    )


def test_finalize_without_begin_returns_none():
    ledger = TransitionLedger()
    assert ledger.finalize(_frame([[0]])) is None


def test_records_basic_transition_with_diff():
    ledger = TransitionLedger()
    before = _frame([[0, 0], [0, 0]])
    after = _frame([[0, 0], [0, 5]])
    action = Action(name=ActionName.ACTION1)
    ledger.begin(before, action)
    record = ledger.finalize(after)
    assert record.action == action
    assert record.changed_pixels == 1
    assert record.change_bbox == (1, 1, 1, 1)
    assert record.terminal is False


def test_no_change_is_flagged_repeated_and_not_effective():
    ledger = TransitionLedger()
    frame = _frame([[0, 0], [0, 5]])
    ledger.begin(frame, Action(name=ActionName.ACTION1))
    record = ledger.finalize(frame)
    assert record.changed_pixels == 0
    assert record.change_bbox is None
    assert record.repeated_state is True
    assert record.effective is False


def test_score_delta_and_terminal_make_a_transition_effective_without_pixel_change():
    before = _frame([[0]], score=1)
    after = _frame([[0]], score=3, is_game_over=True)
    ledger = TransitionLedger()
    ledger.begin(before, Action(name=ActionName.ACTION5))
    record = ledger.finalize(after)
    assert record.score_delta == 2
    assert record.terminal is True
    assert record.effective is True


def test_step_increments_across_begin_finalize_pairs():
    ledger = TransitionLedger()
    frame = _frame([[0]])
    ledger.begin(frame, Action(name=ActionName.ACTION1))
    first = ledger.finalize(frame)
    ledger.begin(frame, Action(name=ActionName.ACTION1))
    second = ledger.finalize(frame)
    assert first.step == 0
    assert second.step == 1


def test_reset_clears_pending_transition():
    ledger = TransitionLedger()
    ledger.begin(_frame([[0]]), Action(name=ActionName.ACTION1))
    ledger.reset()
    assert ledger.finalize(_frame([[0]])) is None


def test_detects_loop_beyond_the_immediate_step():
    ledger = TransitionLedger()
    frame_a = _frame([[0, 0], [0, 0]])
    frame_b = _frame([[0, 0], [0, 5]])
    ledger.begin(frame_a, Action(name=ActionName.ACTION1))
    ledger.finalize(frame_b)
    ledger.begin(frame_b, Action(name=ActionName.ACTION2))
    record = ledger.finalize(frame_a)  # back to frame_a's exact state
    assert record.repeated_state is True


def test_records_legal_actions_before_and_after():
    before = _frame([[0]], legal=frozenset({ActionName.ACTION1}))
    after = _frame([[0]], legal=frozenset({ActionName.ACTION1, ActionName.ACTION5}))
    ledger = TransitionLedger()
    ledger.begin(before, Action(name=ActionName.ACTION1))
    record = ledger.finalize(after)
    assert record.legal_before == frozenset({ActionName.ACTION1})
    assert record.legal_after == frozenset({ActionName.ACTION1, ActionName.ACTION5})


def test_grid_hash_is_public_and_deterministic():
    frame = _frame([[1, 2], [3, 4]])
    assert grid_hash(frame) == grid_hash(frame)
    assert isinstance(grid_hash(frame), str)


def test_grid_hash_differs_for_different_grids():
    a = _frame([[0, 0], [0, 0]])
    b = _frame([[0, 0], [0, 1]])
    assert grid_hash(a) != grid_hash(b)
