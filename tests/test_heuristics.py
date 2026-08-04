from zerx.heuristics import DeadSignatureTracker, rank_click_candidates
from zerx.perception import LabeledObject, PerceptionResult


def _obj(label, color, cells):
    return LabeledObject(label=label, color=color, cells=tuple(cells))


def test_new_signature_has_zero_penalty():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    assert tracker.penalty(obj) == 0.0


def test_ineffective_outcome_increases_penalty():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    tracker.record_outcome(obj, effective=False)
    assert tracker.penalty(obj) > 0.0


def test_effective_outcome_recovers_penalty():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    tracker.record_outcome(obj, effective=False)
    penalty_after_fail = tracker.penalty(obj)
    tracker.record_outcome(obj, effective=True)
    assert tracker.penalty(obj) < penalty_after_fail


def test_penalty_stays_within_zero_one_range():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    for _ in range(10):
        tracker.record_outcome(obj, effective=False)
    assert tracker.penalty(obj) == 1.0
    for _ in range(10):
        tracker.record_outcome(obj, effective=True)
    assert tracker.penalty(obj) == 0.0


def test_reset_clears_penalties():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    tracker.record_outcome(obj, effective=False)
    tracker.reset()
    assert tracker.penalty(obj) == 0.0


def test_rank_click_candidates_empty_perception_returns_empty():
    result = PerceptionResult(ascii_grid="0", objects=())
    assert rank_click_candidates(result, DeadSignatureTracker()) == []


def test_rank_click_candidates_down_ranks_but_keeps_fully_penalized_object():
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    tracker = DeadSignatureTracker()
    for _ in range(10):
        tracker.record_outcome(obj, effective=False)
    result = PerceptionResult(ascii_grid="", objects=(obj,))
    candidates = rank_click_candidates(result, tracker)
    assert len(candidates) == 1  # still present, never hard-excluded
    assert candidates[0].score == 0.0


def test_rank_click_candidates_prefers_smaller_object():
    small = _obj("small", color=1, cells=[(0, 0)])
    big = _obj("big", color=2, cells=[(2, 0), (2, 1), (2, 2), (2, 3)])
    result = PerceptionResult(ascii_grid="", objects=(big, small))
    candidates = rank_click_candidates(result, DeadSignatureTracker())
    assert candidates[0].object_label == "small"


def test_click_candidate_coordinates_within_bounds():
    obj = _obj("obj0", color=1, cells=[(0, 0)])
    result = PerceptionResult(ascii_grid="", objects=(obj,))
    candidates = rank_click_candidates(result, DeadSignatureTracker(), grid_size=64)
    assert 0 <= candidates[0].x <= 63
    assert 0 <= candidates[0].y <= 63
