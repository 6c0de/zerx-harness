import pytest

from zerx.budget import evaluate_budget


def test_evaluate_budget_below_threshold():
    signal = evaluate_budget(actions_taken=5, soft_cap=50)
    assert signal.should_favor_execution is False


def test_evaluate_budget_at_threshold_favors_execution():
    signal = evaluate_budget(actions_taken=40, soft_cap=50, favor_threshold=0.8)
    assert signal.should_favor_execution is True


def test_evaluate_budget_above_soft_cap_still_favors_execution():
    signal = evaluate_budget(actions_taken=100, soft_cap=50)
    assert signal.should_favor_execution is True
    assert signal.actions_taken == 100
    assert signal.soft_cap == 50


def test_evaluate_budget_rejects_non_positive_soft_cap():
    with pytest.raises(ValueError):
        evaluate_budget(actions_taken=1, soft_cap=0)
