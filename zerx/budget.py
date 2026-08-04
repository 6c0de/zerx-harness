"""Observable action-efficiency signal. RHAE's human-median denominator is
hidden evaluation data the agent cannot see — this module only ever looks
at the agent's own observable action count against a configurable soft cap,
and produces a *strategy signal*, never a forced or invented action.
Legality is always enforced by policy.py's validation, not here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSignal:
    actions_taken: int
    soft_cap: int
    should_favor_execution: bool


def evaluate_budget(
    actions_taken: int, soft_cap: int, favor_threshold: float = 0.8
) -> BudgetSignal:
    """`should_favor_execution` flips once `actions_taken` crosses
    `favor_threshold` of `soft_cap` — a hint to prefer a more confident
    candidate over an exploratory one. It never selects or invents an
    action itself.
    """
    if soft_cap <= 0:
        raise ValueError("soft_cap must be positive")
    ratio = actions_taken / soft_cap
    return BudgetSignal(
        actions_taken=actions_taken,
        soft_cap=soft_cap,
        should_favor_execution=ratio >= favor_threshold,
    )
