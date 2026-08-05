"""Exact (state, action) ineffective-action suppression (STRATEGY.md §3.1,
`baseline-115-exact-state-memory`). Narrower and more confident than
`zerx.heuristics.DeadSignatureTracker`'s structural down-ranking: when the
literal same grid state has already produced zero visible change and zero
level delta for the literal same action, suppress proposing that exact
pair again -- but only until contradicting evidence (`later_disconfirmed`)
arrives for that same pair, since suppression must stay graded, never a
permanent ban (STRATEGY.md §2.5).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from zerx.types import Action, ActionName


def action_signature(action: Action) -> str:
    """A stable string identity for an Action, distinguishing ACTION6
    clicks by exact coordinate (two different click targets are two
    different actions for suppression purposes).
    """
    if action.name == ActionName.ACTION6:
        return f"{action.name.value}:{action.x},{action.y}"
    return action.name.value


@dataclass(frozen=True)
class ExactStateRecord:
    state_signature: str
    action_signature: str
    attempt_count: int
    visible_change: bool
    level_delta: int
    later_disconfirmed: bool


class ExactStateMemory:
    """Per-(state, action) outcome memory. `record_outcome` is called for
    every action's transition result (not just heuristic-sourced ones);
    `is_suppressed` is consulted before accepting a proposed action against
    the current state.
    """

    def __init__(self) -> None:
        self._records: Dict[Tuple[str, str], ExactStateRecord] = {}

    def record_outcome(
        self,
        state_signature: str,
        action_signature: str,
        visible_change: bool,
        level_delta: int,
    ) -> None:
        key = (state_signature, action_signature)
        existing = self._records.get(key)
        if existing is None:
            self._records[key] = ExactStateRecord(
                state_signature=state_signature,
                action_signature=action_signature,
                attempt_count=1,
                visible_change=visible_change,
                level_delta=level_delta,
                later_disconfirmed=False,
            )
            return

        was_ineffective = not existing.visible_change and existing.level_delta == 0
        now_effective = visible_change or level_delta != 0
        later_disconfirmed = existing.later_disconfirmed or (was_ineffective and now_effective)

        self._records[key] = replace(
            existing,
            attempt_count=existing.attempt_count + 1,
            visible_change=visible_change,
            level_delta=level_delta,
            later_disconfirmed=later_disconfirmed,
        )

    def is_suppressed(self, state_signature: str, action_signature: str) -> bool:
        record = self._records.get((state_signature, action_signature))
        if record is None:
            return False
        return (
            not record.visible_change
            and record.level_delta == 0
            and not record.later_disconfirmed
        )

    def record_for(self, state_signature: str, action_signature: str) -> Optional[ExactStateRecord]:
        return self._records.get((state_signature, action_signature))

    def reset(self) -> None:
        """Clear between games -- exact-state evidence from one game must
        never leak into the next, same discipline as MemoryState.reset(),
        DeadSignatureTracker.reset(), and TransitionLedger.reset().
        """
        self._records.clear()
