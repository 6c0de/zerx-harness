"""Evidence-first transition ledger (STRATEGY.md's Tycho-informed
adoption). Pairs each action with the *next* frame into a
TransitionRecord — never inferred before that frame exists. This is
baseline infrastructure: it costs no model calls and no action budget, and
must work even when memory and heuristics are off.
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Deque, FrozenSet, Optional, Tuple

from zerx.types import Action, ActionName, GameFrame


def grid_hash(frame: GameFrame) -> str:
    flat = ",".join(str(v) for row in frame.grid for v in row)
    return hashlib.sha256(flat.encode("utf-8")).hexdigest()[:16]


_grid_hash = grid_hash


def _shapes_match(before: GameFrame, after: GameFrame) -> bool:
    return len(before.grid) == len(after.grid) and all(
        len(row_b) == len(row_a) for row_b, row_a in zip(before.grid, after.grid)
    )


def _diff(
    before: GameFrame, after: GameFrame
) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
    """A grid-shape change is itself a real transition, not a comparable
    cell-by-cell diff. Two shapes actually occur in a live run: the very
    first NOT_PLAYED frame has an empty grid (`FrameData.frame == []`), so
    empty->64x64 used to report "no change" even though the whole board
    appeared, and 64x64->empty used to raise IndexError out of finalize()
    (silently costing the whole decision step via my_agent's crash
    boundary). Report a shape change as "everything changed" with no bbox,
    which is both true and safe for `TransitionRecord.effective`.
    """
    if not _shapes_match(before, after):
        changed_cells = max(
            sum(len(row) for row in before.grid), sum(len(row) for row in after.grid)
        )
        return changed_cells, None
    height = len(before.grid)
    width = len(before.grid[0]) if height else 0
    changed = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if before.grid[y][x] != after.grid[y][x]
    ]
    if not changed:
        return 0, None
    xs = [c[0] for c in changed]
    ys = [c[1] for c in changed]
    return len(changed), (min(xs), min(ys), max(xs), max(ys))


@dataclass(frozen=True)
class TransitionRecord:
    step: int
    before_hash: str
    action: Action
    after_hash: str
    changed_pixels: int
    change_bbox: Optional[Tuple[int, int, int, int]]
    legal_before: FrozenSet[ActionName]
    legal_after: FrozenSet[ActionName]
    score_delta: int
    terminal: bool
    repeated_state: bool

    @property
    def effective(self) -> bool:
        """An action "did something" if it changed the grid or the score.
        Feeds zerx.heuristics.DeadSignatureTracker.record_outcome.
        """
        return self.changed_pixels > 0 or self.score_delta != 0


class TransitionLedger:
    """Stateful pairing of "action taken against frame X" with "frame X+1
    arrived". `begin()` records a pending action; `finalize()` — called at
    the start of the *next* choose_action, once the new frame exists —
    completes the record. `history_size` bounds a recent-hash window used
    for loop/repeated-state detection beyond the immediate before/after
    pair.
    """

    def __init__(self, history_size: int = 20) -> None:
        self._pending: Optional[Tuple[int, GameFrame, Action]] = None
        self._step = 0
        self._recent_hashes: Deque[str] = deque(maxlen=history_size)

    def begin(self, before: GameFrame, action: Action) -> None:
        self._pending = (self._step, before, action)
        self._recent_hashes.append(_grid_hash(before))
        self._step += 1

    def finalize(self, after: GameFrame) -> Optional[TransitionRecord]:
        if self._pending is None:
            return None
        step, before, action = self._pending
        self._pending = None
        before_hash = _grid_hash(before)
        after_hash = _grid_hash(after)
        changed_pixels, bbox = _diff(before, after)
        repeated_state = after_hash in self._recent_hashes
        return TransitionRecord(
            step=step,
            before_hash=before_hash,
            action=action,
            after_hash=after_hash,
            changed_pixels=changed_pixels,
            change_bbox=bbox,
            legal_before=before.legal_actions,
            legal_after=after.legal_actions,
            score_delta=after.score - before.score,
            terminal=after.is_game_over,
            repeated_state=repeated_state,
        )

    def reset(self) -> None:
        self._pending = None
        self._step = 0
        self._recent_hashes.clear()
