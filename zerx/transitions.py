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
from typing import Callable, Deque, FrozenSet, Optional, Sequence, Tuple

from zerx.types import Action, ActionName, GameFrame

TransitionClassifier = Callable[[GameFrame, GameFrame], str]
# (before, after) -> a short semantic label for what changed. Injected, so
# this module never imports zerx.scene: the classifier is expensive (full
# scene segmentation plus boundary tracing on both frames) and is only
# supplied when Config.duck_objects_on is set.


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
    change_label: Optional[str] = None  # optional semantic label from an
    # injected TransitionClassifier (zerx/scene.py's classify_transition,
    # under Config.duck_objects_on). None means "not classified", which is
    # different from "nothing happened" -- changed_pixels carries that.

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

    def finalize(
        self, after: GameFrame, classifier: Optional[TransitionClassifier] = None
    ) -> Optional[TransitionRecord]:
        if self._pending is None:
            return None
        step, before, action = self._pending
        self._pending = None
        before_hash = _grid_hash(before)
        after_hash = _grid_hash(after)
        changed_pixels, bbox = _diff(before, after)
        repeated_state = after_hash in self._recent_hashes
        change_label: Optional[str] = None
        if classifier is not None:
            try:
                change_label = classifier(before, after)
            except Exception:  # noqa: BLE001 - an optional descriptive label
                # must never be able to break the evidence loop it annotates.
                change_label = None
        return TransitionRecord(
            change_label=change_label,
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


def _describe_action(action: Action) -> str:
    if action.name == ActionName.ACTION6:
        return f"{action.name.value}(x={action.x}, y={action.y})"
    return action.name.value


def _describe_outcome(record: TransitionRecord) -> str:
    """One clause saying what the action actually did. Ordered by how much
    the fact matters to a player: a level completion outranks pixels, and
    "nothing happened" is stated explicitly rather than omitted.
    """
    if record.score_delta > 0:
        return f"COMPLETED A LEVEL (+{record.score_delta})"
    if record.terminal:
        return "ended the game (terminal state)"
    if record.changed_pixels == 0:
        return "changed NOTHING on the board"
    where = ""
    if record.change_bbox is not None:
        min_x, min_y, max_x, max_y = record.change_bbox
        where = f" in region (x {min_x}-{max_x}, y {min_y}-{max_y})"
    label = f" [{record.change_label}]" if record.change_label else ""
    suffix = " (board returned to an earlier state)" if record.repeated_state else ""
    return f"changed {record.changed_pixels} cells{where}{label}{suffix}"


def render_transition_history(
    records: Sequence[TransitionRecord], limit: int = 8
) -> str:
    """Compact "what your recent actions actually did" block for the prompt.

    This is the evidence channel ARC-AGI-3 rewards and the agent previously
    had none of: every step the model was shown the current board and the
    legal actions, but never the outcome of anything it had already tried,
    so it had no way to notice that an action was a no-op and no reason to
    stop re-proposing it. The ledger already recorded all of this; nothing
    read it.

    Bounded to the last `limit` records: this goes into every prompt, so it
    must stay small next to the 64x64 grid already there.
    """
    if not records:
        return "(no actions taken yet)"
    shown = list(records)[-limit:]
    return "\n".join(
        f"- {_describe_action(r.action)} -> {_describe_outcome(r)}" for r in shown
    )


def summarize_transitions(records: Sequence[TransitionRecord]) -> str:
    """Deterministic reflection summary over the recorded window.

    `Config.memory_on` previously ran `maybe_refresh` with
    `summarizer=lambda prev, ctx: prev`, so the summary could never become
    non-empty and every prompt permanently read "What you've learned so
    far: (nothing yet)" -- the flag defaulted to True while controlling
    nothing (ARC-HANDOFF-003). This is a real summarizer that costs no
    model call and therefore no extra latency: it aggregates the evidence
    the ledger already has into per-action verdicts.
    """
    if not records:
        return ""

    effective: dict = {}
    ineffective: dict = {}
    levels = 0
    for record in records:
        key = _describe_action(record.action)
        if record.score_delta > 0:
            levels += record.score_delta
        if record.effective:
            effective[key] = effective.get(key, 0) + 1
        else:
            ineffective[key] = ineffective.get(key, 0) + 1

    parts = [f"Over the last {len(records)} actions:"]
    if levels:
        parts.append(f"{levels} level(s) completed.")
    if effective:
        top = sorted(effective.items(), key=lambda kv: -kv[1])[:4]
        parts.append(
            "Actions that changed the board: "
            + ", ".join(f"{name} ({count}x)" for name, count in top)
            + "."
        )
    # Only report an action as a dead end when it was never once effective —
    # a single no-op observation is weak evidence, and permanently writing
    # off an action the agent later needs is worse than saying nothing.
    dead = [name for name in ineffective if name not in effective]
    if dead:
        top_dead = sorted(dead, key=lambda name: -ineffective[name])[:4]
        parts.append(
            "Did nothing every time tried: "
            + ", ".join(f"{name} ({ineffective[name]}x)" for name in top_dead)
            + "."
        )
    if len(parts) == 1:
        return ""
    return " ".join(parts)
