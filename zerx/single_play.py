"""Single-play policy, written for the rules the gateway actually enforces.

What the rules are
------------------
Measured against a real competition-mode gateway (`eval/gateway_smoke.py`),
not inferred:

* One play per game. `arc_agi/api.py`'s RESET handler refuses to execute a
  reset when `_action_count == 0` — exactly the condition that would start a
  new play — so the "explore freely, then replay a minimal solution" route is
  closed. Actions accumulate for the whole game.
* Level *k*'s action count runs from the completion of level *k-1* to the
  completion of level *k* (`arc_agi/scorecard.py::_calculate_score`).
* `level_score = min(115, (human_baseline / actions_on_that_level) ** 2 * 100)`.

The quadratic is the whole design brief. With a human baseline of 22 actions:

    22 actions -> 100%      44 actions -> 25%
    35 actions ->  39%      66 actions -> 11%       200 actions -> 1.2%

So a level is won or lost in its first ~2-3x baseline actions. Past that it is
worth roughly nothing no matter what happens.

That cuts both ways, and the second half is what most of this file is about:
**actions spent on a level that is never completed cost nothing.** They are
charged to that level, whose score is 0 either way, and they never touch
another level's count. So the correct policy is not uniformly careful — it is
careful while the current level is still worth winning, and completely
unrestrained afterwards, because by then the only thing left to buy is the
chance of stumbling into the *next* level, whose count has not started yet.

Hence two phases per level, `careful` then `reckless`, switched by how many
actions the current level has already cost.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

ActionKey = Tuple[str, int, int]  # (name, x, y); x=y=-1 unless ACTION6
RESET: ActionKey = ("RESET", -1, -1)

# ACTION7 is undo. It can only ever move the board backwards, and every probe
# of it costs a real action on the level being scored, so it is never proposed.
SIMPLE_ORDER = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5")


def as_grid(frame_layer) -> Tuple[Tuple[int, ...], ...]:
    """Normalise one rendered frame layer to nested tuples.

    The engine hands back numpy arrays for raw frames and nested lists
    otherwise; `if not array` raises on anything multi-element, so callers
    normalise here and the rest of the module uses plain equality.
    """
    if frame_layer is None:
        return ()
    if hasattr(frame_layer, "tolist"):
        frame_layer = frame_layer.tolist()
    return tuple(tuple(int(cell) for cell in row) for row in frame_layer)


def click_candidates(grid: Sequence[Sequence[int]], limit: int = 28) -> List[Tuple[int, int]]:
    """Coordinates worth clicking, rarest colour region first.

    The interactive element is essentially never the background, and the
    background is by construction the most common colour. Each rare colour
    contributes the cell of its region nearest that region's centroid, so the
    click lands inside the thing rather than on its edge.

    Deliberately short and deliberately *not* padded with a lattice sweep. In
    the free-exploration regime a lattice was worth having; here every miss is
    charged to the level being scored, so an ordered shortlist of plausible
    targets beats coverage.
    """
    if not grid or not grid[0]:
        return [(32, 32)]
    positions: Dict[int, List[Tuple[int, int]]] = {}
    for y, row in enumerate(grid):
        for x, colour in enumerate(row):
            positions.setdefault(colour, []).append((x, y))

    area = len(grid) * len(grid[0])
    out: List[Tuple[int, int]] = []
    for _colour, cells in sorted(positions.items(), key=lambda kv: (len(kv[1]), kv[0])):
        if len(cells) > area // 3:
            continue  # background-sized region, not a target
        cx = sum(c[0] for c in cells) // len(cells)
        cy = sum(c[1] for c in cells) // len(cells)
        if (cx, cy) not in cells:
            cx, cy = min(cells, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
        if (cx, cy) not in out:
            out.append((cx, cy))
        if len(out) >= limit:
            break
    return out or [(32, 32)]


@dataclass
class _Stats:
    uses: int = 0
    changes: int = 0

    @property
    def rate(self) -> float:
        return self.changes / self.uses if self.uses else 1.0


@dataclass
class SinglePlayAgent:
    """One game, one play, careful early and reckless once the level is lost."""

    max_seconds: float = 120.0
    max_actions: int = 4000
    sticky: float = 0.7
    seed: int = 0
    # How many actions a level may cost before it stops being worth protecting.
    # Human level-1 baselines on the public set run 17-78 actions; at 4x the
    # top of that range the level is already worth under 1% and the careful
    # phase is only costing us exploration.
    careful_budget: int = 220
    deadline: Optional[float] = None

    # A cell that changes on more than this fraction of steps is animation,
    # HUD or a timer — not a consequence of what we did.
    noise_fraction: float = 0.35

    _rng: random.Random = field(default_factory=random.Random)
    _churn: Dict[Tuple[int, int], int] = field(default_factory=dict)
    _steps_seen: int = 0
    _stats: Dict[ActionKey, _Stats] = field(default_factory=dict)
    _noop: Set[Tuple[int, ActionKey]] = field(default_factory=set)
    _queue: List[ActionKey] = field(default_factory=list)
    _pool: List[Tuple[int, int]] = field(default_factory=list)
    _last: Optional[ActionKey] = None
    _last_changed: bool = False
    _prev_grid: Optional[Sequence[Sequence[int]]] = None
    _prev_hash: int = 0
    _prev_levels: int = 0
    _since_level: int = 0
    actions_used: int = 0
    levels: int = 0
    _started: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ---- budget -------------------------------------------------------

    @property
    def out_of_budget(self) -> bool:
        if self.actions_used >= self.max_actions:
            return True
        if time.time() - self._started >= self.max_seconds:
            return True
        return self.deadline is not None and time.time() >= self.deadline

    @property
    def careful(self) -> bool:
        return self._since_level < self.careful_budget

    # ---- observation ---------------------------------------------------

    def _signal_change(self, grid) -> bool:
        """Did the previous action change anything *we caused*?

        Naively comparing whole frames does not answer that. These games
        animate: a timer digit, a blinking cursor or a scrolling HUD ticks on
        essentially every frame, so `grid != previous` is true no matter what
        the agent did. That made every action look effective, so the no-op
        memory never fired and the careful phase repeated one move forever —
        which is exactly what the first honest measurement showed (every level
        found came from the reckless phase, none from the careful one).

        So track how often each cell changes at all, and once there is enough
        history, ignore the cells that change regardless of us.
        """
        if self._prev_grid is None or len(grid) != len(self._prev_grid):
            return True

        diff = [
            (x, y)
            for y, (row_a, row_b) in enumerate(zip(self._prev_grid, grid))
            for x, (a, b) in enumerate(zip(row_a, row_b))
            if a != b
        ]
        self._steps_seen += 1
        for cell in diff:
            self._churn[cell] = self._churn.get(cell, 0) + 1

        if self._steps_seen < 12:
            return bool(diff)  # not enough history to call anything noise yet
        threshold = self._steps_seen * self.noise_fraction
        return any(self._churn.get(cell, 0) <= threshold for cell in diff)

    def _observe(self, grid, levels_completed: int) -> None:
        """Fold the previous action's outcome into memory."""
        if self._last is not None and self._last != RESET:
            changed = self._signal_change(grid)
            stat = self._stats.setdefault(self._last, _Stats())
            stat.uses += 1
            if changed:
                stat.changes += 1
            else:
                # Exact-state no-op: this action does nothing *from this
                # board*. Recording it per state rather than globally is what
                # lets a direction that is blocked here stay available
                # elsewhere.
                self._noop.add((self._prev_hash, self._last))
            self._last_changed = changed

        if levels_completed > self._prev_levels:
            self.levels = levels_completed
            self._since_level = 0
            # A new level is a new board with new mechanics; stale click
            # targets and stale no-ops would both mislead.
            self._pool, self._queue = [], []
            self._noop.clear()
            # Keep `_churn` across levels on purpose: which cells are HUD and
            # which are gameplay is a property of the game, not of the level,
            # and rebuilding that map would cost another 12 blind actions on a
            # level whose count has only just started.
        self._prev_levels = levels_completed

    # ---- action selection ------------------------------------------------

    def _candidates(self, grid, legal: Sequence[str]) -> List[ActionKey]:
        out: List[ActionKey] = [(n, -1, -1) for n in SIMPLE_ORDER if n in legal]
        if "ACTION6" in legal:
            if not self._pool:
                self._pool = click_candidates(grid)
            out.extend(("ACTION6", x, y) for x, y in self._pool)
        return out

    def _pick_careful(self, grid, legal: Sequence[str], state_hash: int) -> ActionKey:
        """Spend the level's cheap actions on things not already known useless."""
        # A move that just changed the board is the best next move: these are
        # grid worlds and progress comes in runs of the same direction.
        if (
            self._last is not None
            and self._last_changed
            and self._last[0] in legal
            and (state_hash, self._last) not in self._noop
            and self._rng.random() < self.sticky
        ):
            return self._last

        fresh = [a for a in self._candidates(grid, legal)
                 if (state_hash, a) not in self._noop]
        if not fresh:
            # Every option is a known no-op from this exact board. Nothing here
            # can progress; take the cheapest legal action and let the state
            # change through the game's own dynamics.
            options = [a for a in self._candidates(grid, legal)] or [RESET]
            return options[0]

        untried = [a for a in fresh if a not in self._stats]
        if untried:
            return untried[0]
        return max(fresh, key=lambda a: self._stats[a].rate)

    def _pick_reckless(self, grid, legal: Sequence[str]) -> ActionKey:
        """This level is already worth ~nothing; hunt the next one instead."""
        if (
            self._last is not None
            and self._last[0] in legal
            and self._last != RESET
            and self._rng.random() < self.sticky
        ):
            return self._last
        names = [n for n in legal if n not in ("RESET", "ACTION7")]
        if not names:
            return RESET
        name = self._rng.choice(names)
        if name != "ACTION6":
            return (name, -1, -1)
        if not self._pool:
            self._pool = click_candidates(grid)
        if self._rng.random() < 0.4:
            return ("ACTION6", self._rng.randrange(1, 64, 4), self._rng.randrange(1, 64, 4))
        return ("ACTION6", *self._pool[self._rng.randrange(len(self._pool))])

    # ---- entry point ------------------------------------------------------

    def step(
        self,
        grid,
        legal: Sequence[str],
        levels_completed: int,
        game_over: bool,
        won: bool,
    ) -> Optional[ActionKey]:
        """Next action to submit, or None when this game is finished."""
        self._observe(grid, levels_completed)
        self._prev_grid = grid
        self._prev_hash = hash(grid)

        if won or self.out_of_budget:
            return None

        if game_over or not legal:
            # Only RESET is legal on a game-over frame (docs/actions.md). With
            # actions already taken it is a level reset, so completed levels
            # survive it.
            action = RESET
        elif self.careful:
            action = self._pick_careful(grid, legal, self._prev_hash)
        else:
            action = self._pick_reckless(grid, legal)

        self._last = action
        self.actions_used += 1
        self._since_level += 1
        return action
