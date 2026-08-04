"""Shared, adapter-normalized types used across the zerx package.

Nothing in this module depends on the upstream `arc-agi` package — the
harness adapter (agent/my_agent.py) is the only place that translates
between these types and the real upstream Frame/GameAction types.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet, Optional, Tuple


class ActionName(str, Enum):
    RESET = "RESET"
    ACTION1 = "ACTION1"
    ACTION2 = "ACTION2"
    ACTION3 = "ACTION3"
    ACTION4 = "ACTION4"
    ACTION5 = "ACTION5"
    ACTION6 = "ACTION6"
    ACTION7 = "ACTION7"


@dataclass(frozen=True)
class Action:
    """One validated action ready to send to the harness."""

    name: ActionName
    x: Optional[int] = None
    y: Optional[int] = None

    def __post_init__(self) -> None:
        if self.name == ActionName.ACTION6:
            if self.x is None or self.y is None:
                raise ValueError("ACTION6 requires x and y")
            if not (0 <= self.x <= 63) or not (0 <= self.y <= 63):
                raise ValueError(
                    f"ACTION6 coordinates out of range: ({self.x}, {self.y})"
                )
        elif self.x is not None or self.y is not None:
            raise ValueError(f"{self.name} does not take x/y data")


@dataclass(frozen=True)
class GameFrame:
    """Our internal, adapter-normalized view of one upstream game frame."""

    grid: Tuple[Tuple[int, ...], ...]
    legal_actions: FrozenSet[ActionName]
    is_game_over: bool
    score: int = 0
