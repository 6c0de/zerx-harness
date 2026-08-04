"""Convert a GameFrame (+ trailing history) into a compact, model-ready
representation: an ASCII grid and a list of labeled same-color objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from zerx.types import GameFrame

BACKGROUND_COLOR = 0


@dataclass(frozen=True)
class LabeledObject:
    label: str
    color: int
    cells: Tuple[Tuple[int, int], ...]

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        xs = [c[0] for c in self.cells]
        ys = [c[1] for c in self.cells]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def size(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class PerceptionResult:
    ascii_grid: str
    objects: Tuple[LabeledObject, ...]


def _find_objects(grid: Sequence[Sequence[int]]) -> List[LabeledObject]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    visited = [[False] * width for _ in range(height)]
    objects: List[LabeledObject] = []
    label_counter = 0

    for y in range(height):
        for x in range(width):
            if visited[y][x]:
                continue
            color = grid[y][x]
            visited[y][x] = True
            if color == BACKGROUND_COLOR:
                continue
            stack = [(x, y)]
            cells = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                        if grid[ny][nx] == color:
                            visited[ny][nx] = True
                            cells.append((nx, ny))
                            stack.append((nx, ny))
            objects.append(
                LabeledObject(label=f"obj{label_counter}", color=color, cells=tuple(cells))
            )
            label_counter += 1
    return objects


def _render_ascii(grid: Sequence[Sequence[int]]) -> str:
    return "\n".join(
        "".join(f"{cell:x}" if cell < 16 else "?" for cell in row) for row in grid
    )


def perceive(frame: GameFrame, history: Sequence[GameFrame] = ()) -> PerceptionResult:
    """Render `frame` into an ASCII grid and labeled connected-component
    (4-connectivity) objects. `history` is accepted for interface stability
    (future movement-delta perception) but the baseline only looks at
    `frame` itself.
    """
    objects = _find_objects(frame.grid)
    ascii_grid = _render_ascii(frame.grid)
    return PerceptionResult(ascii_grid=ascii_grid, objects=tuple(objects))
