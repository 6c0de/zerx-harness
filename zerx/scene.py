"""Duck-informed relational object representation and cross-frame
correspondence/classification (STRATEGY.md SS5, exp-150-duck-tools
Variants A+B). Pure and deterministic -- no model calls, no shared
mutable state. Kept as new, unwired functions per
docs/superpowers/plans/parallel-day3/README.md's additive-only etiquette:
nothing in the live decide() loop calls this module yet.

`boundary` coordinates are in grid-line units (one more than the max cell
index along that axis) -- e.g. a single cell at (1, 1) has boundary
corners at (1, 1), (2, 1), (2, 2), (1, 2), matching a pixel-edge
convention rather than a cell-index one.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

from zerx.perception import LabeledObject, _find_objects
from zerx.types import GameFrame


@dataclass(frozen=True)
class SceneObject:
    object_id: int
    color: int
    area: int
    bbox: Tuple[int, int, int, int]
    centroid: Tuple[float, float]
    boundary: Tuple[Tuple[int, int], ...]
    shape_hash: str
    child_ids: Tuple[int, ...]
    adjacent_ids: Tuple[int, ...]


def _shape_hash(color: int, cells: Sequence[Tuple[int, int]]) -> str:
    """Position-independent color-and-shape hash (STRATEGY.md SS5.3):
    normalizes cells to the object's own bounding-box origin so a moved
    but otherwise identical object hashes the same.
    """
    min_x = min(c[0] for c in cells)
    min_y = min(c[1] for c in cells)
    normalized = tuple(sorted((cx - min_x, cy - min_y) for cx, cy in cells))
    payload = f"{color}:{normalized}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _compute_adjacency(
    objects_cells: Dict[int, FrozenSet[Tuple[int, int]]]
) -> Dict[int, Set[int]]:
    cell_owner: Dict[Tuple[int, int], int] = {}
    for oid, cells in objects_cells.items():
        for cell in cells:
            cell_owner[cell] = oid
    adjacency: Dict[int, Set[int]] = {oid: set() for oid in objects_cells}
    for oid, cells in objects_cells.items():
        for cx, cy in cells:
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                other = cell_owner.get((nx, ny))
                if other is not None and other != oid:
                    adjacency[oid].add(other)
    return adjacency


def _find_children(
    height: int,
    width: int,
    objects_cells: Dict[int, FrozenSet[Tuple[int, int]]],
    parent_id: int,
) -> List[int]:
    """Objects entirely enclosed by `parent_id`'s boundary: flood-fill from
    the grid border, excluding parent cells; anything unreached and not
    part of the parent is enclosed. Any other object whose every cell
    falls in that enclosed area is a child.
    """
    parent_cells = objects_cells[parent_id]
    reachable: Set[Tuple[int, int]] = set()
    stack: List[Tuple[int, int]] = []
    for x in range(width):
        for y in (0, height - 1):
            if (x, y) not in parent_cells:
                stack.append((x, y))
    for y in range(height):
        for x in (0, width - 1):
            if (x, y) not in parent_cells:
                stack.append((x, y))

    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in reachable or (cx, cy) in parent_cells:
            continue
        reachable.add((cx, cy))
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if (
                0 <= nx < width
                and 0 <= ny < height
                and (nx, ny) not in parent_cells
                and (nx, ny) not in reachable
            ):
                stack.append((nx, ny))

    children = []
    for oid, cells in objects_cells.items():
        if oid == parent_id:
            continue
        if cells and all(c not in reachable and c not in parent_cells for c in cells):
            children.append(oid)
    return children


def _trace_boundary(cells: FrozenSet[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """Clockwise outer boundary (STRATEGY.md SS5.3) via unit-edge tracing:
    collect every cell-to-background edge, link them into closed loops,
    keep the loop with the largest *positive* signed area (the outer
    boundary -- hole boundaries trace with the opposite, negative
    orientation and are discarded), then collapse colinear points so the
    result is the minimal corner polygon, not every unit step.
    """
    edges: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for cx, cy in cells:
        if (cx, cy - 1) not in cells:
            edges.append(((cx, cy), (cx + 1, cy)))
        if (cx + 1, cy) not in cells:
            edges.append(((cx + 1, cy), (cx + 1, cy + 1)))
        if (cx, cy + 1) not in cells:
            edges.append(((cx + 1, cy + 1), (cx, cy + 1)))
        if (cx - 1, cy) not in cells:
            edges.append(((cx, cy + 1), (cx, cy)))

    start_map: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for start, end in edges:
        start_map.setdefault(start, []).append(end)

    remaining = set(edges)
    loops: List[List[Tuple[int, int]]] = []
    while remaining:
        first = next(iter(remaining))
        loop = [first[0]]
        current = first
        while True:
            remaining.discard(current)
            next_start = current[1]
            loop.append(next_start)
            if next_start == loop[0]:
                break
            candidates = [
                (next_start, end)
                for end in start_map.get(next_start, [])
                if (next_start, end) in remaining
            ]
            current = candidates[0]
        loops.append(loop[:-1])

    def signed_area(loop: List[Tuple[int, int]]) -> float:
        total = 0.0
        n = len(loop)
        for i in range(n):
            x0, y0 = loop[i]
            x1, y1 = loop[(i + 1) % n]
            total += x0 * y1 - x1 * y0
        return total / 2

    outer = max(loops, key=signed_area)

    def simplify(loop: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if len(loop) <= 2:
            return loop
        simplified = []
        n = len(loop)
        for i in range(n):
            prev = loop[i - 1]
            curr = loop[i]
            nxt = loop[(i + 1) % n]
            dx1, dy1 = curr[0] - prev[0], curr[1] - prev[1]
            dx2, dy2 = nxt[0] - curr[0], nxt[1] - curr[1]
            if dx1 * dy2 - dy1 * dx2 != 0:
                simplified.append(curr)
        return simplified

    return tuple(simplify(outer))


def perceive_scene(frame: GameFrame) -> Tuple[SceneObject, ...]:
    """Reuses zerx.perception's flood-fill segmentation, then adds bbox,
    centroid, clockwise outer boundary, a position-independent
    color-and-shape hash, and containment/adjacency relations.
    """
    labeled: List[LabeledObject] = _find_objects(frame.grid)
    height = len(frame.grid)
    width = len(frame.grid[0]) if height else 0
    objects_cells = {i: frozenset(o.cells) for i, o in enumerate(labeled)}
    adjacency = _compute_adjacency(objects_cells)

    scene_objects = []
    for i, obj in enumerate(labeled):
        area = obj.size
        xs = [c[0] for c in obj.cells]
        ys = [c[1] for c in obj.cells]
        centroid = (sum(xs) / area, sum(ys) / area)
        boundary = _trace_boundary(objects_cells[i])
        shape_hash = _shape_hash(obj.color, obj.cells)
        children = tuple(sorted(_find_children(height, width, objects_cells, i)))
        scene_objects.append(
            SceneObject(
                object_id=i,
                color=obj.color,
                area=area,
                bbox=obj.bbox,
                centroid=centroid,
                boundary=boundary,
                shape_hash=shape_hash,
                child_ids=children,
                adjacent_ids=tuple(sorted(adjacency[i])),
            )
        )
    return tuple(scene_objects)
