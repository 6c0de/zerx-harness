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
    """Clockwise outer boundary (STRATEGY.md SS5.3) via wall-following
    contour tracing: starts from the topmost-then-leftmost eastward-facing
    edge (always on the true outer boundary, never a hole), then at each
    vertex picks the next unit edge by a fixed turn-priority -- right turn,
    then straight, then left turn, then back -- which is the standard,
    deterministic technique for tracing a pixel region's outer contour
    without depending on incidental iteration order. Colinear points are
    then collapsed so the result is the minimal corner polygon.

    If a hole touches the outer boundary at exactly one grid vertex (a
    "pinch"), that vertex appears twice in the returned sequence -- a
    valid degenerate simple closed curve, not corrupted data. Consumers
    that need a strictly-simple polygon should treat a repeated vertex as
    a seam, not an error.
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

    east_starts = [s for (s, e) in edges if e[0] > s[0] and e[1] == s[1]]
    start_vertex = min(east_starts, key=lambda v: (v[1], v[0]))

    turn_right = {(1, 0): (0, 1), (0, 1): (-1, 0), (-1, 0): (0, -1), (0, -1): (1, 0)}
    turn_left = {v: k for k, v in turn_right.items()}
    turn_back = {(1, 0): (-1, 0), (-1, 0): (1, 0), (0, 1): (0, -1), (0, -1): (0, 1)}

    def direction(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
        return (b[0] - a[0], b[1] - a[1])

    loop = [start_vertex]
    current_vertex = start_vertex
    current_dir = (1, 0)
    used: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    while True:
        candidates = start_map.get(current_vertex, [])
        priority = [
            turn_right[current_dir],
            current_dir,
            turn_left[current_dir],
            turn_back[current_dir],
        ]
        next_end = None
        for want_dir in priority:
            for end in candidates:
                if (current_vertex, end) in used:
                    continue
                if direction(current_vertex, end) == want_dir:
                    next_end = end
                    break
            if next_end is not None:
                break
        used.add((current_vertex, next_end))
        current_dir = direction(current_vertex, next_end)
        current_vertex = next_end
        if current_vertex == start_vertex:
            break
        loop.append(current_vertex)

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

    return tuple(simplify(loop))


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
