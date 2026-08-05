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
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

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


def _bbox_overlap_ratio(
    a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix0 > ix1 or iy0 > iy1:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def correspond_objects(
    before: Tuple[SceneObject, ...], after: Tuple[SceneObject, ...]
) -> Dict[int, Optional[int]]:
    """Maps each `before` object_id to its best-guess `after` object_id (or
    None if it disappeared). STRATEGY.md SS5.5: match by shape hash first;
    when several before/after objects share a shape hash, disambiguate by
    nearest centroid instead of always picking the same index -- duplicate
    hashes must never collapse into one identity. Objects whose shape hash
    has no match fall back to bbox overlap, preferring same-color matches
    but still allowing a cross-color match (with no bonus) so a pure
    recolor-in-place -- which never matches by hash, since shape_hash
    includes color -- is still found via position instead of being
    reported as one object disappearing and an unrelated one appearing.
    """
    used_after: Set[int] = set()
    result: Dict[int, Optional[int]] = {}

    after_by_hash: Dict[str, List[SceneObject]] = {}
    for obj in after:
        after_by_hash.setdefault(obj.shape_hash, []).append(obj)

    before_by_hash: Dict[str, List[SceneObject]] = {}
    for obj in before:
        before_by_hash.setdefault(obj.shape_hash, []).append(obj)

    for shape_hash, before_group in before_by_hash.items():
        after_group = [
            o for o in after_by_hash.get(shape_hash, []) if o.object_id not in used_after
        ]
        if not after_group:
            continue
        if len(before_group) == 1 and len(after_group) == 1:
            result[before_group[0].object_id] = after_group[0].object_id
            used_after.add(after_group[0].object_id)
            continue
        pairs = []
        for b in before_group:
            for a in after_group:
                dist = (
                    (b.centroid[0] - a.centroid[0]) ** 2
                    + (b.centroid[1] - a.centroid[1]) ** 2
                ) ** 0.5
                pairs.append((dist, b.object_id, a.object_id))
        pairs.sort(key=lambda p: p[0])
        assigned_before: Set[int] = set()
        for _, bid, aid in pairs:
            if bid in assigned_before or aid in used_after:
                continue
            result[bid] = aid
            assigned_before.add(bid)
            used_after.add(aid)

    remaining_after = [o for o in after if o.object_id not in used_after]
    for b in before:
        if b.object_id in result:
            continue
        best = None
        best_score = 0.0
        for a in remaining_after:
            score = _bbox_overlap_ratio(b.bbox, a.bbox)
            if a.color == b.color:
                score *= 1.2
            if score > best_score:
                best_score = score
                best = a
        if best is not None:
            result[b.object_id] = best.object_id
            remaining_after.remove(best)
            used_after.add(best.object_id)
        else:
            result[b.object_id] = None
    return result


find_correspondences = correspond_objects

NO_CHANGE = "NO_CHANGE"
HUD_ONLY = "HUD_ONLY"
OBJECT_MOVE = "OBJECT_MOVE"
OBJECT_APPEAR_DISAPPEAR = "OBJECT_APPEAR_DISAPPEAR"
RECOLOR_OR_TRANSFORM = "RECOLOR_OR_TRANSFORM"
LEVEL_BOUNDARY = "LEVEL_BOUNDARY"
TERMINAL = "TERMINAL"
UNKNOWN_CHANGE = "UNKNOWN_CHANGE"

_HUD_MAX_AREA = 4


def _touches_edge(bbox: Tuple[int, int, int, int], width: int, height: int) -> bool:
    min_x, min_y, max_x, max_y = bbox
    return min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1


def classify_transition(
    before: Tuple[SceneObject, ...],
    after: Tuple[SceneObject, ...],
    correspondence: Dict[int, Optional[int]],
    terminal: bool,
    level_delta: int,
    grid_width: int = 64,
    grid_height: int = 64,
) -> str:
    """STRATEGY.md SS5.4's gameplay-change taxonomy -- deterministic where
    possible, explicitly uncertain otherwise. `terminal` and `level_delta`
    take priority over any pixel-level classification. HUD_ONLY only fires
    when *every* changed object is small and edge-adjacent; anything less
    clear-cut falls through toward UNKNOWN_CHANGE rather than a falsely
    confident label -- "a shrinking edge bar is never, by itself, proof a
    puzzle action succeeded."
    """
    if terminal:
        return TERMINAL
    if level_delta != 0:
        return LEVEL_BOUNDARY

    before_by_id = {o.object_id: o for o in before}
    after_by_id = {o.object_id: o for o in after}
    matched_after_ids = {v for v in correspondence.values() if v is not None}

    disappeared = [before_by_id[bid] for bid, aid in correspondence.items() if aid is None]
    appeared = [a for a in after if a.object_id not in matched_after_ids]

    moved: List[SceneObject] = []
    recolored: List[SceneObject] = []
    for bid, aid in correspondence.items():
        if aid is None:
            continue
        b, a = before_by_id[bid], after_by_id[aid]
        color_changed = b.color != a.color
        shape_changed = b.shape_hash != a.shape_hash and not color_changed
        # If both color AND area changed significantly, likely disappear/appear not recolor
        area_ratio = max(b.area, a.area) / min(b.area, a.area) if min(b.area, a.area) > 0 else float('inf')
        if color_changed and area_ratio > 3:
            disappeared.append(b)
            appeared.append(a)
        elif color_changed or shape_changed:
            recolored.append(b)
        elif b.centroid != a.centroid:
            moved.append(b)

    changed_objects = disappeared + appeared + moved + recolored
    if not changed_objects:
        return NO_CHANGE

    if all(
        obj.area <= _HUD_MAX_AREA and _touches_edge(obj.bbox, grid_width, grid_height)
        for obj in changed_objects
    ):
        return HUD_ONLY

    if disappeared or appeared:
        return OBJECT_APPEAR_DISAPPEAR
    if recolored:
        return RECOLOR_OR_TRANSFORM
    if moved:
        return OBJECT_MOVE
    return UNKNOWN_CHANGE
