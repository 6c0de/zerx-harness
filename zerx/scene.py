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

from zerx.heuristics import size_rarity_scores
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


def _build_cell_owner(
    objects_cells: Dict[int, FrozenSet[Tuple[int, int]]]
) -> Dict[Tuple[int, int], int]:
    cell_owner: Dict[Tuple[int, int], int] = {}
    for oid, cells in objects_cells.items():
        for cell in cells:
            cell_owner[cell] = oid
    return cell_owner


def _compute_adjacency(
    objects_cells: Dict[int, FrozenSet[Tuple[int, int]]],
    cell_owner: Dict[Tuple[int, int], int],
) -> Dict[int, Set[int]]:
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
    cell_owner: Dict[Tuple[int, int], int],
    parent_id: int,
    parent_bbox: Tuple[int, int, int, int],
) -> List[int]:
    """Objects entirely enclosed by `parent_id`'s boundary: flood-fill from
    the border of `parent_id`'s own bounding box (padded by one cell,
    clamped to the grid), excluding parent cells; anything unreached and
    not part of the parent is enclosed. Any other object whose every cell
    falls in that enclosed area is a child.

    The search is restricted to `parent_id`'s own (padded) bbox rather
    than the whole grid: a cell outside a parent's bbox can always route
    around that bbox to the true grid border without crossing the
    parent's cells, so it can never be "enclosed" by that parent --
    restricting the flood-fill to the bbox is exactly equivalent to a
    full-grid flood-fill for this parent, just far cheaper for small
    objects on a large grid. Candidate children are narrowed via
    `cell_owner` to only objects that actually own a cell inside the
    enclosed region, instead of re-scanning every object in the scene for
    every parent -- this is what keeps `perceive_scene` fast on frames
    with hundreds of objects (previously O(objects x grid_area) per
    parent from an unrestricted scan; now bounded by the parent's own
    bbox area).
    """
    parent_cells = objects_cells[parent_id]
    min_x, min_y, max_x, max_y = parent_bbox
    lo_x, lo_y = max(0, min_x - 1), max(0, min_y - 1)
    hi_x, hi_y = min(width - 1, max_x + 1), min(height - 1, max_y + 1)

    reachable: Set[Tuple[int, int]] = set()
    stack: List[Tuple[int, int]] = []
    for x in range(lo_x, hi_x + 1):
        for y in (lo_y, hi_y):
            if (x, y) not in parent_cells:
                stack.append((x, y))
    for y in range(lo_y, hi_y + 1):
        for x in (lo_x, hi_x):
            if (x, y) not in parent_cells:
                stack.append((x, y))

    while stack:
        cx, cy = stack.pop()
        if (cx, cy) in reachable or (cx, cy) in parent_cells:
            continue
        reachable.add((cx, cy))
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if (
                lo_x <= nx <= hi_x
                and lo_y <= ny <= hi_y
                and (nx, ny) not in parent_cells
                and (nx, ny) not in reachable
            ):
                stack.append((nx, ny))

    candidate_owners: Set[int] = set()
    for x in range(lo_x, hi_x + 1):
        for y in range(lo_y, hi_y + 1):
            cell = (x, y)
            if cell in parent_cells or cell in reachable:
                continue
            owner = cell_owner.get(cell)
            if owner is not None and owner != parent_id:
                candidate_owners.add(owner)

    children = []
    for oid in candidate_owners:
        cells = objects_cells[oid]
        if not all(lo_x <= cx <= hi_x and lo_y <= cy <= hi_y for cx, cy in cells):
            continue
        if all(c not in reachable and c not in parent_cells for c in cells):
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
    cell_owner = _build_cell_owner(objects_cells)
    adjacency = _compute_adjacency(objects_cells, cell_owner)

    scene_objects = []
    for i, obj in enumerate(labeled):
        area = obj.size
        xs = [c[0] for c in obj.cells]
        ys = [c[1] for c in obj.cells]
        centroid = (sum(xs) / area, sum(ys) / area)
        boundary = _trace_boundary(objects_cells[i])
        shape_hash = _shape_hash(obj.color, obj.cells)
        children = tuple(
            sorted(_find_children(height, width, objects_cells, cell_owner, i, obj.bbox))
        )
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


_MIN_MATCH_CONFIDENCE = 0.5


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

    The fallback tier refuses a match below `_MIN_MATCH_CONFIDENCE`
    overlap regardless of color: a weak positional overlap means this is
    the least-bad candidate available, not a confident correspondence,
    and every consumer of this mapping (classify_transition,
    compare_frames) should see the same "no, these are not the same
    object" answer rather than each having to re-derive that judgment
    independently.
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
        if best is not None and best_score >= _MIN_MATCH_CONFIDENCE:
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


def _pair_diff(
    before: Tuple[SceneObject, ...],
    after: Tuple[SceneObject, ...],
    correspondence: Dict[int, Optional[int]],
) -> Tuple[
    List[SceneObject],
    List[SceneObject],
    List[Tuple[SceneObject, SceneObject]],
    List[Tuple[SceneObject, SceneObject]],
]:
    """Shared change-classification core for classify_transition and
    compare_frames -- both need the same disappeared/appeared/moved/
    recolored breakdown, and duplicating the logic in each let them drift
    out of sync (compare_frames originally had no shape-change term at
    all, so a same-position shape transformation silently reported
    "no_change"). Returns (disappeared, appeared, moved, recolored); moved
    and recolored keep the full (before, after) pair, not just the before
    object, so callers can inspect both endpoints -- classify_transition's
    HUD_ONLY check needs both, not just where the object started.
    """
    before_by_id = {o.object_id: o for o in before}
    after_by_id = {o.object_id: o for o in after}
    matched_after_ids = {v for v in correspondence.values() if v is not None}

    disappeared = [before_by_id[bid] for bid, aid in correspondence.items() if aid is None]
    appeared = [a for a in after if a.object_id not in matched_after_ids]

    moved: List[Tuple[SceneObject, SceneObject]] = []
    recolored: List[Tuple[SceneObject, SceneObject]] = []
    for bid, aid in correspondence.items():
        if aid is None:
            continue
        b, a = before_by_id[bid], after_by_id[aid]
        color_changed = b.color != a.color
        shape_changed = b.shape_hash != a.shape_hash and not color_changed
        if color_changed or shape_changed:
            recolored.append((b, a))
        elif b.centroid != a.centroid:
            moved.append((b, a))
    return disappeared, appeared, moved, recolored


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
    when *every* changed object is small and edge-adjacent at *every*
    endpoint it has (both before and after, for a moved/recolored pair --
    an object that moves from the edge into the play field is not
    HUD_ONLY just because it started at the edge); anything less
    clear-cut falls through toward UNKNOWN_CHANGE rather than a falsely
    confident label -- "a shrinking edge bar is never, by itself, proof a
    puzzle action succeeded."

    Low-confidence correspondences (a matched pair that doesn't really
    look like the same object) are filtered out one layer down, by
    correspond_objects itself refusing to report them -- see its
    docstring. That keeps this function's job purely "given a set of
    trusted matches, classify what changed," instead of re-deriving match
    confidence here too.
    """
    if terminal:
        return TERMINAL
    if level_delta != 0:
        return LEVEL_BOUNDARY

    disappeared, appeared, moved, recolored = _pair_diff(before, after, correspondence)

    if not disappeared and not appeared and not moved and not recolored:
        return NO_CHANGE

    def single_is_hud(obj: SceneObject) -> bool:
        return obj.area <= _HUD_MAX_AREA and _touches_edge(obj.bbox, grid_width, grid_height)

    def pair_is_hud(b: SceneObject, a: SceneObject) -> bool:
        return single_is_hud(b) and single_is_hud(a)

    all_hud = (
        all(single_is_hud(o) for o in disappeared)
        and all(single_is_hud(o) for o in appeared)
        and all(pair_is_hud(b, a) for b, a in moved)
        and all(pair_is_hud(b, a) for b, a in recolored)
    )
    if all_hud:
        return HUD_ONLY

    if disappeared or appeared:
        return OBJECT_APPEAR_DISAPPEAR
    if recolored:
        return RECOLOR_OR_TRANSFORM
    if moved:
        return OBJECT_MOVE
    return UNKNOWN_CHANGE


def list_salient_objects(scene: Tuple[SceneObject, ...]) -> Tuple[SceneObject, ...]:
    """Small/rare/high-contrast objects first -- reuses
    zerx.heuristics.size_rarity_scores (the pure scoring core behind
    rank_click_candidates) instead of duplicating that formula.
    """
    if not scene:
        return ()
    sizes = tuple(o.area for o in scene)
    colors = tuple(o.color for o in scene)
    scores = size_rarity_scores(sizes, colors)
    ranked = sorted(zip(scene, scores), key=lambda pair: pair[1], reverse=True)
    return tuple(obj for obj, _ in ranked)


def compare_frames(before: Tuple[SceneObject, ...], after: Tuple[SceneObject, ...]) -> str:
    """Compact text summary of what changed -- for prompt inclusion, never
    a full grid dump (STRATEGY.md SS5.5 point 4). Uses the same
    disappeared/appeared/moved/recolored breakdown as classify_transition
    (via _pair_diff) so the two never disagree about what happened.
    """
    correspondence = correspond_objects(before, after)
    disappeared, appeared, moved, recolored = _pair_diff(before, after, correspondence)

    parts = [f"objects_before={len(before)}", f"objects_after={len(after)}"]
    if appeared:
        parts.append(f"appeared={len(appeared)}")
    if disappeared:
        parts.append(f"disappeared={len(disappeared)}")
    if moved:
        parts.append(f"moved={len(moved)}")
    if recolored:
        parts.append(f"recolored={len(recolored)}")
    if len(parts) == 2:
        parts.append("no_change")
    return ", ".join(parts)


def inspect_local_crop(frame: GameFrame, bbox: Tuple[int, int, int, int]) -> str:
    """A small region as compact hex-encoded text -- never the full 64x64
    grid, per STRATEGY.md SS5.5 point 4. Caller supplies a valid in-range
    bbox; this is an internal analysis helper, not a user-input boundary.
    """
    min_x, min_y, max_x, max_y = bbox
    rows = []
    for y in range(min_y, max_y + 1):
        row = frame.grid[y][min_x : max_x + 1]
        rows.append("".join(f"{cell:x}" if cell < 16 else "?" for cell in row))
    return "\n".join(rows)
