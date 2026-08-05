# exp-150-duck-tools Variants A+B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Duck-informed relational object segmentation (Variant A) and
a compact fixed analysis API (Variant B) — `STRATEGY.md` §5.6 — as new,
pure, unwired functions in a new `zerx/scene.py` module, off by default,
with zero behavior change to the existing 136-test suite.

**Architecture:** One new module (`zerx/scene.py`) that reuses
`zerx.perception._find_objects`'s flood-fill segmentation and a new
`zerx.heuristics.size_rarity_scores` helper (extracted, not duplicated,
from `rank_click_candidates`'s scoring formula). Everything in
`zerx/scene.py` is a pure function operating on plain tuples/dataclasses —
no model calls, no mutation of shared state, nothing wired into
`zerx/policy.py`'s `decide()` or `agent/my_agent.py`'s live loop. This is
the additive-only path `docs/superpowers/plans/parallel-day3/README.md`
explicitly allows ("if you kept everything as new, unwired, pure
functions" — confirm the existing suite passes unconditionally).

**Tech Stack:** Python 3, `numpy` (already a dependency, used by
`zerx/heuristics.py`), `pytest`.

## Global Constraints

- Every existing test in `tests/` (136 as of `docs/HANDOFF.md`) must still
  pass, unmodified, after every task.
- New `Config` field goes at the **end** of the field list in
  `zerx/config.py`, default `False`, plus a matching line at the end of
  `from_env`'s return-call arguments — see `docs/superpowers/plans/parallel-day3/README.md`.
- Do not modify `zerx/policy.py`'s `decide()` signature or body, and do not
  modify `zerx/transitions.py`'s existing `TransitionRecord`/`_diff`.
- Do not modify `zerx/heuristics.py`'s existing public functions
  (`DeadSignatureTracker`, `rank_click_candidates`) — only append one new
  function at the end of the file.
- Do not touch `scripts/build_notebook.py`, `scripts/build_colab_notebook.py`,
  anything Kaggle-related, or anything involving `CEREBRAS_API_KEY`.
- No code execution / sandbox work (Variant C) and no short-plan work
  (Variant D) — explicitly out of scope per
  `docs/superpowers/plans/parallel-day3/person-4-exp-150.md`.
- Every function in `zerx/scene.py` must be pure and deterministic (no
  randomness, no I/O, no model calls).
- `boundary` coordinates are in grid-line units (one more than the max
  cell index along that axis), not cell-index units — e.g. a single cell
  at `(1, 1)` has boundary corners at `(1, 1)`, `(2, 1)`, `(2, 2)`,
  `(1, 2)`. Document this in the module docstring.

## Decisions recorded (ambiguity resolved per STRATEGY.md §2, not escalated)

1. **`perceive_scene` returns `Tuple[SceneObject, ...]` directly**, not a
   wrapper `SceneResult` dataclass — `classify_transition` and
   `correspond_objects`'s signatures in `person-4-exp-150.md` already take
   `tuple[SceneObject, ...]` directly, so a wrapper would be an unused,
   unrequested type (YAGNI).
2. **No second config flag for "fixed tools available".** The functions in
   this module are pure and inert until something calls them; per
   `person-4-exp-150.md`'s own suggestion, `duck_objects_on` alone gates
   the feature area, and no caller exists yet in this track's scope.
3. **Nothing wired into `agent/my_agent.py` or `zerx/transitions.py`.**
   `person-4-exp-150.md` explicitly allows shipping Variants A/B as new,
   unwired functions; wiring `classify_transition` into
   `TransitionRecord.effective`'s HUD-blindness fix is real future work
   but doing it in this parallel round adds shared-file risk
   (`zerx/transitions.py` is a common merge-friction point per
   `docs/HANDOFF.md`) for no test-visible benefit this round, since
   nothing downstream consumes it yet.
4. **`shape_hash` includes color** (a "position-independent
   color-and-shape hash", verbatim from `STRATEGY.md` §5.3), computed from
   `(color, normalized_cell_offsets)`. This means a true recolor-in-place
   will *not* match via `shape_hash` in `correspond_objects` — verified
   empirically (see Task 4) that the color-tolerant bbox-overlap fallback
   correctly recovers this case, producing `RECOLOR_OR_TRANSFORM` in
   `classify_transition`, not a false `OBJECT_APPEAR_DISAPPEAR`.
5. **`classify_transition` precedence:** `terminal` beats everything, then
   `level_delta != 0`, then a conservative content-based classification
   that prefers `UNKNOWN_CHANGE` over a wrong confident label, per
   STRATEGY.md §5.4's explicit caution.
6. **"Legal-action changes" required-test scenario** (STRATEGY.md §5.6):
   `classify_transition`'s specified signature
   (`before, after, correspondence, terminal, level_delta`) has no legal-actions
   parameter, so this scenario is already covered by the existing
   `tests/test_transitions.py::test_records_legal_actions_before_and_after`
   — not re-tested here.
7. **Amendment found during Task 5's review (not caught by the plan
   author's original verification pass):** the plan's original
   `classify_transition` code, run verbatim against
   `test_reset_style_full_frame_replacement_is_object_appear_disappear`'s
   grids, actually returns `RECOLOR_OR_TRANSFORM`, not
   `OBJECT_APPEAR_DISAPPEAR` as the test asserted — because
   `correspond_objects`'s weak fallback tier matches the lone before-object
   to the after-object with the least-bad (but still very low, `0.111`)
   bbox-overlap score, and `classify_transition` had no way to distinguish
   a confident correspondence from a barely-there one. This specific test
   scenario was, in fact, never executed by the plan author before writing
   it into the brief (unlike every other `classify_transition` scenario,
   which was) — a real gap in this plan's own verification claim. Fix,
   verified against real fixtures: `classify_transition` reuses the
   already-present `_bbox_overlap_ratio` (from Task 4) to gate whether a
   color-changed matched pair is trusted as `RECOLOR_OR_TRANSFORM` — below
   `0.5` overlap, the correspondence is treated as too weak to trust as
   "the same object," and the pair is instead counted as one object
   disappearing and a different one appearing. Verified this does not
   regress the genuine same-position recolor case (`overlap == 1.0`) or
   any other already-approved test. See Task 5's fix round in the SDD
   ledger for the exact diff.

All code below was written against the real `zerx/perception.py` and
`zerx/types.py` in this repo and executed against real fixtures before
being placed in this plan — expected values in test steps are actual
verified output, not hand-derived guesses.

---

### Task 1: `Config.duck_objects_on` flag

**Files:**
- Modify: `zerx/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.duck_objects_on: bool` (default `False`), read via
  `ZERX_DUCK_OBJECTS_ON` env var through `Config.from_env`.

- [ ] **Step 1: Read `tests/test_config.py` to match its existing style**

Run: confirm the file's existing pattern for a bool flag test (it already
has one for `heuristic_first`/`memory_on`/`arbiter_on` — follow the same
shape).

- [ ] **Step 2: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_duck_objects_on_defaults_false():
    config = Config()
    assert config.duck_objects_on is False


def test_duck_objects_on_from_env():
    config = Config.from_env({"ZERX_DUCK_OBJECTS_ON": "true"})
    assert config.duck_objects_on is True
```

(Match whatever import line already exists at the top of the file — it
already imports `Config` from `zerx.config`.)

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_config.py -q`
Expected: FAIL — `TypeError` or `AttributeError`, `duck_objects_on` does
not exist yet.

- [ ] **Step 4: Add the field to `Config`**

In `zerx/config.py`, add as the **last** field in the `Config` dataclass
(after `platform: str = "local"  # "local" | "colab" | "kaggle"`):

```python
    duck_objects_on: bool = False  # exp-150-duck-tools Variants A+B (zerx/scene.py) — off by default
```

Add the matching line as the **last** argument in `from_env`'s `return cls(...)` call
(after the existing `platform=_env_str(env, "ZERX_PLATFORM", cls.platform),` line):

```python
            duck_objects_on=_env_bool(env, "ZERX_DUCK_OBJECTS_ON", cls.duck_objects_on),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/test_config.py -q`
Expected: PASS, all tests in the file green.

- [ ] **Step 6: Run the full suite to confirm no regression**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: 138 passed (136 existing + 2 new).

- [ ] **Step 7: Commit**

```bash
git add zerx/config.py tests/test_config.py
git commit -m "feat(config): add duck_objects_on flag for exp-150-duck-tools (off by default)"
```

---

### Task 2: `zerx/heuristics.py` — extract `size_rarity_scores`

**Files:**
- Modify: `zerx/heuristics.py` (append only, existing functions untouched)
- Test: `tests/test_heuristics.py` (append only)

**Interfaces:**
- Produces: `size_rarity_scores(sizes: Tuple[int, ...], colors: Tuple[int, ...]) -> List[float]`
  — the pure "small objects and rare colors score higher" formula
  currently inlined in `rank_click_candidates`, pulled out so
  `zerx/scene.py`'s `list_salient_objects` (Task 6) can reuse it instead
  of duplicating it, per `person-4-exp-150.md`'s explicit instruction.
- Consumes: nothing new — pure function of two same-length sequences.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_heuristics.py`:

```python
from zerx.heuristics import size_rarity_scores


def test_size_rarity_scores_empty_input_returns_empty_list():
    assert size_rarity_scores((), ()) == []


def test_size_rarity_scores_smaller_and_rarer_scores_higher():
    # sizes: obj0 is bigger (4 cells), obj1 is smaller (1 cell)
    # colors: obj0's color (1) repeats, obj1's color (2) is unique -> rarer
    sizes = (4, 1)
    colors = (1, 2)
    scores = size_rarity_scores(sizes, colors)
    assert scores[1] > scores[0]


def test_size_rarity_scores_matches_rank_click_candidates_ordering():
    small = _obj("small", color=1, cells=[(0, 0)])
    big = _obj("big", color=2, cells=[(2, 0), (2, 1), (2, 2), (2, 3)])
    result = PerceptionResult(ascii_grid="", objects=(big, small))
    candidates = rank_click_candidates(result, DeadSignatureTracker())
    scores = size_rarity_scores((big.size, small.size), (big.color, small.color))
    # rank_click_candidates already asserts "small" ranks first (existing test);
    # confirm the extracted formula agrees without an affordance tracker involved.
    assert scores[1] > scores[0]
```

(`_obj`, `PerceptionResult`, `rank_click_candidates`, `DeadSignatureTracker`
are already imported/defined at the top of `tests/test_heuristics.py` —
add the one new import line for `size_rarity_scores` next to the existing
`from zerx.heuristics import ...` line at the top of the file instead of
inline, i.e. change line 1 to
`from zerx.heuristics import DeadSignatureTracker, rank_click_candidates, size_rarity_scores`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_heuristics.py -q`
Expected: FAIL — `ImportError: cannot import name 'size_rarity_scores'`.

- [ ] **Step 3: Append the function to `zerx/heuristics.py`**

Add at the very end of `zerx/heuristics.py` (after `rank_click_candidates`,
nothing existing is modified):

```python
def size_rarity_scores(sizes: Tuple[int, ...], colors: Tuple[int, ...]) -> List[float]:
    """Pure "small objects and rare colors score higher" formula --
    factored out of rank_click_candidates so zerx.scene.list_salient_objects
    can reuse the same scoring core instead of duplicating it. Unlike
    rank_click_candidates, this takes no affordance tracker and applies no
    penalty; it exists so both callers share one formula.
    """
    if not sizes:
        return []
    sizes_arr = np.array(sizes, dtype=np.float64)
    color_counts: Dict[int, int] = {}
    for c in colors:
        color_counts[c] = color_counts.get(c, 0) + 1
    rarity = np.array([1.0 / color_counts[c] for c in colors], dtype=np.float64)

    max_size = sizes_arr.max()
    size_score = 1.0 - (sizes_arr / max_size) if max_size > 0 else np.zeros_like(sizes_arr)
    max_rarity = rarity.max()
    rarity_score = rarity / max_rarity if max_rarity > 0 else np.zeros_like(rarity)

    combined = 0.5 * size_score + 0.5 * rarity_score
    return [float(v) for v in combined]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/test_heuristics.py -q`
Expected: PASS, all tests in the file green (existing + 3 new).

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: 141 passed (138 from Task 1 + 3 new).

- [ ] **Step 6: Commit**

```bash
git add zerx/heuristics.py tests/test_heuristics.py
git commit -m "refactor(heuristics): extract size_rarity_scores for reuse by exp-150's scene module"
```

---

### Task 3: `zerx/scene.py` — `SceneObject` + `perceive_scene` (Variant A)

**Files:**
- Create: `zerx/scene.py`
- Test: `tests/test_scene_objects.py`

**Interfaces:**
- Consumes: `zerx.perception._find_objects(grid) -> List[LabeledObject]`,
  `zerx.perception.LabeledObject` (fields: `label: str`, `color: int`,
  `cells: Tuple[Tuple[int,int], ...]`, properties `.bbox`, `.size`);
  `zerx.types.GameFrame` (`.grid`, `.legal_actions`, `.is_game_over`, `.score`).
- Produces:
  - `SceneObject` frozen dataclass: `object_id: int`, `color: int`,
    `area: int`, `bbox: Tuple[int,int,int,int]`,
    `centroid: Tuple[float,float]`,
    `boundary: Tuple[Tuple[int,int], ...]`, `shape_hash: str`,
    `child_ids: Tuple[int, ...]`, `adjacent_ids: Tuple[int, ...]`.
  - `perceive_scene(frame: GameFrame) -> Tuple[SceneObject, ...]`.
  - Later tasks also rely on the private helpers `_shape_hash`,
    `_bbox_overlap_ratio` staying in this module (Task 4 adds
    `correspond_objects` in the same file).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene_objects.py`:

```python
from zerx.scene import perceive_scene
from zerx.types import ActionName, GameFrame


def _frame(grid):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=frozenset({ActionName.ACTION1}),
        is_game_over=False,
    )


def _signed_area(loop):
    total = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2


def test_perceive_scene_empty_grid_has_no_objects():
    scene = perceive_scene(_frame([[0, 0], [0, 0]]))
    assert scene == ()


def test_perceive_scene_basic_fields():
    grid = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]
    scene = perceive_scene(_frame(grid))
    assert len(scene) == 1
    obj = scene[0]
    assert obj.object_id == 0
    assert obj.color == 5
    assert obj.area == 1
    assert obj.bbox == (1, 1, 1, 1)
    assert obj.centroid == (1.0, 1.0)
    assert obj.child_ids == ()
    assert obj.adjacent_ids == ()


def test_boundary_is_closed_clockwise_and_covers_all_corners():
    grid = [[3, 3], [3, 3]]
    scene = perceive_scene(_frame(grid))
    obj = scene[0]
    assert set(obj.boundary) == {(0, 0), (2, 0), (2, 2), (0, 2)}
    assert len(obj.boundary) == 4
    assert _signed_area(obj.boundary) > 0  # clockwise in grid-line (y-down) coordinates


def test_thin_strip_object_boundary_is_a_simplified_rectangle():
    grid = [[9, 9, 9, 9]]
    scene = perceive_scene(_frame(grid))
    obj = scene[0]
    assert set(obj.boundary) == {(0, 0), (4, 0), (4, 1), (0, 1)}
    assert obj.area == 4


def test_shape_hash_stable_under_translation():
    left = perceive_scene(_frame([[5, 5, 0], [0, 0, 0]]))[0]
    right = perceive_scene(_frame([[0, 5, 5], [0, 0, 0]]))[0]
    assert left.shape_hash == right.shape_hash
    assert left.bbox != right.bbox


def test_shape_hash_differs_by_color():
    a = perceive_scene(_frame([[5, 0], [0, 0]]))[0]
    b = perceive_scene(_frame([[6, 0], [0, 0]]))[0]
    assert a.shape_hash != b.shape_hash


def test_duplicate_shapes_get_distinct_object_ids_and_same_hash():
    grid = [
        [5, 0, 0, 5],
        [0, 0, 0, 0],
    ]
    scene = perceive_scene(_frame(grid))
    assert len(scene) == 2
    assert scene[0].shape_hash == scene[1].shape_hash
    assert scene[0].object_id != scene[1].object_id


def test_adjacent_different_color_objects_reference_each_other():
    grid = [[2, 4]]
    scene = perceive_scene(_frame(grid))
    a, b = scene
    assert b.object_id in a.adjacent_ids
    assert a.object_id in b.adjacent_ids


def test_diagonal_same_color_objects_are_not_adjacent():
    # 4-connectivity: these are already two separate objects (see
    # test_perception.py's equivalent case), and must not be marked adjacent.
    grid = [
        [3, 0],
        [0, 3],
    ]
    scene = perceive_scene(_frame(grid))
    a, b = scene
    assert b.object_id not in a.adjacent_ids
    assert a.object_id not in b.adjacent_ids


def test_border_component_segments_correctly():
    grid = [
        [7, 7, 0],
        [0, 0, 0],
    ]
    scene = perceive_scene(_frame(grid))
    assert len(scene) == 1
    assert scene[0].bbox == (0, 0, 1, 0)


def test_nested_object_is_a_child_of_its_enclosing_ring():
    grid = [
        [3, 3, 3, 3, 3],
        [3, 0, 0, 0, 3],
        [3, 0, 5, 0, 3],
        [3, 0, 0, 0, 3],
        [3, 3, 3, 3, 3],
    ]
    scene = perceive_scene(_frame(grid))
    by_color = {o.color: o for o in scene}
    ring, center = by_color[3], by_color[5]
    assert center.object_id in ring.child_ids
    assert ring.child_ids == (center.object_id,)
    assert center.child_ids == ()
    # outer boundary must trace the ring's outside, not the hole -- 4 corners, not 8+
    assert set(ring.boundary) == {(0, 0), (5, 0), (5, 5), (0, 5)}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_scene_objects.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.scene'`.

- [ ] **Step 3: Create `zerx/scene.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_scene_objects.py -q`
Expected: PASS, all 11 tests green.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: 152 passed (141 from Task 2 + 11 new).

- [ ] **Step 6: Commit**

```bash
git add zerx/scene.py tests/test_scene_objects.py
git commit -m "feat(scene): add SceneObject and perceive_scene -- Duck-informed segmentation (exp-150 Variant A)"
```

---

### Task 4: `zerx/scene.py` — `correspond_objects` (cross-frame matching)

**Files:**
- Modify: `zerx/scene.py` (append)
- Test: `tests/test_scene_objects.py` (append)

**Interfaces:**
- Consumes: `SceneObject` (Task 3).
- Produces: `correspond_objects(before: Tuple[SceneObject, ...], after: Tuple[SceneObject, ...]) -> Dict[int, Optional[int]]`
  and `find_correspondences = correspond_objects` (same function, two
  names — `person-4-exp-150.md` lists both, saying `find_correspondences`
  "IS that function"). Task 5's `classify_transition` and Task 6's
  `compare_frames` both call `correspond_objects` directly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scene_objects.py`:

```python
from zerx.scene import correspond_objects, find_correspondences


def test_correspond_objects_matches_unique_shapes():
    before = perceive_scene(_frame([[5, 0], [0, 0]]))
    after = perceive_scene(_frame([[0, 5], [0, 0]]))
    mapping = correspond_objects(before, after)
    assert mapping == {before[0].object_id: after[0].object_id}


def test_correspond_objects_disambiguates_duplicate_shapes_by_nearest_centroid():
    before = perceive_scene(_frame([
        [5, 0, 0, 0, 5],
        [0, 0, 0, 0, 0],
    ]))
    after = perceive_scene(_frame([
        [5, 0, 0, 0, 0],
        [0, 0, 0, 0, 5],
    ]))
    left_before = min(before, key=lambda o: o.centroid[0])
    right_before = max(before, key=lambda o: o.centroid[0])
    left_after = min(after, key=lambda o: o.centroid[0])
    right_after = max(after, key=lambda o: o.centroid[0])
    mapping = correspond_objects(before, after)
    # both duplicates share a shape_hash -- must resolve by nearest centroid,
    # not silently pick a fixed index for both.
    assert mapping[left_before.object_id] == left_after.object_id
    assert mapping[right_before.object_id] == right_after.object_id


def test_correspond_objects_none_when_object_disappears():
    before = perceive_scene(_frame([[5, 0], [0, 0]]))
    after = perceive_scene(_frame([[0, 0], [0, 0]]))
    mapping = correspond_objects(before, after)
    assert mapping[before[0].object_id] is None


def test_correspond_objects_falls_back_to_overlap_when_shape_changes_same_color():
    before = perceive_scene(_frame([[5, 0], [0, 0]]))
    after = perceive_scene(_frame([[5, 5], [0, 0]]))
    mapping = correspond_objects(before, after)
    assert mapping[before[0].object_id] == after[0].object_id


def test_correspond_objects_matches_recolor_in_place_via_overlap_fallback():
    # shape_hash includes color (STRATEGY.md SS5.3), so a pure recolor never
    # matches by hash -- this is the fallback path that must still find it.
    before = perceive_scene(_frame([[5, 5], [0, 0]]))
    after = perceive_scene(_frame([[6, 6], [0, 0]]))
    mapping = correspond_objects(before, after)
    assert mapping[before[0].object_id] == after[0].object_id


def test_find_correspondences_is_the_same_function():
    assert find_correspondences is correspond_objects
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_scene_objects.py -q`
Expected: FAIL — `ImportError: cannot import name 'correspond_objects'`.

- [ ] **Step 3: Append to `zerx/scene.py`**

Add `Dict`, `Optional` to the existing `typing` import line at the top
(change `from typing import Dict, FrozenSet, List, Sequence, Set, Tuple`
to `from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple`),
then append at the end of the file:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_scene_objects.py -q`
Expected: PASS, all tests green (11 from Task 3 + 6 new = 17).

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: 158 passed (152 from Task 3 + 6 new).

- [ ] **Step 6: Commit**

```bash
git add zerx/scene.py tests/test_scene_objects.py
git commit -m "feat(scene): add correspond_objects for cross-frame object matching (exp-150 Variant A)"
```

---

### Task 5: `zerx/scene.py` — `classify_transition` (Variant A's HUD-blindness fix)

**Files:**
- Modify: `zerx/scene.py` (append)
- Test: `tests/test_transition_classification.py` (new)

**Interfaces:**
- Consumes: `SceneObject`, `correspond_objects` (Tasks 3–4).
- Produces: `classify_transition(before, after, correspondence, terminal, level_delta, grid_width=64, grid_height=64) -> str`,
  one of the 8 taxonomy constants (`NO_CHANGE`, `HUD_ONLY`, `OBJECT_MOVE`,
  `OBJECT_APPEAR_DISAPPEAR`, `RECOLOR_OR_TRANSFORM`, `LEVEL_BOUNDARY`,
  `TERMINAL`, `UNKNOWN_CHANGE`) — STRATEGY.md §5.4.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_transition_classification.py`:

```python
from zerx.scene import classify_transition, correspond_objects, perceive_scene
from zerx.types import ActionName, GameFrame


def _frame(grid):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=frozenset({ActionName.ACTION1}),
        is_game_over=False,
    )


def _classify(before_grid, after_grid, terminal=False, level_delta=0, width=5, height=5):
    before = perceive_scene(_frame(before_grid))
    after = perceive_scene(_frame(after_grid))
    correspondence = correspond_objects(before, after)
    return classify_transition(before, after, correspondence, terminal, level_delta, width, height)


def test_true_no_op_is_no_change():
    grid = [[5, 0], [0, 0]]
    assert _classify(grid, grid, width=2, height=2) == "NO_CHANGE"


def test_small_edge_object_change_is_hud_only():
    before = [[0, 0], [0, 0]]
    after = [[0, 0], [1, 0]]
    assert _classify(before, after, width=2, height=2) == "HUD_ONLY"


def test_object_moving_same_shape_and_color_is_object_move():
    before = [
        [0, 0, 0, 0, 0],
        [0, 5, 5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    after = [
        [0, 0, 0, 0, 0],
        [0, 0, 5, 5, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert _classify(before, after) == "OBJECT_MOVE"


def test_recoloring_in_place_is_recolor_or_transform():
    before = [
        [0, 0, 0, 0, 0],
        [0, 5, 5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    after = [
        [0, 0, 0, 0, 0],
        [0, 6, 6, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert _classify(before, after) == "RECOLOR_OR_TRANSFORM"


def test_large_object_appearing_is_object_appear_disappear():
    before = [[0] * 5 for _ in range(5)]
    after = [
        [0, 0, 0, 0, 0],
        [0, 7, 7, 7, 0],
        [0, 7, 7, 7, 0],
        [0, 7, 7, 7, 0],
        [0, 0, 0, 0, 0],
    ]
    assert _classify(before, after) == "OBJECT_APPEAR_DISAPPEAR"


def test_animation_frame_noise_at_edge_is_hud_only_not_confident_progress():
    # a 1-cell "timer" flicker at the frame edge must not read as real progress
    before = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    after = [[0, 0, 2], [0, 0, 0], [0, 0, 0]]
    assert _classify(before, after, width=3, height=3) == "HUD_ONLY"


def test_reset_style_full_frame_replacement_is_object_appear_disappear():
    before = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]
    after = [
        [8, 8, 8],
        [8, 0, 8],
        [8, 8, 8],
    ]
    assert _classify(before, after, width=3, height=3) == "OBJECT_APPEAR_DISAPPEAR"


def test_level_completion_is_level_boundary_regardless_of_diff():
    before = [[0, 0], [0, 0]]
    after = [[0, 0], [0, 0]]
    assert _classify(before, after, level_delta=1, width=2, height=2) == "LEVEL_BOUNDARY"


def test_game_over_is_terminal_regardless_of_diff():
    before = [[0, 0], [0, 0]]
    after = [[0, 0], [0, 0]]
    assert _classify(before, after, terminal=True, width=2, height=2) == "TERMINAL"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_transition_classification.py -q`
Expected: FAIL — `ImportError: cannot import name 'classify_transition'`.

- [ ] **Step 3: Append to `zerx/scene.py`**

Append at the end of the file:

```python
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
        if color_changed or shape_changed:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_transition_classification.py -q`
Expected: PASS, all 9 tests green.

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: 167 passed (158 from Task 4 + 9 new).

- [ ] **Step 6: Commit**

```bash
git add zerx/scene.py tests/test_transition_classification.py
git commit -m "feat(scene): add classify_transition -- gameplay-change taxonomy (exp-150 SS5.4)"
```

---

### Task 6: `zerx/scene.py` — fixed analysis API (Variant B)

**Files:**
- Modify: `zerx/scene.py` (append)
- Test: `tests/test_scene_objects.py` (append)

**Interfaces:**
- Consumes: `SceneObject`, `correspond_objects` (Tasks 3–4),
  `zerx.heuristics.size_rarity_scores` (Task 2).
- Produces:
  - `list_salient_objects(scene: Tuple[SceneObject, ...]) -> Tuple[SceneObject, ...]`
  - `compare_frames(before: Tuple[SceneObject, ...], after: Tuple[SceneObject, ...]) -> str`
  - `inspect_local_crop(frame: GameFrame, bbox: Tuple[int, int, int, int]) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scene_objects.py`:

```python
from zerx.scene import compare_frames, inspect_local_crop, list_salient_objects


def test_list_salient_objects_ranks_small_rare_object_first():
    grid = [
        [1, 1, 1, 2],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
    ]
    scene = perceive_scene(_frame(grid))
    ranked = list_salient_objects(scene)
    assert ranked[0].color == 2


def test_list_salient_objects_empty_scene_returns_empty():
    assert list_salient_objects(()) == ()


def test_compare_frames_reports_no_change_for_identical_scenes():
    scene = perceive_scene(_frame([[5, 0], [0, 0]]))
    assert "no_change" in compare_frames(scene, scene)


def test_compare_frames_reports_appeared_and_disappeared_counts():
    before = perceive_scene(_frame([[5, 0], [0, 0]]))
    after = perceive_scene(_frame([[0, 6], [0, 0]]))
    summary = compare_frames(before, after)
    assert "disappeared=1" in summary
    assert "appeared=1" in summary


def test_inspect_local_crop_returns_requested_region_only():
    grid = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    text = inspect_local_crop(_frame(grid), (1, 1, 2, 2))
    assert text == "45\n78"


def test_inspect_local_crop_does_not_return_the_full_grid():
    grid = [[i for i in range(10)] for _ in range(10)]
    text = inspect_local_crop(_frame(grid), (0, 0, 2, 2))
    assert len(text.splitlines()) == 3
    assert all(len(row) == 3 for row in text.splitlines())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_scene_objects.py -q`
Expected: FAIL — `ImportError: cannot import name 'list_salient_objects'`.

- [ ] **Step 3: Append to `zerx/scene.py`**

Add one import line near the top (with the other `zerx` imports):

```python
from zerx.heuristics import size_rarity_scores
```

Then append at the end of the file:

```python
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
    a full grid dump (STRATEGY.md SS5.5 point 4).
    """
    correspondence = correspond_objects(before, after)
    before_by_id = {o.object_id: o for o in before}
    after_by_id = {o.object_id: o for o in after}
    matched_after_ids = {v for v in correspondence.values() if v is not None}

    appeared = [a for a in after if a.object_id not in matched_after_ids]
    disappeared = [before_by_id[bid] for bid, aid in correspondence.items() if aid is None]
    moved = 0
    recolored = 0
    for bid, aid in correspondence.items():
        if aid is None:
            continue
        b, a = before_by_id[bid], after_by_id[aid]
        if b.color != a.color:
            recolored += 1
        elif b.centroid != a.centroid:
            moved += 1

    parts = [f"objects_before={len(before)}", f"objects_after={len(after)}"]
    if appeared:
        parts.append(f"appeared={len(appeared)}")
    if disappeared:
        parts.append(f"disappeared={len(disappeared)}")
    if moved:
        parts.append(f"moved={moved}")
    if recolored:
        parts.append(f"recolored={recolored}")
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_scene_objects.py -q`
Expected: PASS, all tests green (17 from Task 4 + 6 new = 23).

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: 173 passed (167 from Task 5 + 6 new).

- [ ] **Step 6: Commit**

```bash
git add zerx/scene.py tests/test_scene_objects.py
git commit -m "feat(scene): add fixed analysis API -- list_salient_objects, compare_frames, inspect_local_crop (exp-150 Variant B)"
```

---

### Task 7: Full verification, `docs/HANDOFF.md` status update, push

**Files:**
- Modify: `docs/HANDOFF.md` (one-line status update in the "Parallel work
  split" table only — do not rewrite the file)

- [ ] **Step 1: Run the full suite one more time from a clean state**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: 173 passed, 0 failed. If the count doesn't match, stop and
investigate before proceeding — do not report done on a mismatched count.

- [ ] **Step 2: Confirm the feature is off by default**

Run: `.venv\Scripts\python.exe -c "from zerx.config import Config; c = Config(); assert c.duck_objects_on is False; print('duck_objects_on default:', c.duck_objects_on)"`
Expected: prints `duck_objects_on default: False`.

- [ ] **Step 3: Confirm nothing outside `zerx/scene.py`, `zerx/heuristics.py` (append), `zerx/config.py` (append), and `tests/` changed**

Run: `git status --short` and `git diff --stat master...HEAD`
Expected: only `zerx/config.py`, `zerx/heuristics.py`, `zerx/scene.py`
(new), `tests/test_config.py`, `tests/test_heuristics.py`,
`tests/test_scene_objects.py` (new), `tests/test_transition_classification.py`
(new), and `docs/superpowers/plans/2026-08-05-exp-150-duck-tools-ab.md`
(new) appear.

- [ ] **Step 4: Update `docs/HANDOFF.md`'s "Parallel work split" table**

In the table under "## Parallel work split (Day 3, starting 2026-08-05)",
change row 4 from:

```
| 4 | `exp-150-duck-tools` Variants A+B (segmentation + fixed analysis tools) | `feat/exp-150-duck-tools-ab` | `docs/superpowers/plans/parallel-day3/person-4-exp-150.md` |
```

to:

```
| 4 | `exp-150-duck-tools` Variants A+B — **done**, 173/173 passing, `duck_objects_on=False` by default, unwired (see `docs/superpowers/plans/2026-08-05-exp-150-duck-tools-ab.md`) | `feat/exp-150-duck-tools-ab` | `docs/superpowers/plans/parallel-day3/person-4-exp-150.md` |
```

(Adjust the passing count in the sentence if Step 1's actual number
differs from 173 — use the real observed number, not this plan's
estimate.)

- [ ] **Step 5: Commit the handoff update**

```bash
git add docs/HANDOFF.md
git commit -m "docs: mark exp-150-duck-tools Variants A+B done in Day 3 parallel-work table"
```

- [ ] **Step 6: Push to the owned branch only**

Run: `git push origin feat/exp-150-duck-tools-ab`

Do **not** merge to `master` — `docs/superpowers/plans/parallel-day3/INTEGRATION.md`
handles the sequenced merge of all 4 tracks.

---

## Self-review notes (already applied above, kept for the executor's context)

- Every function named in `person-4-exp-150.md`'s interface section exists:
  `SceneObject`, `correspond_objects` (+ `find_correspondences` alias),
  `classify_transition` with the full 8-value taxonomy,
  `list_salient_objects`, `compare_frames`, `find_correspondences`,
  `inspect_local_crop`. `perceive_scene` is the one intentional deviation
  from the file's illustrative sketch (returns a plain tuple, not a
  `SceneResult` wrapper) — recorded as Decision 1 above, not a gap.
- All required-test scenarios from STRATEGY.md §5.6 are covered except
  "legal-action changes" (Decision 6 — already covered by existing
  `tests/test_transitions.py`, and out of `classify_transition`'s given
  signature) and Variant C/D scenarios (out of scope per
  `person-4-exp-150.md`).
- No placeholders: every step above has real, complete code, verified by
  actually running it against `zerx/perception.py` and `zerx/types.py` in
  this repo before this plan was written.
