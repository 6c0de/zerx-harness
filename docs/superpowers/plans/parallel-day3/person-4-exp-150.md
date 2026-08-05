# Person 4 — `exp-150-duck-tools` Variants A + B (segmentation + fixed analysis tools)

**Read `README.md` in this directory first** — shared context, branch
table, and the shared-file etiquette rules this file assumes.

**Your branch:** `feat/exp-150-duck-tools-ab` (already exists on the
remote, forked from `master`, tests green at fork time).

## What you're building

From `STRATEGY.md` §5 (read all of §5, it's long but every subsection
matters) and the ladder entry in §7:

> `exp-150-duck-tools` — Duck-informed: object segmentation → fixed
> tools → sandboxed recommendation-only Python → state-checked short
> plans. Promote when: improves held-out performance at reasonable cost
> without reducing action safety.

**You are building Variants A and B only** (§5.6's own recommended order,
steps 1–4): relational object segmentation, object correspondence +
change classification, compact fixed analysis functions. **Variant C**
(sandboxed model-written Python) **and Variant D** (state-checked plans)
are explicitly NOT your scope — §5.6 itself sequences them after A/B
("only after Variant C succeeds" for D; Variant C itself is step 5–6 of
the recommended order, after A/B are done and evaluated). Building a code
sandbox is a much bigger, higher-risk undertaking than the other 3
tracks' scope and deliberately excluded from this parallel round.

## Design constraints from STRATEGY.md (read §5.3–§5.5 before starting)

- §5.3: "The raw numeric grid is deliberately hidden from model-written
  Python" — not directly your concern (no Python execution in your
  scope), but it signals the general design philosophy: give the model
  compact, structured summaries, never raw dumps.
- §5.4's honest limitation, already documented for the *current* codebase:
  `zerx/transitions.py`'s `TransitionRecord.effective` can't currently
  distinguish a real gameplay change from a HUD-only animation (a
  shrinking timer bar counts as "effective" today). **This is explicitly
  your fix to make** — §5.4: "fixing it properly needs object-level
  correspondence (§5.3), which is explicitly `exp-150-duck-tools` scope."
- §5.5 point 4: "Compact, fixed analysis API — not unrestricted repository
  access... Return compact summaries; never dump a full 64×64 grid into
  tool output unless a narrow diagnostic explicitly needs it." Applies to
  the fixed-tools API you're building (Variant B) even though there's no
  sandboxed code calling it yet in your scope.

## Interfaces you're producing

**1. Relational object representation** — extend `zerx/perception.py`.
§5.3's target shape (verbatim, adapted from Duck's actual object schema):

```python
@dataclass(frozen=True)
class SceneObject:
    object_id: int
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    boundary: tuple[tuple[int, int], ...]
    shape_hash: str
    child_ids: tuple[int, ...]
    adjacent_ids: tuple[int, ...]
```

§5.3: "Keep both a shape-only hash (movement tracking) and a contextual
signature (affordance evidence) — don't conflate them." `shape_hash`
should be **position-independent** (a same-shaped object that moved
should hash the same) — this is what makes cross-frame correspondence
possible. Existing `LabeledObject` (Day 1, `zerx/perception.py`) already
gives you connected-component segmentation (4-connectivity) and a `label`;
`SceneObject` extends what it captures per object, it doesn't replace the
existing flood-fill algorithm — reuse `_find_objects`'s logic, don't
reimplement segmentation from scratch. Decide whether `SceneObject`
replaces `LabeledObject` in `perceive()`'s return type or is a new,
richer, opt-in representation computed by a separate function (e.g.
`perceive_scene(frame) -> SceneResult` alongside the existing
`perceive()`) — given `zerx/heuristics.py`'s `rank_click_candidates` and
`zerx/policy.py`'s `decide()` both currently consume `PerceptionResult`/
`LabeledObject` directly, changing that return type is a bigger blast
radius than adding a new, parallel, opt-in function. Strongly prefer the
additive approach given `README.md`'s etiquette rules.

**Child/adjacency computation**: `child_ids` (objects enclosed by another
object's boundary) and `adjacent_ids` (edge-sharing neighbors) need real
geometric logic against the segmented objects — this is genuine,
non-trivial work, budget real time for it. Test with grids that have
nested objects (a colored ring around a different-colored center) and
grids with touching same-size objects of different colors.

**2. Object correspondence across frames** — §5.5 point 2: match objects
between two frames "by shape hash, color, overlap, centroid displacement,
bbox proximity, area change, containment/adjacency context, edge contact;
never equate identical hashes with identical semantic roles when
duplicates exist." New function, e.g.:

```python
def correspond_objects(
    before: tuple[SceneObject, ...], after: tuple[SceneObject, ...]
) -> dict[int, Optional[int]]:
    """Maps each `before` object_id to its best-guess `after` object_id
    (or None if it disappeared). Handle the duplicate-shape-hash case
    explicitly -- STRATEGY.md is explicit this must not silently pick
    wrong when two identical-looking objects exist."""
    ...
```

**3. Gameplay-change classification** — §5.4's exact taxonomy:

```text
NO_CHANGE
HUD_ONLY
OBJECT_MOVE
OBJECT_APPEAR_DISAPPEAR
RECOLOR_OR_TRANSFORM
LEVEL_BOUNDARY
TERMINAL
UNKNOWN_CHANGE
```

```python
def classify_transition(
    before: tuple[SceneObject, ...],
    after: tuple[SceneObject, ...],
    correspondence: dict[int, Optional[int]],
    terminal: bool,
    level_delta: int,
) -> str:  # one of the taxonomy values above
    ...
```

§5.4's own honest caveat applies to you too: "a shrinking edge bar is
never, by itself, proof a puzzle action succeeded" — don't build a
classifier that's falsely confident. `HUD_ONLY` vs `OBJECT_MOVE` in
particular is a real, hard, somewhat heuristic distinction (small,
edge-adjacent, low-area objects that change every frame regardless of
action are a reasonable HUD signal — document whatever heuristic you use,
its false-positive/negative risk, and keep it conservative: prefer
`UNKNOWN_CHANGE` over a wrong confident classification).

**4. Fixed analysis API (Variant B)** — §5.5 point 4's list, adapted (no
model-written-code caller exists yet in your scope, but build the
functions as if one will call them later):

```python
def list_salient_objects(scene: tuple[SceneObject, ...]) -> tuple[SceneObject, ...]:
    """Small/rare/high-contrast objects first -- reuses
    zerx/heuristics.py's rank_click_candidates scoring logic where
    sensible; don't duplicate that scoring from scratch, import and reuse
    it or the pieces of it that apply."""
    ...

def compare_frames(before: tuple[SceneObject, ...], after: tuple[SceneObject, ...]) -> str:
    """Compact text summary of what changed -- for prompt inclusion."""
    ...

def find_correspondences(before, after) -> dict[int, Optional[int]]:
    """Same as correspond_objects above -- this IS that function, listed
    here because it's part of the fixed-tools API surface too."""
    ...

def inspect_local_crop(frame: GameFrame, bbox: tuple[int, int, int, int]) -> str:
    """A small region as compact text/grid -- never the full 64x64 grid,
    per SS5.5 point 4."""
    ...
```

Keep every function pure, deterministic, and independently testable — no
model calls anywhere in this track.

## Config field

Add to `zerx/config.py` (end of field list, default preserves current
behavior):

```python
duck_objects_on: bool = False
```

(§5.6's suggested config block lists `duck_fixed_tools_on`,
`duck_python_on`, `duck_short_plan_on` too, but those gate Variants B/C/D
respectively — since you're building B's *functions* but not necessarily
wiring them into the live decision loop, use your judgment on whether you
need a second flag for "the fixed-tools API is available for later use"
vs. just shipping the functions ungated since they're pure/inert until
someone calls them. Document your choice.)

## Wiring into `agent/my_agent.py` / `zerx/transitions.py`

If you wire `classify_transition` into the live loop (e.g. to improve
`TransitionRecord.effective`'s HUD-blindness per §5.4), follow
`README.md`'s rule: this is exactly the kind of change that could tempt
you to modify `zerx/transitions.py`'s existing `TransitionRecord`/
`_diff` — **don't restructure them.** Prefer adding a new, optional field
or a separate function that classifies alongside the existing effective
computation, gated by your config flag, so `TransitionRecord.effective`'s
existing behavior is provably unchanged when your feature is off (every
other track and the existing test suite depends on that).

## Tests

New file(s) — `tests/test_scene_objects.py` and
`tests/test_transition_classification.py` (or your own naming), covering
exactly the scenarios §5.6's "Required tests" paragraph lists for
segmentation and transition evidence:

- Border components, nested objects (containment → `child_ids`),
  adjacent same/different-color objects, duplicate shapes, holes/thin
  boundaries, hash stability under translation (move an object, same
  `shape_hash`), distinct contextual identities for duplicate hashes
  (two identical-looking objects must not get merged into one identity),
  timer/HUD-like edge strips.
- True no-op, HUD-only change, movement, recoloring, animation frames,
  reset, level completion, game over, legal-action changes — one test per
  taxonomy value in `classify_transition`.

Confirm the full existing suite (136 tests) still passes with your
feature's wiring off (if you did wire anything into the live loop) or
unconditionally (if you kept everything as new, unwired, pure functions).

## Explicitly out of scope

- Variant C (sandboxed model-written Python) and Variant D (state-checked
  short plans) — see above, deliberately excluded from this round.
- Any actual code-execution sandbox, even a stub — building sandbox
  infrastructure without the Python-execution feature it's for is wasted
  effort; don't start it.
- Anything from `baseline-115`, `baseline-130`, or `exp-140`.
