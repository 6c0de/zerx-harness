# ARC-AGI-3 Local Model-Free Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the official ARC-AGI-3 Kaggle starter kit and build the complete `zerx/` model-free package (perception, heuristics, memory, budget, JSON policy parsing/repair, decide() orchestrator, evidence ledger) plus a thin harness adapter, fully unit-tested on CPU with no model load — the "Local → Colab" promotion gate from `AGENTS.md`.

**Architecture:** A model-free `zerx/` package implements perception, heuristics, memory, budget, JSON policy parsing/repair, and an evidence-first transition ledger against our own internal types (`zerx/types.py`), fully unit-tested without GPU or model access via a `ModelBackend` protocol and `FakeModelBackend` test double. A thin `agent/my_agent.py` adapter — added on top of the vendored official starter — translates the real upstream Frame/GameAction API into these internal types and back, finalizes each transition once the next frame exists, feeds click outcomes back into graded negative affordances, and isolates the one place real-API uncertainty lives. `eval/run_ablation.py` owns the experiment-record schema and config-sweep generator used for ablation once the adapter is wired to real games.

**Tech Stack:** Python 3.12 (starter's `arc-agi` package requirement), pytest, numpy (heuristics scoring only), stdlib `dataclasses`/`json`/`hashlib`/`re`/`random`/`urllib`/`collections`, the official `arcprize/ARC-AGI-3-Kaggle-Starter` framework (exact import paths confirmed in Task 1, not assumed), its `make` workflow (`make setup`, `make play-local`, `make verify-local`), and a development-only Cerebras Inference Cloud backend (`gemma-4-31b`, network API, mocked in the default test suite).

## Strategy alignment

[`STRATEGY.md`](../../../STRATEGY.md) is the prior-art and experiment guide (ReKi, Murad/Forge VLM, ProjectForty2 FORGE, Tycho). This plan implements `baseline-100-minimal` plus the transition-ledger portion of `baseline-110-evidence` — the minimum Zerx foundation, with evidence recording as baseline infrastructure (Task 13) and graded negative affordances (Task 5) since both are cheap and the modules they touch hadn't shipped yet. Structured hypothesis/belief tracking, executable world models, planners, and builder agents are later, explicitly isolated experiments (`baseline-120` onward) — not part of this plan. If a simple module this plan builds (e.g. `zerx/memory.py`'s free-text `MemoryState`) looks like it "should" be the richer Tycho-style structured version, it shouldn't be — that's deliberately deferred, see `STRATEGY.md`.

## Global Constraints

- Python 3.12 required (`arc-agi` package requirement) — see [AGENTS.md](../../../AGENTS.md).
- No internet access at Kaggle evaluation time; no closed/API-based models (GPT/Claude/Gemini) anywhere in the submitted agent — Gemma-4-31B only. This includes Cerebras: it is a development-only proxy lane and must never appear in the Kaggle runtime or artifact, under any condition (see Task 9/10 and `AGENTS.md`'s "Cerebras development boundary").
- Never load Gemma-4-31B locally (RTX 4060, insufficient VRAM) — model loading only happens on Colab Pro or Kaggle's `rtx6000` accelerator, in later plans.
- `ACTION6` coordinates must validate to the inclusive range `[0, 63]`.
- `ACTION1`–`ACTION5` have no fixed/hardcoded meaning — semantics are game-specific.
- `ACTION7` must never be returned unless a game's frame metadata explicitly lists it as legal.
- Feature modules never read environment variables directly — only `zerx/config.py` reads env vars, at startup. The one deliberate exception is `CEREBRAS_API_KEY`, read only inside the Cerebras backend's own client constructor (Task 9) — a credential is not a config value and must never be serialized, hashed, or logged.
- `choose_action` (and everything it calls) must never raise an unhandled exception.
- Only 5 official Kaggle submissions/day exist — nothing in this plan pushes to Kaggle or spends that quota; that's a later plan requiring explicit approval per `AGENTS.md`'s Kaggle gate. This is also a 5-day delivery window (today is Day 1, due Day 5 — see `docs/TEAM_WORKFLOW.md`); this plan is Day 1's local-skeleton portion of that schedule.
- `heuristic_first` and `arbiter_on` both default to **off** until ablation comparisons demonstrate value.
- No `CEREBRAS_API_KEY` is required to run the default `pytest` suite — all Cerebras calls are mocked; live-network Cerebras tests are opt-in and separately marked.

---

## Task 1: Import the official starter kit and record baseline-000

**Files:**
- Create: `agent/my_agent.py`, `scripts/`, `notebooks/`, `Makefile` (copied from upstream)
- Modify: `.gitignore` (merge upstream's starter `.gitignore`)
- Create: `docs/superpowers/experiments/baseline-000.md`

**Interfaces:**
- Produces: the real `Agent`, `GameAction`, and frame-data type import paths (recorded in `baseline-000.md`) that Task 14's adapter depends on.

- [ ] **Step 1: Clone the upstream starter into a scratch directory**

```bash
git clone https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter.git .starter-import
```
Expected: clones successfully; `.starter-import/README.md` exists.

- [ ] **Step 2: Record the exact upstream commit**

```bash
git -C .starter-import rev-parse HEAD
```
Expected: prints a 40-character SHA. Copy it — it goes into `baseline-000.md` in Step 7.

- [ ] **Step 3: Copy the starter's structural files into this repo's root**

```bash
cp -r .starter-import/agent .starter-import/scripts .starter-import/notebooks .
cp .starter-import/Makefile .
cp .starter-import/.gitignore .gitignore.starter
```
Expected: `agent/my_agent.py`, `scripts/play_local.py`, `scripts/build_notebook.py`, `notebooks/kernel-metadata.json`, `Makefile` now exist at repo root.

- [ ] **Step 4: Merge `.gitignore`**

Open `.gitignore.starter` and `.gitignore` (create the latter if it doesn't exist yet) and combine their contents into one `.gitignore` with no duplicate lines, at minimum keeping `.venv/`, `vendor/`, `.kaggle/`, `environment_files/`, and any generated-notebook paths from the starter's ignore file. Delete `.gitignore.starter` once merged.

- [ ] **Step 5: Inspect the real agent/frame API**

Open `agent/my_agent.py` (the file just copied in Step 3) and read it fully. It contains a working random-action baseline agent, which means its imports and method bodies are the authoritative, real example of the upstream API — not something to guess at. Note down, verbatim:
- The exact import line(s) for `Agent`, `GameAction`, and whatever type represents a single frame (e.g. `FrameData`, `Frame`).
- The exact attribute/method names used to read the grid, the list of currently-legal actions, and the game-over/terminal state off a frame object.
- The exact method used to attach `{"x", "y"}` data to `GameAction.ACTION6` (the README shows `env.step(GameAction.ACTION6, data={...})` for the standalone agent framework, but the Kaggle starter's `MyAgent.choose_action` may return the action differently — check what the random baseline actually returns).

These go into `baseline-000.md` in Step 7. Task 14 depends on this being accurate, not assumed.

- [ ] **Step 6: Run the unchanged starter baseline**

```bash
make setup
make play-local
```
Expected: `make setup` finishes without error (Python 3.12 venv, `arc-agi` package, `kaggle` CLI installed). `make play-local` runs the random-action agent against every local game and prints a per-game score summary. Per the starter's own README, an unmodified random agent is expected to score `0.0` — that's the correct, expected baseline-000 result, not a bug.

- [ ] **Step 7: Write `docs/superpowers/experiments/baseline-000.md`**

Create the file with this structure, filling in the real values from Steps 2, 5, and 6 (no placeholders left in the committed file):

```markdown
# baseline-000 — unmodified starter

- Date: <today's date>
- Upstream repo: https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter
- Upstream commit: <SHA from Step 2>
- Python version: <output of `python --version`>
- Agent/GameAction/Frame import paths: <exact lines from Step 5>
- Frame attribute names (grid / legal actions / terminal state): <from Step 5>
- `make play-local` result: <pasted output summary>
- Conclusion: baseline recorded, matches README's documented 0.0 score for
  the random agent. No zerx code involved yet.
```

- [ ] **Step 8: Clean up the scratch clone and commit**

```bash
rm -rf .starter-import
git add agent scripts notebooks Makefile .gitignore docs/superpowers/experiments/baseline-000.md
git commit -m "Import official ARC-AGI-3 starter kit, record baseline-000"
```
Expected: commit succeeds; `git status` is clean.

---

## Task 2: `zerx/types.py` — shared internal types

**Files:**
- Create: `zerx/__init__.py` (empty)
- Create: `zerx/types.py`
- Create: `tests/__init__.py` (empty)
- Test: `tests/test_types.py`
- Create: `requirements-zerx.txt`

**Interfaces:**
- Produces: `ActionName` (Enum), `Action` (validated dataclass), `GameFrame` (dataclass) — used by every other `zerx` module and by Task 14's adapter.

- [ ] **Step 1: Add test tooling to the venv**

Create `requirements-zerx.txt`:
```
numpy
pytest
```
Run:
```bash
.venv/bin/pip install -r requirements-zerx.txt
```
(On Windows: `.venv\Scripts\pip install -r requirements-zerx.txt`)
Expected: installs without error.

- [ ] **Step 2: Write the failing tests**

`tests/test_types.py`:
```python
import pytest

from zerx.types import Action, ActionName, GameFrame


def test_action6_requires_coordinates():
    with pytest.raises(ValueError):
        Action(name=ActionName.ACTION6)


def test_action6_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError):
        Action(name=ActionName.ACTION6, x=64, y=0)
    with pytest.raises(ValueError):
        Action(name=ActionName.ACTION6, x=0, y=-1)


def test_action6_accepts_boundary_coordinates():
    Action(name=ActionName.ACTION6, x=0, y=0)
    Action(name=ActionName.ACTION6, x=63, y=63)


def test_non_click_action_rejects_coordinates():
    with pytest.raises(ValueError):
        Action(name=ActionName.ACTION1, x=1, y=1)


def test_simple_action_constructs():
    action = Action(name=ActionName.RESET)
    assert action.name == ActionName.RESET
    assert action.x is None and action.y is None


def test_gameframe_is_frozen():
    frame = GameFrame(
        grid=((0, 0), (0, 0)),
        legal_actions=frozenset({ActionName.ACTION1}),
        is_game_over=False,
    )
    with pytest.raises(Exception):
        frame.is_game_over = True
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_types.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.types'` (or similar import error).

- [ ] **Step 4: Implement `zerx/types.py`**

```python
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
```

Create empty `zerx/__init__.py` and `tests/__init__.py`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_types.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add zerx/__init__.py zerx/types.py tests/__init__.py tests/test_types.py requirements-zerx.txt
git commit -m "feat(zerx): add shared internal Action/ActionName/GameFrame types"
```

---

## Task 3: `zerx/config.py` — typed, serializable configuration

**Files:**
- Create: `zerx/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Config` (frozen dataclass) with fields `experiment_id`, `heuristic_first`, `heuristic_confidence_threshold`, `memory_on`, `memory_refresh_interval`, `arbiter_on`, `budget_soft_cap`, `model_revision`, `backend` (`"fake" | "cerebras_dev" | "gemma_local" | "gemma_kaggle"`), `platform` (`"local" | "colab" | "kaggle"`); `Config.from_env(env=None)` (raises `ValueError` for the illegal `backend="cerebras_dev"` + `platform="kaggle"` combination), `Config.to_json()`, `Config.config_hash()`. Every later module receives a `Config` instance by parameter, never reads env vars itself.

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:
```python
import json

import pytest

from zerx.config import Config


def test_default_config_values():
    cfg = Config()
    assert cfg.experiment_id == "dev"
    assert cfg.heuristic_first is False
    assert cfg.arbiter_on is False
    assert cfg.memory_on is True
    assert cfg.budget_soft_cap == 50


def test_from_env_missing_uses_defaults():
    assert Config.from_env({}) == Config()


def test_from_env_overrides_selected_fields():
    cfg = Config.from_env({
        "ZERX_HEURISTIC_FIRST": "true",
        "ZERX_BUDGET_SOFT_CAP": "25",
        "ZERX_EXPERIMENT_ID": "exp-1",
    })
    assert cfg.heuristic_first is True
    assert cfg.budget_soft_cap == 25
    assert cfg.experiment_id == "exp-1"
    assert cfg.memory_on is True  # untouched, stays default


def test_config_hash_is_deterministic():
    assert Config().config_hash() == Config().config_hash()


def test_config_hash_changes_with_field_value():
    assert Config().config_hash() != Config(heuristic_first=True).config_hash()


def test_to_json_round_trips_all_fields():
    cfg = Config(experiment_id="exp-2", budget_soft_cap=10)
    payload = json.loads(cfg.to_json())
    assert payload["experiment_id"] == "exp-2"
    assert payload["budget_soft_cap"] == 10


def test_default_backend_and_platform_are_safe():
    cfg = Config()
    assert cfg.backend == "fake"
    assert cfg.platform == "local"


def test_from_env_rejects_cerebras_dev_on_kaggle_platform():
    with pytest.raises(ValueError):
        Config.from_env({"ZERX_BACKEND": "cerebras_dev", "ZERX_PLATFORM": "kaggle"})


def test_from_env_allows_cerebras_dev_on_local_platform():
    cfg = Config.from_env({"ZERX_BACKEND": "cerebras_dev", "ZERX_PLATFORM": "local"})
    assert cfg.backend == "cerebras_dev"


def test_from_env_allows_gemma_kaggle_on_kaggle_platform():
    cfg = Config.from_env({"ZERX_BACKEND": "gemma_kaggle", "ZERX_PLATFORM": "kaggle"})
    assert cfg.platform == "kaggle"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.config'`.

- [ ] **Step 3: Implement `zerx/config.py`**

```python
"""Typed, serializable configuration. Only this module reads environment
variables — feature modules receive a resolved Config via dependency
injection and must never read os.environ themselves (see AGENTS.md).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Mapping, Optional


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    return default if raw is None else int(raw)


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    return default if raw is None else float(raw)


def _env_str(env: Mapping[str, str], key: str, default: str) -> str:
    return env.get(key, default)


@dataclass(frozen=True)
class Config:
    experiment_id: str = "dev"
    heuristic_first: bool = False
    heuristic_confidence_threshold: float = 0.8
    memory_on: bool = True
    memory_refresh_interval: int = 10
    arbiter_on: bool = False
    budget_soft_cap: int = 50
    model_revision: str = "gemma-4-31b-it"
    backend: str = "fake"  # "fake" | "cerebras_dev" | "gemma_local" | "gemma_kaggle"
    platform: str = "local"  # "local" | "colab" | "kaggle"

    def __post_init__(self) -> None:
        if self.backend == "cerebras_dev" and self.platform == "kaggle":
            raise ValueError(
                "cerebras_dev is a development-only backend and must never be "
                "selected on platform=kaggle"
            )

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "Config":
        env = os.environ if env is None else env
        return cls(
            experiment_id=_env_str(env, "ZERX_EXPERIMENT_ID", cls.experiment_id),
            heuristic_first=_env_bool(env, "ZERX_HEURISTIC_FIRST", cls.heuristic_first),
            heuristic_confidence_threshold=_env_float(
                env,
                "ZERX_HEURISTIC_CONFIDENCE_THRESHOLD",
                cls.heuristic_confidence_threshold,
            ),
            memory_on=_env_bool(env, "ZERX_MEMORY_ON", cls.memory_on),
            memory_refresh_interval=_env_int(
                env, "ZERX_MEMORY_REFRESH_INTERVAL", cls.memory_refresh_interval
            ),
            arbiter_on=_env_bool(env, "ZERX_ARBITER_ON", cls.arbiter_on),
            budget_soft_cap=_env_int(env, "ZERX_BUDGET_SOFT_CAP", cls.budget_soft_cap),
            model_revision=_env_str(env, "ZERX_MODEL_REVISION", cls.model_revision),
            backend=_env_str(env, "ZERX_BACKEND", cls.backend),
            platform=_env_str(env, "ZERX_PLATFORM", cls.platform),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:12]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_config.py -v
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add zerx/config.py tests/test_config.py
git commit -m "feat(zerx): add typed Config with env resolution and hashing"
```

---

## Task 4: `zerx/perception.py` — frame to ASCII grid + labeled objects

**Files:**
- Create: `zerx/perception.py`
- Test: `tests/test_perception.py`

**Interfaces:**
- Consumes: `GameFrame` from `zerx/types.py`.
- Produces: `LabeledObject` (with `.label: str`, `.color: int`, `.cells: Tuple[Tuple[int,int],...]`, `.bbox` property, `.size` property), `PerceptionResult` (`.ascii_grid: str`, `.objects: Tuple[LabeledObject,...]`), `perceive(frame, history=()) -> PerceptionResult`.

- [ ] **Step 1: Write the failing tests**

`tests/test_perception.py`:
```python
from zerx.perception import perceive
from zerx.types import ActionName, GameFrame


def _frame(grid):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=frozenset({ActionName.ACTION1}),
        is_game_over=False,
    )


def test_perceive_empty_grid_has_no_objects():
    result = perceive(_frame([[0, 0], [0, 0]]))
    assert result.objects == ()


def test_perceive_single_cell_object():
    result = perceive(_frame([[0, 0], [0, 5]]))
    assert len(result.objects) == 1
    obj = result.objects[0]
    assert obj.color == 5
    assert obj.size == 1
    assert obj.bbox == (1, 1, 1, 1)


def test_perceive_groups_contiguous_same_color():
    grid = [
        [0, 3, 3],
        [0, 3, 0],
        [0, 0, 0],
    ]
    result = perceive(_frame(grid))
    assert len(result.objects) == 1
    assert result.objects[0].size == 3


def test_perceive_separates_touching_different_colors():
    grid = [[2, 4]]
    result = perceive(_frame(grid))
    colors = sorted(obj.color for obj in result.objects)
    assert colors == [2, 4]


def test_perceive_separates_diagonal_same_color_as_two_objects():
    # 4-connectivity only: diagonal touches don't merge.
    grid = [
        [3, 0],
        [0, 3],
    ]
    result = perceive(_frame(grid))
    assert len(result.objects) == 2


def test_ascii_grid_matches_dimensions_and_hex_encodes_colors():
    grid = [[0, 10], [1, 2]]
    result = perceive(_frame(grid))
    rows = result.ascii_grid.split("\n")
    assert len(rows) == 2
    assert all(len(row) == 2 for row in rows)
    assert rows[0] == "0a"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_perception.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.perception'`.

- [ ] **Step 3: Implement `zerx/perception.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_perception.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add zerx/perception.py tests/test_perception.py
git commit -m "feat(zerx): add perception (ascii grid + connected-component labeling)"
```

---

## Task 5: `zerx/heuristics.py` — click candidates and graded negative affordances

Per `STRATEGY.md`'s "soft negative affordances": an ineffective click
down-ranks an object signature's future score instead of permanently
banning it, and a later effective use of the same signature recovers some
of that penalty. This is a small, self-contained change to a module that
hasn't shipped yet, so it ships graded from the start rather than as a
hard-exclusion set that gets rewritten later.

**Files:**
- Create: `zerx/heuristics.py`
- Test: `tests/test_heuristics.py`

**Interfaces:**
- Consumes: `LabeledObject`, `PerceptionResult` from `zerx/perception.py`.
- Produces: `DeadSignatureTracker` (`.record_outcome(obj, effective: bool)`, `.penalty(obj) -> float` in `[0.0, 1.0]`, `.reset()`), `ClickCandidate` (`.x`, `.y`, `.object_label`, `.score`), `rank_click_candidates(perception, affordance, grid_size=64) -> List[ClickCandidate]` (sorted highest score first — penalized objects are down-ranked, never removed). Who calls `record_outcome` and when: not this module — that requires observing the *next* frame, which is the harness adapter's job (Task 14), using `zerx/transitions.py` (Task 13) to know whether a click was effective.

- [ ] **Step 1: Write the failing tests**

`tests/test_heuristics.py`:
```python
from zerx.heuristics import DeadSignatureTracker, rank_click_candidates
from zerx.perception import LabeledObject, PerceptionResult


def _obj(label, color, cells):
    return LabeledObject(label=label, color=color, cells=tuple(cells))


def test_new_signature_has_zero_penalty():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    assert tracker.penalty(obj) == 0.0


def test_ineffective_outcome_increases_penalty():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    tracker.record_outcome(obj, effective=False)
    assert tracker.penalty(obj) > 0.0


def test_effective_outcome_recovers_penalty():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    tracker.record_outcome(obj, effective=False)
    penalty_after_fail = tracker.penalty(obj)
    tracker.record_outcome(obj, effective=True)
    assert tracker.penalty(obj) < penalty_after_fail


def test_penalty_stays_within_zero_one_range():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    for _ in range(10):
        tracker.record_outcome(obj, effective=False)
    assert tracker.penalty(obj) == 1.0
    for _ in range(10):
        tracker.record_outcome(obj, effective=True)
    assert tracker.penalty(obj) == 0.0


def test_reset_clears_penalties():
    tracker = DeadSignatureTracker()
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    tracker.record_outcome(obj, effective=False)
    tracker.reset()
    assert tracker.penalty(obj) == 0.0


def test_rank_click_candidates_empty_perception_returns_empty():
    result = PerceptionResult(ascii_grid="0", objects=())
    assert rank_click_candidates(result, DeadSignatureTracker()) == []


def test_rank_click_candidates_down_ranks_but_keeps_fully_penalized_object():
    obj = _obj("obj0", color=5, cells=[(1, 1)])
    tracker = DeadSignatureTracker()
    for _ in range(10):
        tracker.record_outcome(obj, effective=False)
    result = PerceptionResult(ascii_grid="", objects=(obj,))
    candidates = rank_click_candidates(result, tracker)
    assert len(candidates) == 1  # still present, never hard-excluded
    assert candidates[0].score == 0.0


def test_rank_click_candidates_prefers_smaller_object():
    small = _obj("small", color=1, cells=[(0, 0)])
    big = _obj("big", color=2, cells=[(2, 0), (2, 1), (2, 2), (2, 3)])
    result = PerceptionResult(ascii_grid="", objects=(big, small))
    candidates = rank_click_candidates(result, DeadSignatureTracker())
    assert candidates[0].object_label == "small"


def test_click_candidate_coordinates_within_bounds():
    obj = _obj("obj0", color=1, cells=[(0, 0)])
    result = PerceptionResult(ascii_grid="", objects=(obj,))
    candidates = rank_click_candidates(result, DeadSignatureTracker(), grid_size=64)
    assert 0 <= candidates[0].x <= 63
    assert 0 <= candidates[0].y <= 63
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_heuristics.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.heuristics'`.

- [ ] **Step 3: Implement `zerx/heuristics.py`**

```python
"""No-GPU fallback/candidate layer: numpy-based click-candidate scoring and
a graded, decaying negative-affordance tracker (STRATEGY.md's "soft
negative affordances" — down-rank an ineffective object signature instead
of permanently banning it).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from zerx.perception import LabeledObject, PerceptionResult


def _signature(obj: LabeledObject) -> Tuple[int, int]:
    """A coarse (color, size-bucket) signature that identifies "the same
    kind of thing" across frames, not the exact same object instance.
    """
    size_bucket = min(obj.size, 20) // 4
    return (obj.color, size_bucket)


class DeadSignatureTracker:
    """Per-signature penalty in [0.0, 1.0]. An ineffective click raises it
    by `penalty_step` (clamped at 1.0); an effective use of the same
    signature lowers it by `recovery_step` (clamped at 0.0). A signature
    that has never been observed has penalty 0.0 — it starts trusted.
    """

    def __init__(self, penalty_step: float = 0.35, recovery_step: float = 0.5) -> None:
        self._penalty: Dict[Tuple[int, int], float] = {}
        self._penalty_step = penalty_step
        self._recovery_step = recovery_step

    def record_outcome(self, obj: LabeledObject, effective: bool) -> None:
        sig = _signature(obj)
        current = self._penalty.get(sig, 0.0)
        if effective:
            current = max(0.0, current - self._recovery_step)
        else:
            current = min(1.0, current + self._penalty_step)
        if current == 0.0:
            self._penalty.pop(sig, None)
        else:
            self._penalty[sig] = current

    def penalty(self, obj: LabeledObject) -> float:
        return self._penalty.get(_signature(obj), 0.0)

    def reset(self) -> None:
        self._penalty.clear()


@dataclass(frozen=True)
class ClickCandidate:
    x: int
    y: int
    object_label: str
    score: float


def _object_center(obj: LabeledObject) -> Tuple[int, int]:
    min_x, min_y, max_x, max_y = obj.bbox
    return ((min_x + max_x) // 2, (min_y + max_y) // 2)


def rank_click_candidates(
    perception: PerceptionResult,
    affordance: DeadSignatureTracker,
    grid_size: int = 64,
) -> List[ClickCandidate]:
    """Score objects by "small, rare-colored, button-like" heuristics, then
    scale each score by `(1 - penalty)`. Smaller objects and less-common
    colors score higher; penalized signatures rank lower but are always
    still returned — ranking, not exclusion, is how "soft" affordances
    stay soft.
    """
    objects = perception.objects
    if not objects:
        return []

    sizes = np.array([o.size for o in objects], dtype=np.float64)
    colors = [o.color for o in objects]
    color_counts = {c: colors.count(c) for c in set(colors)}
    rarity = np.array([1.0 / color_counts[c] for c in colors], dtype=np.float64)

    max_size = sizes.max()
    size_score = 1.0 - (sizes / max_size) if max_size > 0 else np.zeros_like(sizes)
    max_rarity = rarity.max()
    rarity_score = rarity / max_rarity if max_rarity > 0 else np.zeros_like(rarity)

    combined = 0.5 * size_score + 0.5 * rarity_score

    candidates = []
    for obj, base_score in zip(objects, combined):
        score = float(base_score) * (1.0 - affordance.penalty(obj))
        cx, cy = _object_center(obj)
        cx = min(max(cx, 0), grid_size - 1)
        cy = min(max(cy, 0), grid_size - 1)
        candidates.append(ClickCandidate(x=cx, y=cy, object_label=obj.label, score=score))

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_heuristics.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add zerx/heuristics.py tests/test_heuristics.py
git commit -m "feat(zerx): add click-candidate heuristic with graded negative affordances"
```

---

## Task 6: `zerx/memory.py` — reflection memory refresh

**Files:**
- Create: `zerx/memory.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: nothing outside stdlib.
- Produces: `MemoryState` (`.summary: str`, `.step_count: int`, `.last_refreshed_step: int`, `.reset()`), `Summarizer` type alias, `maybe_refresh(state, recent_context, summarizer, refresh_interval) -> MemoryState` (returns a new `MemoryState`, does not mutate input).

- [ ] **Step 1: Write the failing tests**

`tests/test_memory.py`:
```python
from zerx.memory import MemoryState, maybe_refresh


def test_memory_state_reset_clears_all_fields():
    state = MemoryState(summary="learned stuff", step_count=5, last_refreshed_step=3)
    state.reset()
    assert state.summary == ""
    assert state.step_count == 0
    assert state.last_refreshed_step == 0


def test_maybe_refresh_not_due_keeps_summary_and_skips_summarizer():
    state = MemoryState(summary="old", step_count=0, last_refreshed_step=0)

    def boom(prev, ctx):
        raise AssertionError("summarizer should not be called")

    new_state = maybe_refresh(state, "context", boom, refresh_interval=10)
    assert new_state.summary == "old"
    assert new_state.step_count == 1
    assert new_state.last_refreshed_step == 0


def test_maybe_refresh_due_calls_summarizer_and_updates():
    state = MemoryState(summary="old", step_count=8, last_refreshed_step=0)
    new_state = maybe_refresh(
        state, "context", lambda prev, ctx: f"{prev}+{ctx}", refresh_interval=9
    )
    assert new_state.step_count == 9
    assert new_state.last_refreshed_step == 9
    assert new_state.summary == "old+context"


def test_maybe_refresh_does_not_mutate_input():
    state = MemoryState(summary="old", step_count=0, last_refreshed_step=0)
    maybe_refresh(state, "context", lambda prev, ctx: "new", refresh_interval=1)
    assert state.summary == "old"
    assert state.step_count == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_memory.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.memory'`.

- [ ] **Step 3: Implement `zerx/memory.py`**

```python
"""Reflection memory: a periodically-refreshed free-text summary of what the
agent has learned this game. Refresh cadence is config-driven; the actual
summarization is injected as a callable so this module has zero model
coupling. Any latency the injected summarizer costs is the caller's
responsibility to measure — this module never touches the action budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

Summarizer = Callable[[str, str], str]  # (previous_summary, recent_context) -> new_summary


@dataclass
class MemoryState:
    summary: str = ""
    step_count: int = 0
    last_refreshed_step: int = 0

    def reset(self) -> None:
        """Clear memory between games — reflection from one game must never
        leak into the next.
        """
        self.summary = ""
        self.step_count = 0
        self.last_refreshed_step = 0


def maybe_refresh(
    state: MemoryState,
    recent_context: str,
    summarizer: Summarizer,
    refresh_interval: int,
) -> MemoryState:
    """Advance the step count by one and, if `refresh_interval` has
    elapsed since the last refresh, produce a new MemoryState with an
    updated summary. Returns a new MemoryState; never mutates `state`.
    """
    new_step_count = state.step_count + 1
    due = (new_step_count - state.last_refreshed_step) >= refresh_interval
    if not due:
        return MemoryState(
            summary=state.summary,
            step_count=new_step_count,
            last_refreshed_step=state.last_refreshed_step,
        )
    new_summary = summarizer(state.summary, recent_context)
    return MemoryState(
        summary=new_summary,
        step_count=new_step_count,
        last_refreshed_step=new_step_count,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_memory.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add zerx/memory.py tests/test_memory.py
git commit -m "feat(zerx): add reflection memory refresh logic"
```

---

## Task 7: `zerx/budget.py` — observable action-efficiency signal

**Files:**
- Create: `zerx/budget.py`
- Test: `tests/test_budget.py`

**Interfaces:**
- Consumes: nothing outside stdlib.
- Produces: `BudgetSignal` (`.actions_taken`, `.soft_cap`, `.should_favor_execution`), `evaluate_budget(actions_taken, soft_cap, favor_threshold=0.8) -> BudgetSignal`.

- [ ] **Step 1: Write the failing tests**

`tests/test_budget.py`:
```python
import pytest

from zerx.budget import evaluate_budget


def test_evaluate_budget_below_threshold():
    signal = evaluate_budget(actions_taken=5, soft_cap=50)
    assert signal.should_favor_execution is False


def test_evaluate_budget_at_threshold_favors_execution():
    signal = evaluate_budget(actions_taken=40, soft_cap=50, favor_threshold=0.8)
    assert signal.should_favor_execution is True


def test_evaluate_budget_above_soft_cap_still_favors_execution():
    signal = evaluate_budget(actions_taken=100, soft_cap=50)
    assert signal.should_favor_execution is True
    assert signal.actions_taken == 100
    assert signal.soft_cap == 50


def test_evaluate_budget_rejects_non_positive_soft_cap():
    with pytest.raises(ValueError):
        evaluate_budget(actions_taken=1, soft_cap=0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_budget.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.budget'`.

- [ ] **Step 3: Implement `zerx/budget.py`**

```python
"""Observable action-efficiency signal. RHAE's human-median denominator is
hidden evaluation data the agent cannot see — this module only ever looks
at the agent's own observable action count against a configurable soft cap,
and produces a *strategy signal*, never a forced or invented action.
Legality is always enforced by policy.py's validation, not here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSignal:
    actions_taken: int
    soft_cap: int
    should_favor_execution: bool


def evaluate_budget(
    actions_taken: int, soft_cap: int, favor_threshold: float = 0.8
) -> BudgetSignal:
    """`should_favor_execution` flips once `actions_taken` crosses
    `favor_threshold` of `soft_cap` — a hint to prefer a more confident
    candidate over an exploratory one. It never selects or invents an
    action itself.
    """
    if soft_cap <= 0:
        raise ValueError("soft_cap must be positive")
    ratio = actions_taken / soft_cap
    return BudgetSignal(
        actions_taken=actions_taken,
        soft_cap=soft_cap,
        should_favor_execution=ratio >= favor_threshold,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_budget.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add zerx/budget.py tests/test_budget.py
git commit -m "feat(zerx): add observable action-budget signal"
```

---

## Task 8: `zerx/model_backend.py` — backend protocol and fake test double

**Files:**
- Create: `zerx/model_backend.py`
- Test: `tests/test_model_backend.py`

**Interfaces:**
- Produces: `ModelBackend` (Protocol with `.generate(prompt: str) -> str`), `FakeModelBackend` (`.responses: List[str]`, `.generate()`, `.call_count`, `.last_prompt`), `GemmaModelBackend` (`.model_revision`, `.generate()` raises `NotImplementedError` until a later Colab/Kaggle plan implements it).

- [ ] **Step 1: Write the failing tests**

`tests/test_model_backend.py`:
```python
import pytest

from zerx.model_backend import FakeModelBackend, GemmaModelBackend


def test_fake_backend_returns_scripted_responses_in_order():
    backend = FakeModelBackend(responses=["first", "second"])
    assert backend.generate("prompt-a") == "first"
    assert backend.generate("prompt-b") == "second"


def test_fake_backend_raises_when_exhausted():
    backend = FakeModelBackend(responses=[])
    with pytest.raises(RuntimeError):
        backend.generate("prompt")


def test_fake_backend_tracks_call_count_and_last_prompt():
    backend = FakeModelBackend(responses=["a", "b"])
    backend.generate("first-prompt")
    backend.generate("second-prompt")
    assert backend.call_count == 2
    assert backend.last_prompt == "second-prompt"


def test_gemma_backend_constructs_without_loading_model():
    backend = GemmaModelBackend(model_revision="gemma-4-31b-it")
    assert backend.model_revision == "gemma-4-31b-it"


def test_gemma_backend_generate_not_yet_implemented():
    backend = GemmaModelBackend(model_revision="gemma-4-31b-it")
    with pytest.raises(NotImplementedError):
        backend.generate("prompt")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_model_backend.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.model_backend'`.

- [ ] **Step 3: Implement `zerx/model_backend.py`**

```python
"""The only module allowed to load/call the Gemma model. Defines a narrow
Protocol so every other module (and all local tests) can depend on
`ModelBackend` without ever importing a real model. `GemmaModelBackend`
loads the real thing and is exercised only on Colab/Kaggle — its
`generate()` is implemented in a later plan, never in local unit tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol


class ModelBackend(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class FakeModelBackend:
    """Test double: returns scripted responses in order."""

    responses: List[str] = field(default_factory=list)
    _calls: List[str] = field(default_factory=list, init=False)

    def generate(self, prompt: str) -> str:
        self._calls.append(prompt)
        if not self.responses:
            raise RuntimeError("FakeModelBackend: no scripted responses left")
        return self.responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def last_prompt(self) -> str:
        if not self._calls:
            raise RuntimeError("FakeModelBackend: generate() was never called")
        return self._calls[-1]


class GemmaModelBackend:
    """Real backend — loads Gemma-4-31B. Constructed but not exercised by
    local unit tests; Colab/Kaggle smoke tests cover this path per
    AGENTS.md's Colab and Kaggle gates.
    """

    def __init__(self, model_revision: str) -> None:
        self.model_revision = model_revision
        self._model = None  # loaded lazily by a later Colab/Kaggle-specific task

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "GemmaModelBackend.generate is implemented in the Colab/Kaggle "
            "model-loading plan, not the local model-free skeleton."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_model_backend.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add zerx/model_backend.py tests/test_model_backend.py
git commit -m "feat(zerx): add ModelBackend protocol, fake test double, Gemma stub"
```

---

## Task 9: `zerx/backends/cerebras_dev.py` — development-only Cerebras backend

**Files:**
- Create: `zerx/backends/__init__.py` (empty)
- Create: `zerx/backends/cerebras_dev.py`
- Test: `tests/test_cerebras_dev.py`

**Interfaces:**
- Consumes: `ModelBackend` protocol shape from Task 8 (duck-typed — `CerebrasDevBackend` satisfies it by implementing `.generate(prompt: str) -> str`, same as `FakeModelBackend`/`GemmaModelBackend`).
- Produces: `CerebrasDevBackend(model_id, api_version="v1", request_timeout_seconds=10.0, max_retries=2, api_key=None, http_post=None)` — `http_post` is an injected callable `(url, headers, json_body, timeout) -> dict` so tests never make a real network call; `api_key` defaults to reading `CEREBRAS_API_KEY` from `os.environ` only if not passed explicitly. `.generate(prompt) -> str`, `.credential_present: bool` property, `.last_latency_seconds: Optional[float]`.

This backend is network-based and needs no local GPU, so — unlike
`GemmaModelBackend` — it can be fully exercised (with a fake HTTP layer) in
this local, model-free plan. Per `AGENTS.md`, it is the one module allowed
to read `CEREBRAS_API_KEY` directly from the environment, and it must never
be selected when `Config.platform == "kaggle"` (enforced by `Config`'s
`__post_init__` from Task 3 — this task's tests confirm the backend itself
also refuses to construct in that case, as defense in depth).

- [ ] **Step 1: Write the failing tests**

`tests/test_cerebras_dev.py`:
```python
import pytest

from zerx.backends.cerebras_dev import CerebrasDevBackend


def _fake_http_post(response_json, captured=None):
    def _post(url, headers, json_body, timeout):
        if captured is not None:
            captured.append({"url": url, "headers": headers, "json_body": json_body, "timeout": timeout})
        return response_json
    return _post


def _ok_response(text='{"action": "ACTION1"}'):
    return {"choices": [{"message": {"content": text}}]}


def test_generate_returns_message_content():
    backend = CerebrasDevBackend(
        model_id="gemma-4-31b",
        api_key="sk-test-not-real",
        http_post=_fake_http_post(_ok_response()),
    )
    assert backend.generate("prompt text") == '{"action": "ACTION1"}'


def test_generate_records_latency_not_credentials():
    backend = CerebrasDevBackend(
        model_id="gemma-4-31b",
        api_key="sk-test-not-real",
        http_post=_fake_http_post(_ok_response()),
    )
    backend.generate("prompt text")
    assert backend.last_latency_seconds is not None
    assert backend.last_latency_seconds >= 0.0


def test_credential_present_true_when_key_given():
    backend = CerebrasDevBackend(model_id="gemma-4-31b", api_key="sk-test-not-real")
    assert backend.credential_present is True


def test_credential_present_false_when_no_key_anywhere(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    backend = CerebrasDevBackend(model_id="gemma-4-31b")
    assert backend.credential_present is False


def test_request_never_contains_raw_key_in_body():
    captured = []
    backend = CerebrasDevBackend(
        model_id="gemma-4-31b",
        api_key="sk-test-not-real",
        http_post=_fake_http_post(_ok_response(), captured=captured),
    )
    backend.generate("prompt text")
    assert "sk-test-not-real" not in str(captured[0]["json_body"])
    assert captured[0]["headers"]["Authorization"] == "Bearer sk-test-not-real"


def test_retries_on_transient_failure_then_succeeds():
    calls = {"count": 0}

    def flaky_post(url, headers, json_body, timeout):
        calls["count"] += 1
        if calls["count"] < 2:
            raise TimeoutError("simulated transient failure")
        return _ok_response()

    backend = CerebrasDevBackend(
        model_id="gemma-4-31b", api_key="sk-test-not-real", http_post=flaky_post, max_retries=2
    )
    assert backend.generate("prompt text") == '{"action": "ACTION1"}'
    assert calls["count"] == 2


def test_raises_after_exhausting_retries():
    def always_fails(url, headers, json_body, timeout):
        raise TimeoutError("simulated permanent failure")

    backend = CerebrasDevBackend(
        model_id="gemma-4-31b", api_key="sk-test-not-real", http_post=always_fails, max_retries=1
    )
    with pytest.raises(TimeoutError):
        backend.generate("prompt text")


def test_never_constructs_when_platform_kaggle():
    with pytest.raises(ValueError):
        CerebrasDevBackend(model_id="gemma-4-31b", api_key="sk-test-not-real", platform="kaggle")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_cerebras_dev.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.backends'`.

- [ ] **Step 3: Implement `zerx/backends/cerebras_dev.py`**

Create empty `zerx/backends/__init__.py`, then:

```python
"""Development-only Cerebras Inference Cloud backend. Never selected when
Config.platform == "kaggle" (enforced both here and in zerx/config.py, as
defense in depth). Reads CEREBRAS_API_KEY directly from the environment —
the one deliberate exception to "only config.py reads env vars", because a
credential is not a config value and must never be serialized, hashed, or
logged (see AGENTS.md's "Cerebras development boundary").

As of August 2026, Cerebras serves `gemma-4-31b` in preview with both text
and image input support — verify this still holds (model catalog and
capabilities can change) before assuming either mode works.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

HttpPost = Callable[[str, dict, dict, float], dict]

_CEREBRAS_CHAT_URL = "https://api.cerebras.ai/v1/chat/completions"


def _default_http_post(url: str, headers: dict, json_body: dict, timeout: float) -> dict:
    import json
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(json_body).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class CerebrasDevBackend:
    def __init__(
        self,
        model_id: str,
        api_version: str = "v1",
        request_timeout_seconds: float = 10.0,
        max_retries: int = 2,
        api_key: Optional[str] = None,
        http_post: Optional[HttpPost] = None,
        platform: str = "local",
    ) -> None:
        if platform == "kaggle":
            raise ValueError("cerebras_dev must never be constructed when platform=kaggle")
        self.model_id = model_id
        self.api_version = api_version
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self._api_key = api_key if api_key is not None else os.environ.get("CEREBRAS_API_KEY")
        self._http_post = http_post if http_post is not None else _default_http_post
        self.last_latency_seconds: Optional[float] = None

    @property
    def credential_present(self) -> bool:
        return self._api_key is not None

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        json_body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            start = time.monotonic()
            try:
                response = self._http_post(
                    _CEREBRAS_CHAT_URL, headers, json_body, self.request_timeout_seconds
                )
                self.last_latency_seconds = time.monotonic() - start
                return response["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised if exhausted
                last_error = exc
        assert last_error is not None
        raise last_error
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_cerebras_dev.py -v
```
Expected: 8 passed. No network access and no real `CEREBRAS_API_KEY` were needed — every test injects `http_post` or a literal test string.

- [ ] **Step 5: Commit**

```bash
git add zerx/backends/__init__.py zerx/backends/cerebras_dev.py tests/test_cerebras_dev.py
git commit -m "feat(zerx): add development-only Cerebras backend with hard kaggle lockout"
```

---

## Task 10: `zerx/secret_scan.py` — artifact secret scanning and backend contract test

**Files:**
- Create: `zerx/secret_scan.py`
- Test: `tests/test_secret_scan.py`
- Test: `tests/test_backend_contract.py`

**Interfaces:**
- Produces: `scan_for_secrets(text, extra_patterns=()) -> List[str]` — returns a list of human-readable findings (empty if clean). Flags `CEREBRAS_API_KEY`-looking assignments, the literal substring `api.cerebras.ai`, and any exact secret value passed in `extra_patterns` (so a real run can pass the actual configured key and confirm it isn't present, without hardcoding key formats here).

- [ ] **Step 1: Write the failing tests**

`tests/test_secret_scan.py`:
```python
from zerx.secret_scan import scan_for_secrets


def test_clean_text_has_no_findings():
    assert scan_for_secrets("this notebook loads gemma from /kaggle/input") == []


def test_flags_cerebras_endpoint_reference():
    findings = scan_for_secrets("client = Client(base_url='https://api.cerebras.ai/v1')")
    assert any("api.cerebras.ai" in f for f in findings)


def test_flags_cerebras_api_key_env_var_name():
    findings = scan_for_secrets('CEREBRAS_API_KEY = "sk-something"')
    assert any("CEREBRAS_API_KEY" in f for f in findings)


def test_flags_extra_secret_value_if_present():
    findings = scan_for_secrets("some text sk-my-actual-key-123 more text", extra_patterns=["sk-my-actual-key-123"])
    assert len(findings) == 1


def test_does_not_flag_extra_secret_value_if_absent():
    findings = scan_for_secrets("clean text here", extra_patterns=["sk-my-actual-key-123"])
    assert findings == []
```

`tests/test_backend_contract.py`:
```python
from zerx.backends.cerebras_dev import CerebrasDevBackend
from zerx.model_backend import FakeModelBackend, GemmaModelBackend


def test_all_backends_expose_generate_method():
    fake = FakeModelBackend(responses=["x"])
    cerebras = CerebrasDevBackend(model_id="gemma-4-31b", api_key="sk-test", http_post=lambda *a, **k: {"choices": [{"message": {"content": "x"}}]})
    gemma = GemmaModelBackend(model_revision="gemma-4-31b-it")

    for backend in (fake, cerebras, gemma):
        assert callable(getattr(backend, "generate", None))


def test_fake_and_cerebras_return_str_from_generate():
    fake = FakeModelBackend(responses=["hello"])
    cerebras = CerebrasDevBackend(model_id="gemma-4-31b", api_key="sk-test", http_post=lambda *a, **k: {"choices": [{"message": {"content": "hello"}}]})
    assert isinstance(fake.generate("p"), str)
    assert isinstance(cerebras.generate("p"), str)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_secret_scan.py tests/test_backend_contract.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.secret_scan'`.

- [ ] **Step 3: Implement `zerx/secret_scan.py`**

```python
"""Scans generated-artifact text (a notebook's source, a built package's
files) for leaked Cerebras credentials/endpoints before it's allowed to
ship anywhere near Kaggle. See AGENTS.md's hard safeguards.
"""
from __future__ import annotations

import re
from typing import Iterable, List

_STATIC_PATTERNS = (
    (re.compile(r"api\.cerebras\.ai"), "reference to api.cerebras.ai"),
    (re.compile(r"CEREBRAS_API_KEY"), "reference to CEREBRAS_API_KEY"),
)


def scan_for_secrets(text: str, extra_patterns: Iterable[str] = ()) -> List[str]:
    findings: List[str] = []
    for pattern, description in _STATIC_PATTERNS:
        if pattern.search(text):
            findings.append(description)
    for secret in extra_patterns:
        if secret and secret in text:
            findings.append("literal secret value found in artifact")
    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_secret_scan.py tests/test_backend_contract.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add zerx/secret_scan.py tests/test_secret_scan.py tests/test_backend_contract.py
git commit -m "feat(zerx): add artifact secret scanner and backend contract test"
```

---

## Task 11: `zerx/policy.py` — JSON parsing, bounded repair, legal validation

**Files:**
- Create: `zerx/policy.py`
- Test: `tests/test_policy_parse.py`

**Interfaces:**
- Consumes: `Action`, `ActionName` from `zerx/types.py`.
- Produces: `ParsedAction` (`.action: Action`, `.repaired: bool`), `parse_action(raw, legal_actions) -> Optional[ParsedAction]` (never raises).

- [ ] **Step 1: Write the failing tests**

`tests/test_policy_parse.py`:
```python
from zerx.policy import parse_action
from zerx.types import ActionName

LEGAL = frozenset({ActionName.ACTION1, ActionName.ACTION5, ActionName.ACTION6})


def test_parse_action_valid_json_first_try():
    result = parse_action('{"action": "ACTION1"}', LEGAL)
    assert result is not None
    assert result.action.name == ActionName.ACTION1
    assert result.repaired is False


def test_parse_action_repairs_markdown_fenced_json():
    raw = '```json\n{"action": "ACTION5"}\n```'
    result = parse_action(raw, LEGAL)
    assert result is not None
    assert result.action.name == ActionName.ACTION5
    assert result.repaired is True


def test_parse_action_rejects_action_not_in_legal_actions():
    result = parse_action('{"action": "ACTION2"}', LEGAL)
    assert result is None


def test_parse_action_rejects_illegal_action6_coordinates():
    raw = '{"action": "ACTION6", "data": {"x": 100, "y": 0}}'
    assert parse_action(raw, LEGAL) is None


def test_parse_action_accepts_valid_action6_coordinates():
    raw = '{"action": "ACTION6", "data": {"x": 10, "y": 20}}'
    result = parse_action(raw, LEGAL)
    assert result is not None
    assert result.action.x == 10 and result.action.y == 20


def test_parse_action_rejects_malformed_json_after_repair_attempt():
    assert parse_action("not json at all, no braces", LEGAL) is None


def test_parse_action_rejects_missing_action_key():
    assert parse_action('{"foo": "bar"}', LEGAL) is None


def test_parse_action_rejects_unknown_action_name():
    assert parse_action('{"action": "FLY"}', LEGAL) is None


def test_parse_action_action6_requires_data():
    assert parse_action('{"action": "ACTION6"}', LEGAL) is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_policy_parse.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.policy'`.

- [ ] **Step 3: Implement `zerx/policy.py`**

```python
"""JSON action parsing, bounded deterministic repair, and legal-action
validation. No model calls live here — see `decide()` (added in Task 12)
for the orchestrator that calls the model backend.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import FrozenSet, Optional

from zerx.types import Action, ActionName


@dataclass(frozen=True)
class ParsedAction:
    action: Action
    repaired: bool


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw: str) -> Optional[str]:
    """Deterministic repair: strip markdown code fences and pull out the
    first {...} substring. No model call, no retried reasoning.
    """
    stripped = raw.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped)
    stripped = re.sub(r"```$", "", stripped).strip()
    match = _JSON_OBJECT_RE.search(stripped)
    return match.group(0) if match else None


def parse_action(raw: str, legal_actions: FrozenSet[ActionName]) -> Optional[ParsedAction]:
    """Try a direct `json.loads` first; on failure, attempt exactly one
    deterministic extraction/repair and retry. Returns None (never raises)
    if both attempts fail or the result doesn't validate — callers fall
    back per the documented fallback chain.
    """
    for attempt, candidate_text in enumerate((raw, _extract_json_object(raw))):
        if candidate_text is None:
            continue
        try:
            payload = json.loads(candidate_text)
        except (json.JSONDecodeError, TypeError):
            continue
        action = _validate_payload(payload, legal_actions)
        if action is not None:
            return ParsedAction(action=action, repaired=(attempt == 1))
    return None


def _validate_payload(payload: object, legal_actions: FrozenSet[ActionName]) -> Optional[Action]:
    if not isinstance(payload, dict) or "action" not in payload:
        return None
    name_raw = payload["action"]
    if not isinstance(name_raw, str) or name_raw not in ActionName.__members__:
        return None
    name = ActionName[name_raw]
    if name not in legal_actions:
        return None
    data = payload.get("data") or {}
    try:
        if name == ActionName.ACTION6:
            return Action(name=name, x=int(data["x"]), y=int(data["y"]))
        return Action(name=name)
    except (KeyError, ValueError, TypeError):
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_policy_parse.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add zerx/policy.py tests/test_policy_parse.py
git commit -m "feat(zerx): add JSON action parsing with bounded deterministic repair"
```

---

## Task 12: `zerx/policy.py` — `decide()` orchestrator

**Files:**
- Modify: `zerx/policy.py` (append to the file created in Task 11)
- Test: `tests/test_policy_decide.py`

**Interfaces:**
- Consumes: `Config` (Task 3), `PerceptionResult`/`perceive` (Task 4), `DeadSignatureTracker`/`rank_click_candidates` (Task 5), `MemoryState`/`maybe_refresh` (Task 6), `BudgetSignal`/`evaluate_budget` (Task 7), `ModelBackend` (Task 8), `GameFrame`/`Action`/`ActionName` (Task 2), `parse_action` (Task 11, same file).
- Produces: `Decision` (`.action: Action`, `.source: str`, `.repaired: bool`, `.budget: Optional[BudgetSignal]`, `.target_object_label: Optional[str]` — set only when the action came from a click candidate, i.e. `source in {"heuristic", "fallback_heuristic"}`; this is what Task 14's adapter feeds back into `DeadSignatureTracker.record_outcome` once the next frame shows whether the click was effective), `build_prompt(perception, memory) -> str`, `decide(frame, history, memory, dead_signatures, config, backend, actions_taken) -> Tuple[Decision, MemoryState]` — never raises, implements the control flow from `AGENTS.md`.

- [ ] **Step 1: Write the failing tests**

`tests/test_policy_decide.py`:
```python
from zerx.budget import BudgetSignal
from zerx.config import Config
from zerx.heuristics import DeadSignatureTracker
from zerx.memory import MemoryState
from zerx.model_backend import FakeModelBackend
from zerx.policy import decide
from zerx.types import Action, ActionName, GameFrame

LEGAL = frozenset(
    {
        ActionName.RESET,
        ActionName.ACTION1,
        ActionName.ACTION5,
        ActionName.ACTION6,
    }
)


def _frame(grid, is_game_over=False, legal=LEGAL):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=legal,
        is_game_over=is_game_over,
    )


def _blank_frame(**kwargs):
    return _frame([[0, 0], [0, 0]], **kwargs)


def test_decide_returns_reset_when_game_over():
    decision, _ = decide(
        frame=_blank_frame(is_game_over=True),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[]),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.RESET
    assert decision.source == "reset"


def test_decide_uses_model_action_when_valid():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.ACTION1
    assert decision.source == "model"
    assert decision.repaired is False


def test_decide_repairs_markdown_fenced_model_output():
    raw = '```json\n{"action": "ACTION5"}\n```'
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[raw]),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.ACTION5
    assert decision.repaired is True


def test_decide_falls_back_to_heuristic_when_model_output_invalid():
    frame = _frame([[0, 0], [0, 5]])  # one clickable object
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=["garbage, not json"]),
        actions_taken=0,
    )
    assert decision.source == "fallback_heuristic"
    assert decision.action.name == ActionName.ACTION6


def test_decide_falls_back_to_deterministic_when_no_candidates_and_model_invalid():
    decision, _ = decide(
        frame=_blank_frame(),  # no objects -> no click candidates
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=["garbage, not json"]),
        actions_taken=0,
    )
    assert decision.source == "fallback_deterministic"
    assert decision.action.name in LEGAL


def test_decide_heuristic_first_skips_model_call_when_confident():
    frame = _frame([[0, 0], [0, 5]])
    backend = FakeModelBackend(responses=[])  # would raise if called
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(heuristic_first=True, heuristic_confidence_threshold=0.0),
        backend=backend,
        actions_taken=0,
    )
    assert decision.source == "heuristic"
    assert backend.call_count == 0


def test_decide_never_raises_when_backend_raises():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=[]),  # raises RuntimeError internally
        actions_taken=0,
    )
    assert decision.action.name in LEGAL


def test_decide_records_budget_signal():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(budget_soft_cap=50),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=12,
    )
    assert decision.budget == BudgetSignal(actions_taken=12, soft_cap=50, should_favor_execution=False)


def test_decide_memory_refreshes_when_on_and_due():
    memory = MemoryState(summary="s", step_count=8, last_refreshed_step=0)
    _, new_memory = decide(
        frame=_blank_frame(),
        history=(),
        memory=memory,
        dead_signatures=DeadSignatureTracker(),
        config=Config(memory_on=True, memory_refresh_interval=9),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert new_memory.step_count == 9
    assert new_memory.last_refreshed_step == 9


def test_decide_memory_untouched_when_off():
    memory = MemoryState(summary="s", step_count=8, last_refreshed_step=0)
    _, new_memory = decide(
        frame=_blank_frame(),
        history=(),
        memory=memory,
        dead_signatures=DeadSignatureTracker(),
        config=Config(memory_on=False),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert new_memory is memory


def test_decide_random_fallback_stays_within_legal_actions_and_never_raises():
    narrow_legal = frozenset({ActionName.ACTION7})
    decision, _ = decide(
        frame=_blank_frame(legal=narrow_legal),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=["garbage"]),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.ACTION7
    assert decision.source == "fallback_random"


def test_decide_records_target_object_label_on_heuristic_source():
    frame = _frame([[0, 0], [0, 5]])
    backend = FakeModelBackend(responses=[])
    decision, _ = decide(
        frame=frame,
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(heuristic_first=True, heuristic_confidence_threshold=0.0),
        backend=backend,
        actions_taken=0,
    )
    assert decision.source == "heuristic"
    assert decision.target_object_label == "obj0"


def test_decide_leaves_target_object_label_none_on_model_source():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=FakeModelBackend(responses=['{"action": "ACTION1"}']),
        actions_taken=0,
    )
    assert decision.source == "model"
    assert decision.target_object_label is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_policy_decide.py -v
```
Expected: FAIL — `ImportError: cannot import name 'decide' from 'zerx.policy'`.

- [ ] **Step 3: Append the orchestrator to `zerx/policy.py`**

Add these imports to the top of `zerx/policy.py` (alongside the existing `json`, `re`, `dataclass`, `FrozenSet`, `Optional` and `zerx.types` imports from Task 11):

```python
import random
from typing import FrozenSet, Optional, Tuple

from zerx.budget import BudgetSignal, evaluate_budget
from zerx.config import Config
from zerx.heuristics import DeadSignatureTracker, rank_click_candidates
from zerx.memory import MemoryState, maybe_refresh
from zerx.model_backend import ModelBackend
from zerx.perception import PerceptionResult, perceive
```

Then append this code to the end of `zerx/policy.py`:

```python
@dataclass(frozen=True)
class Decision:
    action: Action
    source: str  # "model" | "heuristic" | "fallback_heuristic" | "fallback_deterministic" | "fallback_random" | "reset"
    repaired: bool = False
    budget: Optional[BudgetSignal] = None
    target_object_label: Optional[str] = None


_FALLBACK_PREFERENCE = (
    ActionName.ACTION5,
    ActionName.ACTION1,
    ActionName.ACTION2,
    ActionName.ACTION3,
    ActionName.ACTION4,
    ActionName.ACTION6,
)


def _deterministic_fallback(
    legal_actions: FrozenSet[ActionName], grid_size: int = 64
) -> Optional[Action]:
    for name in _FALLBACK_PREFERENCE:
        if name in legal_actions:
            if name == ActionName.ACTION6:
                return Action(name=name, x=grid_size // 2, y=grid_size // 2)
            return Action(name=name)
    if ActionName.RESET in legal_actions:
        return Action(name=ActionName.RESET)
    return None


def _random_fallback(legal_actions: FrozenSet[ActionName], grid_size: int = 64) -> Action:
    name = random.choice(tuple(legal_actions))
    if name == ActionName.ACTION6:
        return Action(
            name=name, x=random.randint(0, grid_size - 1), y=random.randint(0, grid_size - 1)
        )
    return Action(name=name)


def build_prompt(perception: PerceptionResult, memory: MemoryState) -> str:
    object_lines = (
        "\n".join(
            f"- {obj.label}: color={obj.color} size={obj.size} bbox={obj.bbox}"
            for obj in perception.objects
        )
        or "(no non-background objects)"
    )
    return (
        "You are playing a grid-based puzzle game.\n"
        f"Grid:\n{perception.ascii_grid}\n\n"
        f"Objects:\n{object_lines}\n\n"
        f"What you've learned so far: {memory.summary or '(nothing yet)'}\n\n"
        'Respond with exactly one JSON object: {"action": "<ACTION_NAME>", '
        '"data": {"x": <int>, "y": <int>}} (data only required for ACTION6).'
    )


def decide(
    frame: GameFrame,
    history: Tuple[GameFrame, ...],
    memory: MemoryState,
    dead_signatures: DeadSignatureTracker,
    config: Config,
    backend: ModelBackend,
    actions_taken: int,
) -> Tuple[Decision, MemoryState]:
    """Implements the required control flow: terminal check, perception,
    heuristics, one bounded model call with deterministic repair, budget as
    a strategy signal only (never a forced/invented move), and a strict
    validated fallback chain. Never raises.
    """
    if frame.is_game_over:
        return Decision(action=Action(name=ActionName.RESET), source="reset"), memory

    legal_actions = frame.legal_actions
    perception = perceive(frame, history)
    budget = evaluate_budget(actions_taken, config.budget_soft_cap)

    candidates = rank_click_candidates(perception, dead_signatures)
    heuristic_action: Optional[Action] = None
    if candidates and ActionName.ACTION6 in legal_actions:
        top = candidates[0]
        if config.heuristic_first and top.score >= config.heuristic_confidence_threshold:
            heuristic_action = Action(name=ActionName.ACTION6, x=top.x, y=top.y)

    new_memory = memory
    if config.memory_on:
        new_memory = maybe_refresh(
            memory,
            recent_context=perception.ascii_grid,
            summarizer=lambda prev, ctx: prev,  # deterministic no-op for the local skeleton
            refresh_interval=config.memory_refresh_interval,
        )

    if heuristic_action is not None:
        return (
            Decision(
                action=heuristic_action,
                source="heuristic",
                budget=budget,
                target_object_label=top.object_label,
            ),
            new_memory,
        )

    try:
        raw = backend.generate(build_prompt(perception, new_memory))
        parsed = parse_action(raw, legal_actions)
    except Exception:
        parsed = None

    if parsed is not None:
        return (
            Decision(action=parsed.action, source="model", repaired=parsed.repaired, budget=budget),
            new_memory,
        )

    if candidates and ActionName.ACTION6 in legal_actions:
        top = candidates[0]
        return (
            Decision(
                action=Action(name=ActionName.ACTION6, x=top.x, y=top.y),
                source="fallback_heuristic",
                budget=budget,
                target_object_label=top.object_label,
            ),
            new_memory,
        )

    deterministic = _deterministic_fallback(legal_actions)
    if deterministic is not None:
        return (
            Decision(action=deterministic, source="fallback_deterministic", budget=budget),
            new_memory,
        )

    try:
        random_action = _random_fallback(legal_actions)
        return (
            Decision(action=random_action, source="fallback_random", budget=budget),
            new_memory,
        )
    except IndexError:
        return (
            Decision(action=Action(name=ActionName.RESET), source="fallback_random", budget=budget),
            new_memory,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_policy_decide.py -v
```
Expected: 12 passed.

- [ ] **Step 5: Run the full local suite**

```bash
.venv/bin/pytest tests/ -v
```
Expected: all tests across Tasks 2–12 pass (types, config, perception, heuristics, memory, budget, model_backend, cerebras_dev, secret_scan, backend_contract, policy_parse, policy_decide).

- [ ] **Step 6: Commit**

```bash
git add zerx/policy.py tests/test_policy_decide.py
git commit -m "feat(zerx): add decide() orchestrator wiring perception/heuristics/policy/memory/budget"
```

---

## Task 13: `zerx/transitions.py` — evidence-first transition ledger

Per `STRATEGY.md`'s adoption of Tycho's evidence discipline: this is
baseline infrastructure, not a gated feature. It costs no model calls and
no action budget — it's pure bookkeeping over frames we already have.

**Files:**
- Create: `zerx/transitions.py`
- Test: `tests/test_transitions.py`

**Interfaces:**
- Consumes: `GameFrame`, `Action`, `ActionName` from `zerx/types.py` (Task 2).
- Produces: `TransitionRecord` (frozen dataclass: `step`, `before_hash`, `action`, `after_hash`, `changed_pixels`, `change_bbox`, `legal_before`, `legal_after`, `score_delta`, `terminal`, `repeated_state`, plus an `.effective` property), `TransitionLedger` (`.begin(before_frame, action)`, `.finalize(after_frame) -> Optional[TransitionRecord]`, `.reset()`). `begin()`/`finalize()` are two calls straddling the harness's next `choose_action` invocation — see Task 14, which is the only caller. Never infer a transition's outcome before the next frame exists (this is exactly the ordering the two-method split enforces).

- [ ] **Step 1: Write the failing tests**

`tests/test_transitions.py`:
```python
from zerx.transitions import TransitionLedger
from zerx.types import Action, ActionName, GameFrame

DEFAULT_LEGAL = frozenset({ActionName.ACTION1, ActionName.ACTION2, ActionName.ACTION5})


def _frame(grid, legal=None, score=0, is_game_over=False):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=legal if legal is not None else DEFAULT_LEGAL,
        is_game_over=is_game_over,
        score=score,
    )


def test_finalize_without_begin_returns_none():
    ledger = TransitionLedger()
    assert ledger.finalize(_frame([[0]])) is None


def test_records_basic_transition_with_diff():
    ledger = TransitionLedger()
    before = _frame([[0, 0], [0, 0]])
    after = _frame([[0, 0], [0, 5]])
    action = Action(name=ActionName.ACTION1)
    ledger.begin(before, action)
    record = ledger.finalize(after)
    assert record.action == action
    assert record.changed_pixels == 1
    assert record.change_bbox == (1, 1, 1, 1)
    assert record.terminal is False


def test_no_change_is_flagged_repeated_and_not_effective():
    ledger = TransitionLedger()
    frame = _frame([[0, 0], [0, 5]])
    ledger.begin(frame, Action(name=ActionName.ACTION1))
    record = ledger.finalize(frame)
    assert record.changed_pixels == 0
    assert record.change_bbox is None
    assert record.repeated_state is True
    assert record.effective is False


def test_score_delta_and_terminal_make_a_transition_effective_without_pixel_change():
    before = _frame([[0]], score=1)
    after = _frame([[0]], score=3, is_game_over=True)
    ledger = TransitionLedger()
    ledger.begin(before, Action(name=ActionName.ACTION5))
    record = ledger.finalize(after)
    assert record.score_delta == 2
    assert record.terminal is True
    assert record.effective is True


def test_step_increments_across_begin_finalize_pairs():
    ledger = TransitionLedger()
    frame = _frame([[0]])
    ledger.begin(frame, Action(name=ActionName.ACTION1))
    first = ledger.finalize(frame)
    ledger.begin(frame, Action(name=ActionName.ACTION1))
    second = ledger.finalize(frame)
    assert first.step == 0
    assert second.step == 1


def test_reset_clears_pending_transition():
    ledger = TransitionLedger()
    ledger.begin(_frame([[0]]), Action(name=ActionName.ACTION1))
    ledger.reset()
    assert ledger.finalize(_frame([[0]])) is None


def test_detects_loop_beyond_the_immediate_step():
    ledger = TransitionLedger()
    frame_a = _frame([[0, 0], [0, 0]])
    frame_b = _frame([[0, 0], [0, 5]])
    ledger.begin(frame_a, Action(name=ActionName.ACTION1))
    ledger.finalize(frame_b)
    ledger.begin(frame_b, Action(name=ActionName.ACTION2))
    record = ledger.finalize(frame_a)  # back to frame_a's exact state
    assert record.repeated_state is True


def test_records_legal_actions_before_and_after():
    before = _frame([[0]], legal=frozenset({ActionName.ACTION1}))
    after = _frame([[0]], legal=frozenset({ActionName.ACTION1, ActionName.ACTION5}))
    ledger = TransitionLedger()
    ledger.begin(before, Action(name=ActionName.ACTION1))
    record = ledger.finalize(after)
    assert record.legal_before == frozenset({ActionName.ACTION1})
    assert record.legal_after == frozenset({ActionName.ACTION1, ActionName.ACTION5})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_transitions.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'zerx.transitions'`.

- [ ] **Step 3: Implement `zerx/transitions.py`**

```python
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
from typing import Deque, FrozenSet, Optional, Tuple

from zerx.types import Action, ActionName, GameFrame


def _grid_hash(frame: GameFrame) -> str:
    flat = ",".join(str(v) for row in frame.grid for v in row)
    return hashlib.sha256(flat.encode("utf-8")).hexdigest()[:16]


def _diff(
    before: GameFrame, after: GameFrame
) -> Tuple[int, Optional[Tuple[int, int, int, int]]]:
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

    def finalize(self, after: GameFrame) -> Optional[TransitionRecord]:
        if self._pending is None:
            return None
        step, before, action = self._pending
        self._pending = None
        before_hash = _grid_hash(before)
        after_hash = _grid_hash(after)
        changed_pixels, bbox = _diff(before, after)
        repeated_state = after_hash in self._recent_hashes
        return TransitionRecord(
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_transitions.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add zerx/transitions.py tests/test_transitions.py
git commit -m "feat(zerx): add evidence-first transition ledger"
```

---

## Task 14: `agent/my_agent.py` — thin harness adapter

This is also where the evidence ledger (Task 13) and graded negative
affordances (Task 5) actually get wired together into a loop — neither
module calls the other; the adapter is the one place that observes both
"what we did" and "what happened next" per `STRATEGY.md`'s rule: never
infer an action's consequence before the next frame exists.

**Files:**
- Modify: `agent/my_agent.py` (replace the random-baseline body imported in Task 1)

**Interfaces:**
- Consumes: `decide()`, `Decision` (Task 12), `Config` (Task 3), `MemoryState` (Task 6), `DeadSignatureTracker` (Task 5), `TransitionLedger` (Task 13), `GemmaModelBackend` (Task 8), `GameFrame`/`Action`/`ActionName` (Task 2), and the real upstream `Agent`/`GameAction`/frame-data types recorded in `docs/superpowers/experiments/baseline-000.md` (Task 1, Step 5).
- Produces: `MyAgent` — the class the Kaggle harness imports and drives.

- [ ] **Step 1: Re-read the API notes from Task 1**

Open `docs/superpowers/experiments/baseline-000.md` and re-read the recorded import lines and frame attribute/method names. The adapter below uses illustrative placeholder names (`frame.frame`, `frame.available_actions`, `frame.state`, `agent.framework`) — every one of these **must** be replaced with the real names from that file before this task is considered done. Do not guess; if a name doesn't match, re-open the vendored `agent/my_agent.py`'s original body (visible in `git show HEAD~1:agent/my_agent.py` after Task 1's commit) and correct the note file too.

- [ ] **Step 2: Replace `agent/my_agent.py`**

```python
"""Thin harness adapter — translates the real upstream Frame/GameAction API
(see docs/superpowers/experiments/baseline-000.md for the exact names) into
zerx's internal types, delegates to zerx.policy.decide(), finalizes the
previous transition and feeds soft-affordance outcomes back, and
translates the result back. Keep this file free of policy logic; it is
glue only.
"""
from __future__ import annotations

from typing import List, Tuple

# Replace this import with the exact line(s) recorded in baseline-000.md.
from agent.framework import Agent, GameAction, FrameData

from zerx.config import Config
from zerx.heuristics import DeadSignatureTracker
from zerx.memory import MemoryState
from zerx.model_backend import GemmaModelBackend
from zerx.perception import perceive
from zerx.policy import Decision, decide
from zerx.transitions import TransitionLedger
from zerx.types import Action, ActionName, GameFrame


def _to_game_frame(frame: "FrameData") -> GameFrame:
    """Translate the upstream frame object into our internal GameFrame.
    Replace `frame.frame`, `frame.available_actions`, `frame.state`, and
    `frame.score` below with whatever Task 1's inspection actually found
    (score may not exist upstream under that exact name — if there's no
    numeric score field, default to 0 rather than guessing a wrong one;
    score_delta then reads as always 0, which is honest, not broken).
    """
    grid = tuple(tuple(row) for row in frame.frame)
    legal = frozenset(ActionName[a.name] for a in frame.available_actions)
    is_game_over = frame.state == "GAME_OVER"
    score = getattr(frame, "score", 0)
    return GameFrame(grid=grid, legal_actions=legal, is_game_over=is_game_over, score=score)


def _to_game_action(action: Action) -> "GameAction":
    upstream = GameAction[action.name.value]
    if action.name == ActionName.ACTION6:
        return upstream.set_data({"x": action.x, "y": action.y})
    return upstream


def _find_object_by_label(frame: GameFrame, label: str):
    """Recompute perception just far enough to look up the LabeledObject a
    past Decision targeted, so its outcome can be recorded. Cheap (no GPU,
    no model call) — perception is already deterministic and pure.
    """
    for obj in perceive(frame).objects:
        if obj.label == label:
            return obj
    return None


class MyAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._config = Config.from_env()
        self._memory = MemoryState()
        self._dead_signatures = DeadSignatureTracker()
        self._backend = GemmaModelBackend(self._config.model_revision)
        self._transitions = TransitionLedger()
        self._actions_taken = 0
        self._pending_decision: Decision | None = None
        self._pending_before_frame: GameFrame | None = None

    def is_done(self, frames: List["FrameData"], latest_frame: "FrameData") -> bool:
        return latest_frame.state == "GAME_OVER"

    def choose_action(
        self, frames: List["FrameData"], latest_frame: "FrameData"
    ) -> "GameAction":
        frame = _to_game_frame(latest_frame)

        # Finalize the PREVIOUS action's transition now that its result
        # (this frame) exists. Never do this before the frame arrives.
        record = self._transitions.finalize(frame)
        if (
            record is not None
            and self._pending_decision is not None
            and self._pending_decision.target_object_label is not None
            and self._pending_before_frame is not None
        ):
            target = _find_object_by_label(
                self._pending_before_frame, self._pending_decision.target_object_label
            )
            if target is not None:
                self._dead_signatures.record_outcome(target, effective=record.effective)

        history: Tuple[GameFrame, ...] = tuple(_to_game_frame(f) for f in frames[-4:])
        decision, self._memory = decide(
            frame=frame,
            history=history,
            memory=self._memory,
            dead_signatures=self._dead_signatures,
            config=self._config,
            backend=self._backend,
            actions_taken=self._actions_taken,
        )
        self._actions_taken += 1

        self._transitions.begin(frame, decision.action)
        self._pending_decision = decision
        self._pending_before_frame = frame

        return _to_game_action(decision.action)
```

- [ ] **Step 3: Fix import/attribute mismatches by running the 30-second smoke test**

```bash
make verify-local
```
Expected (once names are corrected to match the real API): completes without a Python exception. Since `GemmaModelBackend.generate()` still raises `NotImplementedError` (Task 8), a real playthrough isn't expected to succeed yet — but `AttributeError`/`ImportError` must be gone. If it still raises `NotImplementedError` from deep inside `decide()`'s `try/except Exception`, that's expected and means `decide()` correctly fell back per Task 12's tests: check that `make verify-local`'s output shows fallback actions being taken, not a crash.

- [ ] **Step 4: Commit**

```bash
git add agent/my_agent.py docs/superpowers/experiments/baseline-000.md
git commit -m "feat(agent): wire MyAgent to decide()/transitions/soft-affordance feedback loop"
```

---

## Task 15: `eval/run_ablation.py` — experiment record schema and config sweep

**Files:**
- Create: `eval/__init__.py` (empty)
- Create: `eval/run_ablation.py`
- Test: `tests/test_run_ablation.py`

**Interfaces:**
- Consumes: `Config` (Task 3).
- Produces: `ExperimentRecord` (dataclass matching the reproducibility fields in `AGENTS.md`), `write_records(records, path) -> None` (append JSONL), `sweep_configs(base, **variants) -> List[Config]` (one-flag-at-a-time sweep, per `AGENTS.md`'s "one behavioral change per experiment" rule).

- [ ] **Step 1: Write the failing tests**

`tests/test_run_ablation.py`:
```python
import json

from eval.run_ablation import ExperimentRecord, sweep_configs, write_records
from zerx.config import Config


def _record(**overrides):
    defaults = dict(
        experiment_id="exp-1",
        config_hash="abc123",
        game_id="ls20",
        actions_taken=10,
        levels_completed=1,
        rhae=0.5,
        wall_time_seconds=1.2,
        invalid_outputs=0,
        repairs=0,
        fallbacks=0,
        resets=0,
        exceptions=0,
    )
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


def test_experiment_record_to_json_line_is_valid_json():
    line = _record().to_json_line()
    payload = json.loads(line)
    assert payload["game_id"] == "ls20"
    assert payload["rhae"] == 0.5


def test_write_records_appends_jsonl(tmp_path):
    path = tmp_path / "results.jsonl"
    write_records([_record(game_id="ls20"), _record(game_id="vc33")], path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["game_id"] == "ls20"
    assert json.loads(lines[1])["game_id"] == "vc33"


def test_write_records_appends_to_existing_file(tmp_path):
    path = tmp_path / "results.jsonl"
    write_records([_record(game_id="ls20")], path)
    write_records([_record(game_id="vc33")], path)
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_sweep_configs_includes_base():
    base = Config()
    configs = sweep_configs(base)
    assert configs == [base]


def test_sweep_configs_varies_one_field_at_a_time():
    base = Config(heuristic_first=False, memory_on=True)
    configs = sweep_configs(base, heuristic_first=[True])
    assert len(configs) == 2
    variant = configs[1]
    assert variant.heuristic_first is True
    assert variant.memory_on is True  # everything else stays at base


def test_sweep_configs_skips_value_equal_to_base():
    base = Config(heuristic_first=False)
    configs = sweep_configs(base, heuristic_first=[False])
    assert configs == [base]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_run_ablation.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'eval'` (or `eval.run_ablation`).

- [ ] **Step 3: Implement `eval/run_ablation.py`**

Create empty `eval/__init__.py`, then:

```python
"""Experiment record schema, JSONL writer, and Config-variant sweep
generator used by ablation runs. The reproducibility fields here match
AGENTS.md's "Configuration and reproducibility" section. The actual
"play N local games with this config" loop is wired in once
agent/my_agent.py's harness adapter (Task 14) is exercised against real
games — this module owns the record format independent of that wiring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, List, Optional

from zerx.config import Config


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    config_hash: str
    game_id: str
    actions_taken: int
    levels_completed: int
    rhae: Optional[float]
    wall_time_seconds: float
    invalid_outputs: int
    repairs: int
    fallbacks: int
    resets: int
    exceptions: int

    def to_json_line(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


def write_records(records: Iterable[ExperimentRecord], path: Path) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.to_json_line() + "\n")


def sweep_configs(base: Config, **variants: List[bool]) -> List[Config]:
    """Single-flag-at-a-time sweep: for each keyword arg (a Config field
    name) and its list of candidate values, yield one Config per value with
    everything else held at `base`. Matches AGENTS.md's "one behavioral
    change per experiment where possible" rule rather than a combinatorial
    explosion.
    """
    configs = [base]
    for field_name, values in variants.items():
        for value in values:
            if value == getattr(base, field_name):
                continue
            configs.append(replace(base, **{field_name: value}))
    return configs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_run_ablation.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Run the complete local suite one final time**

```bash
.venv/bin/pytest tests/ -v
```
Expected: all tests pass (Tasks 2–15). This is the "Local → Colab" promotion gate from `AGENTS.md` — full unit suite passes, fake-backend end-to-end coverage exists, terminal state returns `RESET`, malformed output reaches the documented fallback, every action produces a transition record (`baseline-110-evidence` per `STRATEGY.md`), competition-mode configuration rejects `cerebras_dev`, the secret scan passes, no model weights or secrets are committed.

- [ ] **Step 6: Commit**

```bash
git add eval/__init__.py eval/run_ablation.py tests/test_run_ablation.py
git commit -m "feat(eval): add experiment record schema, JSONL writer, config sweep"
```

---

## What this plan does not cover

Per the spec's scope and `AGENTS.md`/`docs/TEAM_WORKFLOW.md`'s phasing, the following are deliberately out of scope here and belong in a follow-on plan or a separately-approved action once this one lands:

- Loading Gemma-4-31B for real (Colab Pro A100/L4 development — `GemmaModelBackend.generate()` stays `NotImplementedError` until then).
- Making a real (non-mocked) call to the Cerebras API — Task 9's tests all inject a fake `http_post`. A live-network Cerebras smoke test (real `CEREBRAS_API_KEY`, querying the account's actual available model IDs) is opt-in and separately marked per `AGENTS.md`, not part of this plan's default suite.
- Running `eval/run_ablation.py`'s sweeps against real games with a real model (Cerebras or Gemma).
- Any Kaggle packaging, `make submit`, or official submission — including the Day 1 "known-working smoke submission" `docs/TEAM_WORKFLOW.md` calls for. That's a quota-consuming, hard-to-reverse action requiring explicit user approval per `AGENTS.md`'s Kaggle gate, not something this plan automates.
- Everything in `STRATEGY.md`'s "Adopt later" section and experiment ladder past `baseline-110-evidence`: structured belief/hypothesis tracking (`baseline-120`/`baseline-130`), the executable world model, planner, and builder specialist (`exp-200`/`exp-210`/`exp-220`). Each is a separate, isolated, off-by-default follow-on plan once this one is built and green — none of it is scaffolded here, not even behind a flag.
- Rewriting `zerx/memory.py` into Tycho-style structured fields — `STRATEGY.md` says explicitly not to do this before `baseline-100-minimal`/`baseline-110-evidence` are stable.
