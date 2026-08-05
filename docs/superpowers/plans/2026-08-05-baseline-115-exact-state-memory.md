# baseline-115-exact-state-memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact `(state, action)` ineffective-action suppression (STRATEGY.md §3.1 / §7 `baseline-115-exact-state-memory`) as a new, off-by-default module wired alongside the existing graded `DeadSignatureTracker`, without touching `decide()`'s signature or any other parallel track's files.

**Architecture:** A new `zerx/exact_state_memory.py` module tracks outcomes keyed by `(grid state hash, action signature)`. `agent/my_agent.py` feeds it every action's outcome (not just heuristic-sourced ones) and, after `decide()` returns, swaps out a decision whose exact `(state, action)` pair is already known to be a no-op for the next legal action in preference order. `zerx/transitions.py` gets one purely-additive line promoting its existing private hash helper to a public `grid_hash()` so both the new module's caller and the existing ledger share one hashing implementation.

**Tech Stack:** Python 3, pytest, dataclasses (frozen, `replace`). No new dependencies.

## Global Constraints

- Feature ships OFF by default: `Config.exact_state_suppression_on: bool = False`. With no env vars set, `decide()`/`choose_action()` behavior must be byte-identical to before this change, for every existing test (`docs/superpowers/plans/parallel-day3/README.md`).
- Do not change `zerx/policy.py`'s `decide()` function signature (`README.md`). This plan uses the "post-check after `decide()` returns, inside `agent/my_agent.py`" option explicitly offered by `docs/superpowers/plans/parallel-day3/person-1-baseline-115.md`.
- `zerx/config.py`: add the new field at the end of the existing field list, and the matching `from_env(...)` line at the end of that method's return-call argument list. Never touch, reorder, or add validation to existing fields (`README.md`).
- `agent/my_agent.py`: any new block is wrapped in `# --- baseline-115-exact-state-memory (feat/baseline-115-exact-state-memory) ---` / `# --- end baseline-115-exact-state-memory ---` comment banners, placed immediately after existing unmodified code, never inside an existing `if`/`try` block, and guarded by `if self._config.exact_state_suppression_on:` (`README.md`).
- `DeadSignatureTracker` (`zerx/heuristics.py`) is untouched — this is a separate, narrower layer alongside it (`person-1-baseline-115.md`, "Explicitly out of scope").
- Suppression must stay graded, not permanent, per STRATEGY.md §2.5/§3.1: "Suppression should still not be a permanent universal ban on the underlying action/object type — only on that exact (state, action) pair" — implemented here via `later_disconfirmed` (see Task 2's design note).
- Full existing test suite (136 tests as of this handoff, per `docs/HANDOFF.md`) must stay green throughout; new tests are additive.
- Push only to `feat/baseline-115-exact-state-memory`. Do not merge to `master`. Do not touch Kaggle/Cerebras/`scripts/build_notebook.py`/`scripts/build_colab_notebook.py` — out of scope.

## Design decisions (STRATEGY.md leaves these to the implementer — recorded here per `person-1-baseline-115.md`'s instruction)

1. **`later_disconfirmed` semantics — operational, not purely diagnostic.** A suppression is lifted (and stays lifted) the first time the *same exact* `(state, action)` pair later produces a visible change or a level delta. Rationale: STRATEGY.md's own closing sentence in §3.1 says suppression must not become "a permanent universal ban," and "graded, not hard" (§2.5) is the project's consistent stance on negative evidence. Once a pair's outcome has been observed to be inconsistent (same state, different result — plausibly a hidden-state/timing effect per §2.5), the "identical state → identical outcome" assumption that justifies suppression at all no longer holds for that pair, so it should never re-suppress, not just skip one cycle.
2. **Where the post-`decide()` check lives:** inside `agent/my_agent.py`'s `_choose_action_inner`, right after the `decide(...)` call and before `self._actions_taken += 1` / `self._transitions.begin(...)`. This is option (a) from `person-1-baseline-115.md` — it keeps `decide()`'s signature and every other track's planned "one new optional kwarg" option untouched, minimizing merge risk.
3. **Replacement action selection:** when the decided action's exact `(state, action)` pair is suppressed, do not call `zerx.policy._deterministic_fallback` verbatim (it is blind to suppression and would deterministically re-select the very same action when that action is also the top `_FALLBACK_PREFERENCE` entry — a no-op fix). Instead, walk `zerx.policy._FALLBACK_PREFERENCE` for the first **legal** action whose **name** differs from the suppressed action's name. Excluding by name (not exact ACTION6 coordinates) is a deliberate simplification: it guarantees termination in one pass over a 6-entry tuple and never risks re-proposing coordinates it hasn't checked, at the cost of occasionally skipping a still-valid different-coordinate ACTION6. Documented here rather than building a second candidate-search loop, consistent with STRATEGY §2.1 ("simplest complete" over machinery).
4. **`grid_hash` promotion:** `zerx/transitions.py`'s `_grid_hash` becomes a public `grid_hash`, with `_grid_hash` kept as an alias so every existing internal call site is untouched. Purely additive; no other Day-3 track touches `zerx/transitions.py` (confirmed against `README.md`'s branch table), so this carries no merge risk.

5. **Post-review amendment (commit `e5bb04d`):** the final whole-branch review found two Critical defects in decision #3's original "walk `_FALLBACK_PREFERENCE` by name" design, both now fixed in `agent/my_agent.py`:
   - The suppression-check block must never run when `frame.is_game_over` — `decide()`'s own terminal short-circuit always returns RESET on a GAME_OVER/NOT_PLAYED frame, and swapping that away (which could happen once `(state, RESET)` itself gets recorded as a no-op) permanently breaks the reset-and-retry loop. Fixed by adding `not frame.is_game_over` to the block's guard.
   - The original design only checked that a replacement candidate's *name* differed from the suppressed action's name — it never checked whether the candidate itself was *also* a known suppressed no-op for the same state, so the agent could get stuck cycling a single suppressed alternative forever, defeating the feature's purpose. Fixed by checking `is_suppressed` on each candidate while walking `_FALLBACK_PREFERENCE`, taking the first non-suppressed one (or leaving `decision` unchanged if all are exhausted).
   Two regression tests (`tests/test_my_agent_exact_state.py`) now cover both cases. See the review notes and `e5bb04d`'s commit message for full detail.

---

### Task 1: Promote `zerx/transitions.py`'s grid hash helper to a public function

**Files:**
- Modify: `zerx/transitions.py:17-19`
- Test: `tests/test_transitions.py`

**Interfaces:**
- Produces: `zerx.transitions.grid_hash(frame: GameFrame) -> str` — public, used by Task 4's wiring in `agent/my_agent.py` and available for `zerx/exact_state_memory.py` callers.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transitions.py` (append at the end of the file, keep the existing `from zerx.transitions import TransitionLedger` import line and extend it):

```python
# --- add to the import line at the top of the file ---
from zerx.transitions import TransitionLedger, grid_hash

# --- append at the end of the file ---
def test_grid_hash_is_public_and_deterministic():
    frame = _frame([[1, 2], [3, 4]])
    assert grid_hash(frame) == grid_hash(frame)
    assert isinstance(grid_hash(frame), str)


def test_grid_hash_differs_for_different_grids():
    a = _frame([[0, 0], [0, 0]])
    b = _frame([[0, 0], [0, 1]])
    assert grid_hash(a) != grid_hash(b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\pytest.exe tests/test_transitions.py -q`
Expected: FAIL with `ImportError: cannot import name 'grid_hash' from 'zerx.transitions'`

- [ ] **Step 3: Promote the function**

In `zerx/transitions.py`, replace:

```python
def _grid_hash(frame: GameFrame) -> str:
    flat = ",".join(str(v) for row in frame.grid for v in row)
    return hashlib.sha256(flat.encode("utf-8")).hexdigest()[:16]
```

with:

```python
def grid_hash(frame: GameFrame) -> str:
    flat = ",".join(str(v) for row in frame.grid for v in row)
    return hashlib.sha256(flat.encode("utf-8")).hexdigest()[:16]


_grid_hash = grid_hash
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\pytest.exe tests/test_transitions.py -q`
Expected: all tests in the file PASS (existing 9 + 2 new = 11)

- [ ] **Step 5: Commit**

```bash
git add zerx/transitions.py tests/test_transitions.py
git commit -m "refactor(transitions): promote grid hash helper to public grid_hash()"
```

---

### Task 2: `zerx/exact_state_memory.py` — new tracker module

**Files:**
- Create: `zerx/exact_state_memory.py`
- Test: Create `tests/test_exact_state_memory.py`

**Interfaces:**
- Consumes: `zerx.types.Action`, `zerx.types.ActionName` (existing).
- Produces:
  - `action_signature(action: Action) -> str`
  - `ExactStateRecord` frozen dataclass with fields `state_signature: str`, `action_signature: str`, `attempt_count: int`, `visible_change: bool`, `level_delta: int`, `later_disconfirmed: bool`
  - `ExactStateMemory` with methods `record_outcome(state_signature: str, action_signature: str, visible_change: bool, level_delta: int) -> None`, `is_suppressed(state_signature: str, action_signature: str) -> bool`, `record_for(state_signature: str, action_signature: str) -> Optional[ExactStateRecord]`, `reset() -> None`
  - Used by Task 4's `agent/my_agent.py` wiring.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_exact_state_memory.py`:

```python
from zerx.exact_state_memory import ExactStateMemory, action_signature
from zerx.types import Action, ActionName


def test_new_pair_has_no_suppression():
    memory = ExactStateMemory()
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_ineffective_outcome_suppresses_exact_pair():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is True


def test_effective_outcome_does_not_suppress():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=True, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_level_delta_outcome_does_not_suppress():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=1)
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_different_action_same_state_not_suppressed():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION2") is False


def test_same_action_different_state_not_suppressed():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-b", "ACTION1") is False


def test_later_disconfirmed_lifts_suppression_and_it_stays_lifted():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is True

    # Same exact (state, action) pair later produces a real change --
    # contradicts "identical state -> identical outcome", so suppression
    # must lift.
    memory.record_outcome("state-a", "ACTION1", visible_change=True, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is False

    # A subsequent no-op observation must NOT re-suppress a disconfirmed pair.
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_reset_clears_all_records():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    memory.reset()
    assert memory.is_suppressed("state-a", "ACTION1") is False


def test_attempt_count_increments_on_repeated_recording():
    memory = ExactStateMemory()
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    memory.record_outcome("state-a", "ACTION1", visible_change=False, level_delta=0)
    record = memory.record_for("state-a", "ACTION1")
    assert record.attempt_count == 2


def test_record_for_returns_none_when_absent():
    memory = ExactStateMemory()
    assert memory.record_for("state-a", "ACTION1") is None


def test_action_signature_stable_for_non_action6():
    action = Action(name=ActionName.ACTION3)
    assert action_signature(action) == "ACTION3"


def test_action_signature_distinguishes_action6_coordinates():
    a = Action(name=ActionName.ACTION6, x=10, y=20)
    b = Action(name=ActionName.ACTION6, x=11, y=20)
    assert action_signature(a) != action_signature(b)


def test_action_signature_same_action6_coordinates_match():
    a = Action(name=ActionName.ACTION6, x=10, y=20)
    b = Action(name=ActionName.ACTION6, x=10, y=20)
    assert action_signature(a) == action_signature(b)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_exact_state_memory.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'zerx.exact_state_memory'`

- [ ] **Step 3: Write the implementation**

Create `zerx/exact_state_memory.py`:

```python
"""Exact (state, action) ineffective-action suppression (STRATEGY.md §3.1,
`baseline-115-exact-state-memory`). Narrower and more confident than
`zerx.heuristics.DeadSignatureTracker`'s structural down-ranking: when the
literal same grid state has already produced zero visible change and zero
level delta for the literal same action, suppress proposing that exact
pair again -- but only until contradicting evidence (`later_disconfirmed`)
arrives for that same pair, since suppression must stay graded, never a
permanent ban (STRATEGY.md §2.5).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

from zerx.types import Action, ActionName


def action_signature(action: Action) -> str:
    """A stable string identity for an Action, distinguishing ACTION6
    clicks by exact coordinate (two different click targets are two
    different actions for suppression purposes).
    """
    if action.name == ActionName.ACTION6:
        return f"{action.name.value}:{action.x},{action.y}"
    return action.name.value


@dataclass(frozen=True)
class ExactStateRecord:
    state_signature: str
    action_signature: str
    attempt_count: int
    visible_change: bool
    level_delta: int
    later_disconfirmed: bool


class ExactStateMemory:
    """Per-(state, action) outcome memory. `record_outcome` is called for
    every action's transition result (not just heuristic-sourced ones);
    `is_suppressed` is consulted before accepting a proposed action against
    the current state.
    """

    def __init__(self) -> None:
        self._records: Dict[Tuple[str, str], ExactStateRecord] = {}

    def record_outcome(
        self,
        state_signature: str,
        action_signature: str,
        visible_change: bool,
        level_delta: int,
    ) -> None:
        key = (state_signature, action_signature)
        existing = self._records.get(key)
        if existing is None:
            self._records[key] = ExactStateRecord(
                state_signature=state_signature,
                action_signature=action_signature,
                attempt_count=1,
                visible_change=visible_change,
                level_delta=level_delta,
                later_disconfirmed=False,
            )
            return

        was_ineffective = not existing.visible_change and existing.level_delta == 0
        now_effective = visible_change or level_delta != 0
        later_disconfirmed = existing.later_disconfirmed or (was_ineffective and now_effective)

        self._records[key] = replace(
            existing,
            attempt_count=existing.attempt_count + 1,
            visible_change=visible_change,
            level_delta=level_delta,
            later_disconfirmed=later_disconfirmed,
        )

    def is_suppressed(self, state_signature: str, action_signature: str) -> bool:
        record = self._records.get((state_signature, action_signature))
        if record is None:
            return False
        return (
            not record.visible_change
            and record.level_delta == 0
            and not record.later_disconfirmed
        )

    def record_for(self, state_signature: str, action_signature: str) -> Optional[ExactStateRecord]:
        return self._records.get((state_signature, action_signature))

    def reset(self) -> None:
        """Clear between games -- exact-state evidence from one game must
        never leak into the next, same discipline as MemoryState.reset(),
        DeadSignatureTracker.reset(), and TransitionLedger.reset().
        """
        self._records.clear()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_exact_state_memory.py -q`
Expected: all 13 tests PASS

- [ ] **Step 5: Commit**

```bash
git add zerx/exact_state_memory.py tests/test_exact_state_memory.py
git commit -m "feat(exact-state-memory): add exact (state, action) suppression tracker"
```

---

### Task 3: `zerx/config.py` — add `exact_state_suppression_on` field

**Files:**
- Modify: `zerx/config.py:35-77`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.exact_state_suppression_on: bool` (default `False`), populated from env var `ZERX_EXACT_STATE_SUPPRESSION_ON` via `Config.from_env`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_default_exact_state_suppression_is_off():
    cfg = Config()
    assert cfg.exact_state_suppression_on is False


def test_from_env_overrides_exact_state_suppression_on():
    cfg = Config.from_env({"ZERX_EXACT_STATE_SUPPRESSION_ON": "true"})
    assert cfg.exact_state_suppression_on is True


def test_from_env_missing_exact_state_suppression_keeps_default():
    cfg = Config.from_env({})
    assert cfg.exact_state_suppression_on is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_config.py -q`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'exact_state_suppression_on'`

- [ ] **Step 3: Add the field**

In `zerx/config.py`, replace:

```python
    backend: str = "fake"  # "fake" | "cerebras_dev" | "gemma_local" | "gemma_kaggle"
    platform: str = "local"  # "local" | "colab" | "kaggle"

    def __post_init__(self) -> None:
```

with:

```python
    backend: str = "fake"  # "fake" | "cerebras_dev" | "gemma_local" | "gemma_kaggle"
    platform: str = "local"  # "local" | "colab" | "kaggle"
    exact_state_suppression_on: bool = False

    def __post_init__(self) -> None:
```

Then replace:

```python
            backend=_env_str(env, "ZERX_BACKEND", cls.backend),
            platform=_env_str(env, "ZERX_PLATFORM", cls.platform),
        )
```

with:

```python
            backend=_env_str(env, "ZERX_BACKEND", cls.backend),
            platform=_env_str(env, "ZERX_PLATFORM", cls.platform),
            exact_state_suppression_on=_env_bool(
                env, "ZERX_EXACT_STATE_SUPPRESSION_ON", cls.exact_state_suppression_on
            ),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_config.py -q`
Expected: all tests PASS (existing 13 + 3 new = 16)

- [ ] **Step 5: Commit**

```bash
git add zerx/config.py tests/test_config.py
git commit -m "feat(config): add exact_state_suppression_on flag, off by default"
```

---

### Task 4: Wire `ExactStateMemory` into `agent/my_agent.py`

**Files:**
- Modify: `agent/my_agent.py:16-34` (imports), `:133-143` (`__init__`), `:150-184` (`_choose_action_inner`)
- Test: Create `tests/test_my_agent_exact_state.py`

**Interfaces:**
- Consumes: `zerx.exact_state_memory.ExactStateMemory`, `zerx.exact_state_memory.action_signature` (Task 2); `zerx.transitions.grid_hash` (Task 1); `zerx.policy._deterministic_fallback`, `zerx.policy._FALLBACK_PREFERENCE` (existing, private, imported directly per `person-1-baseline-115.md`'s explicit allowance); `Config.exact_state_suppression_on` (Task 3).
- Produces: `MyAgent._exact_state_memory: ExactStateMemory` instance attribute; a new `Decision.source` value `"fallback_exact_state_suppressed"` (no schema change — `Decision.source` is a plain `str` field, this is an additive value).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_my_agent_exact_state.py`:

```python
"""Tests for baseline-115-exact-state-memory's wiring into
agent/my_agent.py: feeding every action's outcome into ExactStateMemory,
and swapping out a decision whose exact (state, action) pair is already
known to be a no-op. Mirrors tests/test_my_agent.py's setup (real vendored
ARC-AGI-3-Agents framework, arc_env=None, no live game environment
needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from arcengine import FrameData, GameAction, GameState  # noqa: E402

from agent.my_agent import MyAgent  # noqa: E402


def _make_agent(monkeypatch, suppression_on: bool) -> MyAgent:
    if suppression_on:
        monkeypatch.setenv("ZERX_EXACT_STATE_SUPPRESSION_ON", "true")
    else:
        monkeypatch.delenv("ZERX_EXACT_STATE_SUPPRESSION_ON", raising=False)
    return MyAgent(
        card_id="test-card",
        game_id="test-game",
        agent_name="test-agent",
        ROOT_URL="http://example.invalid",
        record=False,
        arc_env=None,
    )


def _uniform_frame() -> FrameData:
    # All-zero grid -> perception finds zero non-background objects, so
    # decide() never proposes an ACTION6 click and no model backend call
    # can succeed (no server listening) -> it deterministically falls
    # through to zerx.policy._deterministic_fallback, which is the code
    # path this test exercises.
    return FrameData(
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 5],  # -> legal = {ACTION1, ACTION5, RESET}
    )


def test_exact_state_suppression_off_by_default_repeats_the_same_fallback_action(monkeypatch):
    frame = _uniform_frame()
    agent = _make_agent(monkeypatch, suppression_on=False)

    first = agent.choose_action([frame], frame)
    second = agent.choose_action([frame, frame], frame)

    # ACTION5 is _FALLBACK_PREFERENCE's first legal entry both times --
    # suppression is off, so nothing changes that.
    assert first is GameAction.ACTION5
    assert second is GameAction.ACTION5


def test_exact_state_suppression_on_swaps_a_known_noop_for_the_next_legal_action(monkeypatch):
    frame = _uniform_frame()
    agent = _make_agent(monkeypatch, suppression_on=True)

    first = agent.choose_action([frame], frame)
    assert first is GameAction.ACTION5

    # Same frame again: zero visible change, zero score delta for the
    # pending ACTION5 -> recorded as a known no-op for this exact
    # (state, action) pair BEFORE decide() runs again this same call ->
    # decide() would deterministically re-propose ACTION5, but the
    # post-check swaps it for the next legal preference (ACTION1) instead.
    second = agent.choose_action([frame, frame], frame)
    assert second is GameAction.ACTION1
    assert second.reasoning["source"] == "fallback_exact_state_suppressed"


def test_exact_state_suppression_does_not_affect_the_first_ever_action(monkeypatch):
    """With no prior evidence, ExactStateMemory is empty -- the first call
    on any frame must behave identically whether the flag is on or off.
    """
    frame = _uniform_frame()
    agent_off = _make_agent(monkeypatch, suppression_on=False)
    result_off = agent_off.choose_action([frame], frame)

    frame2 = _uniform_frame()
    agent_on = _make_agent(monkeypatch, suppression_on=True)
    result_on = agent_on.choose_action([frame2], frame2)

    assert result_off is result_on is GameAction.ACTION5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_my_agent_exact_state.py -q`
Expected: FAIL — `test_exact_state_suppression_on_swaps_a_known_noop_for_the_next_legal_action` fails with `assert GameAction.ACTION5 is GameAction.ACTION1` (config field already exists from Task 3 so no crash, but nothing consults it yet). The other two tests currently PASS already (unchanged behavior) — that's expected, they exist to lock in the off-by-default guarantee going forward.

- [ ] **Step 3: Add the wiring**

In `agent/my_agent.py`, replace the import block:

```python
import logging
from typing import List, Optional, Tuple
```

with:

```python
import logging
from dataclasses import replace
from typing import List, Optional, Tuple
```

Replace:

```python
from zerx.config import Config
from zerx.heuristics import DeadSignatureTracker
from zerx.memory import MemoryState
from zerx.model_backend import GemmaModelBackend
from zerx.perception import perceive
from zerx.policy import Decision, decide
from zerx.transitions import TransitionLedger
from zerx.types import Action, ActionName, GameFrame
```

with:

```python
from zerx.config import Config
from zerx.exact_state_memory import ExactStateMemory, action_signature
from zerx.heuristics import DeadSignatureTracker
from zerx.memory import MemoryState
from zerx.model_backend import GemmaModelBackend
from zerx.perception import perceive
from zerx.policy import Decision, _deterministic_fallback, _FALLBACK_PREFERENCE, decide
from zerx.transitions import TransitionLedger, grid_hash
from zerx.types import Action, ActionName, GameFrame
```

Replace `__init__`:

```python
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._config = Config.from_env()
        self._memory = MemoryState()
        self._dead_signatures = DeadSignatureTracker()
        self._backend = GemmaModelBackend(self._config.model_revision)
        self._transitions = TransitionLedger()
        self._actions_taken = 0
        self._pending_decision: Optional[Decision] = None
        self._pending_before_frame: Optional[GameFrame] = None
```

with:

```python
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._config = Config.from_env()
        self._memory = MemoryState()
        self._dead_signatures = DeadSignatureTracker()
        self._backend = GemmaModelBackend(self._config.model_revision)
        self._transitions = TransitionLedger()
        self._exact_state_memory = ExactStateMemory()
        self._actions_taken = 0
        self._pending_decision: Optional[Decision] = None
        self._pending_before_frame: Optional[GameFrame] = None
```

Replace the outcome-feedback block in `_choose_action_inner` (add the new block right after it, do not modify the existing `if`):

```python
            if target is not None:
                self._dead_signatures.record_outcome(target, effective=record.effective)

        history: Tuple[GameFrame, ...] = tuple(_to_game_frame(f) for f in frames[-4:])
```

with:

```python
            if target is not None:
                self._dead_signatures.record_outcome(target, effective=record.effective)

        # --- baseline-115-exact-state-memory (feat/baseline-115-exact-state-memory) ---
        if (
            self._config.exact_state_suppression_on
            and record is not None
            and self._pending_decision is not None
            and self._pending_before_frame is not None
        ):
            self._exact_state_memory.record_outcome(
                state_signature=grid_hash(self._pending_before_frame),
                action_signature=action_signature(self._pending_decision.action),
                visible_change=record.changed_pixels > 0,
                level_delta=record.score_delta,
            )
        # --- end baseline-115-exact-state-memory ---

        history: Tuple[GameFrame, ...] = tuple(_to_game_frame(f) for f in frames[-4:])
```

> **⚠️ Superseded by design decision #5, commit `e5bb04d`.** The block below
> is what was originally implemented and task-reviewed — it has two Critical
> bugs (no terminal-frame guard; replacement candidates never checked for
> their own suppression) found by the final whole-branch review and fixed
> post-hoc. **Do not implement this version.** Use the corrected block from
> design decision #5 instead (`not frame.is_game_over` guard; walk
> `_FALLBACK_PREFERENCE` checking `is_suppressed` on each candidate). Left
> here only as the historical record of what Step 3 originally said.

Replace the `decide()` call block:

```python
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
```

with:

```python
        decision, self._memory = decide(
            frame=frame,
            history=history,
            memory=self._memory,
            dead_signatures=self._dead_signatures,
            config=self._config,
            backend=self._backend,
            actions_taken=self._actions_taken,
        )

        # --- baseline-115-exact-state-memory (feat/baseline-115-exact-state-memory) ---
        if self._config.exact_state_suppression_on and self._exact_state_memory.is_suppressed(
            grid_hash(frame), action_signature(decision.action)
        ):
            alternative_names = [
                name
                for name in _FALLBACK_PREFERENCE
                if name in frame.legal_actions and name != decision.action.name
            ]
            if alternative_names:
                name = alternative_names[0]
                replacement = (
                    Action(name=name, x=32, y=32) if name == ActionName.ACTION6 else Action(name=name)
                )
                decision = replace(
                    decision,
                    action=replacement,
                    source="fallback_exact_state_suppressed",
                    target_object_label=None,
                )
        # --- end baseline-115-exact-state-memory ---

        self._actions_taken += 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_my_agent_exact_state.py tests/test_my_agent.py -q`
Expected: all tests PASS (3 new + existing `test_my_agent.py` tests unaffected)

- [ ] **Step 5: Commit**

```bash
git add agent/my_agent.py tests/test_my_agent_exact_state.py
git commit -m "feat(my-agent): wire exact-state suppression behind exact_state_suppression_on"
```

---

### Task 5: Full suite verification, handoff update, push

**Files:**
- Modify: `docs/HANDOFF.md` (Parallel work split table row for track 1)

- [ ] **Step 1: Run the full test suite**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: all tests PASS, total count = 136 (baseline) + 2 (`test_transitions.py`) + 13 (`test_exact_state_memory.py`) + 3 (`test_config.py`) + 3 (`test_my_agent_exact_state.py`) = 157

- [ ] **Step 2: Confirm off-by-default behavior with no env vars set**

Run: `.venv\Scripts\pytest.exe tests/ -q` (same command — no `ZERX_*` env vars are set in this shell by default; this step is the explicit acceptance check that the full suite, not just the new tests, is green with the feature inert)
Expected: same PASS count as Step 1

- [ ] **Step 3: Update `docs/HANDOFF.md`**

In `docs/HANDOFF.md`'s "Parallel work split" table (around line 139), replace:

```markdown
| 1 | `baseline-115-exact-state-memory` | `feat/baseline-115-exact-state-memory` | `docs/superpowers/plans/parallel-day3/person-1-baseline-115.md` |
```

with:

```markdown
| 1 | `baseline-115-exact-state-memory` — **done**, 157/157 tests green, off by default | `feat/baseline-115-exact-state-memory` | `docs/superpowers/plans/parallel-day3/person-1-baseline-115.md` |
```

- [ ] **Step 4: Commit the handoff update**

```bash
git add docs/HANDOFF.md
git commit -m "docs: mark baseline-115-exact-state-memory done in parallel work split table"
```

- [ ] **Step 5: Push to the track's own branch**

```bash
git push origin feat/baseline-115-exact-state-memory
```

Expected: push succeeds, no force-push, no merge to `master`.

---

## Self-review notes (completed during planning, not a task to execute)

- **Spec coverage:** record shape (Task 2) matches `person-1-baseline-115.md` verbatim; `Config` field placement (Task 3) matches `README.md`'s etiquette; `agent/my_agent.py` wiring (Task 4) covers both required hooks ("feed outcomes back" and "consult before accepting") from `person-1-baseline-115.md`; tests cover new-signature/no-suppression, ineffective-outcome suppression, different-action/different-state non-suppression, `reset()`, and `later_disconfirmed` (all required by `person-1-baseline-115.md`'s "Tests" section) plus the integration scenario proving flag-off is unchanged and flag-on suppresses.
- **Placeholder scan:** no TBD/TODO markers; every step has literal code.
- **Type consistency:** `ExactStateMemory.record_outcome`/`is_suppressed`/`record_for`/`reset` signatures used identically in Task 2's tests and Task 4's wiring; `action_signature` and `grid_hash` imported with the exact names Task 1/Task 2 produce.
