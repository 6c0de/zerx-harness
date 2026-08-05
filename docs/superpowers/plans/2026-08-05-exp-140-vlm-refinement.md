# exp-140-vlm-refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build off-by-default multi-candidate generation infrastructure (deterministic scoring, deterministic selection, an optional arbiter hook) so a real matched-budget ablation of candidate count can be run later, per `STRATEGY.md` §3.2/§7's `exp-140-vlm-refinement` entry.

**Architecture:** A new, self-contained module `zerx/candidates.py` calls `backend.generate()` N times, parses each response with the existing `zerx.policy.parse_action`, scores each deterministically, and picks one — either deterministically (default) or via an optional arbiter backend (config-gated, off by default). `zerx/policy.py`'s `decide()` gets exactly one new `if config.candidate_count > 1:` branch wrapping a completely untouched existing single-call path in the `else`. `zerx/config.py` gets one new field, `candidate_count: int = 1`, appended at the end of the field list.

**Tech Stack:** Python 3, pytest, existing `zerx/` package conventions (frozen dataclasses, `Protocol`-based `ModelBackend`, `FakeModelBackend` test doubles). No new dependencies.

## Global Constraints

- Feature ships OFF by default: `candidate_count: int = 1` (single-candidate, current behavior), `arbiter_on` already defaults to `False` and stays that way — running the full suite with no env vars set must produce byte-identical `decide()` behavior to before this change, for every existing test (`docs/superpowers/plans/parallel-day3/README.md`).
- Do not change `decide()`'s function signature (`docs/superpowers/plans/parallel-day3/README.md`).
- Add new `Config` fields at the end of the existing field list only (after `platform`), and the matching `from_env(...)` line at the end of that method's argument list — never interleave, never reorder, never touch an existing field (`docs/superpowers/plans/parallel-day3/README.md`).
- No changes to `scripts/build_notebook.py`, `scripts/build_colab_notebook.py`, anything Kaggle-related, or anything touching `CEREBRAS_API_KEY` (`docs/superpowers/plans/parallel-day3/README.md`).
- Frame descriptor and click-failure-radius are explicitly out of scope for this track (`STRATEGY.md` §3.2, `person-3-exp-140.md`).
- No multi-action plans/queues — `AGENTS.md`'s hard rule; every candidate is exactly one action.
- The existing 136-test suite must stay green; new tests are additive.
- Push only to `feat/exp-140-vlm-refinement` — never merge to `master`.
- Prefer approach (b) from `person-3-exp-140.md`: do **not** touch the shared `ModelBackend` Protocol or `FakeModelBackend` — the "call N times" loop lives entirely in the new `zerx/candidates.py` module, never inside `decide()` itself.

---

## Design notes (for the self-review / for whoever integrates this later)

**Why `select_best_candidate` picks the first candidate on a score tie:** Python's `max(iterable, key=...)` only replaces its running best when a later item's key is *strictly greater*, so the first item with the maximum score wins ties. This is documented in the function's docstring and locked in by a dedicated test.

**Why `static_candidate_score` collapses two of STRATEGY.md §3.2's listed factors:** `zerx.policy.parse_action` already enforces legal-action membership and `ACTION6` coordinate bounds (via `Action.__post_init__`) before it ever returns a `ParsedAction`. That means "parse validity" and "usable-action presence" are the same fact by the time a `Candidate` exists, and "click plausibility" is already guaranteed by the same validation — a separate check would just re-verify what parsing already proved. "Plan length" is dropped entirely: `AGENTS.md` forbids multi-action queues, so every candidate is exactly one action. What's left, and what the function actually scores: parse success (base 1.0, else 0.0), a small penalty for needing the deterministic repair path (weaker quality signal), and a larger penalty for `RESET` (decide() only reaches candidate generation after its own terminal check already returned early, so a `RESET` candidate at that point discards progress rather than solving).

**Why the `zerx.candidates` import inside `decide()` is local, not top-of-file:** `zerx/candidates.py` imports `ParsedAction`/`parse_action` from `zerx.policy` (per `person-3-exp-140.md`'s specified `Candidate.parsed` type). If `zerx/policy.py` also imported `zerx.candidates` at module load time, that would be a circular import and fail immediately on `import zerx.policy`. Importing `zerx.candidates` inside `decide()`'s function body (only reached when `config.candidate_count > 1`, i.e. never in the default/off path) defers the import until both modules have already finished loading, which breaks the cycle without restructuring either module.

---

### Task 1: `Config.candidate_count` field

**Files:**
- Modify: `zerx/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Config.candidate_count: int = 1` (positional/keyword field), validated `>= 1` in `__post_init__`; `Config.from_env` reads `ZERX_CANDIDATE_COUNT`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_default_candidate_count_is_one():
    assert Config().candidate_count == 1


def test_from_env_overrides_candidate_count():
    cfg = Config.from_env({"ZERX_CANDIDATE_COUNT": "3"})
    assert cfg.candidate_count == 3


def test_rejects_non_positive_candidate_count():
    with pytest.raises(ValueError):
        Config(candidate_count=0)
    with pytest.raises(ValueError):
        Config(candidate_count=-1)


def test_from_env_rejects_non_positive_candidate_count():
    with pytest.raises(ValueError):
        Config.from_env({"ZERX_CANDIDATE_COUNT": "0"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_config.py -v`
Expected: the 4 new tests FAIL with `TypeError: __init__() got an unexpected keyword argument 'candidate_count'` (or `AttributeError`).

- [ ] **Step 3: Add the field, validation, and from_env wiring**

In `zerx/config.py`, change the `Config` dataclass field list — add `candidate_count` as the last field, after `platform`:

```python
    platform: str = "local"  # "local" | "colab" | "kaggle"
    candidate_count: int = 1
```

In `__post_init__`, add the new check after the existing `budget_soft_cap` check:

```python
        if self.budget_soft_cap <= 0:
            raise ValueError("budget_soft_cap must be positive")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
```

In `from_env`, add the new line as the last argument, after `platform=...`:

```python
            platform=_env_str(env, "ZERX_PLATFORM", cls.platform),
            candidate_count=_env_int(env, "ZERX_CANDIDATE_COUNT", cls.candidate_count),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_config.py -v`
Expected: all tests PASS (existing + 4 new).

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: all pass, count increased by 4 from the pre-change baseline.

- [ ] **Step 6: Commit**

```bash
git add zerx/config.py tests/test_config.py
git commit -m "feat(config): add candidate_count field for exp-140 multi-candidate generation"
```

---

### Task 2: `zerx/candidates.py` — `Candidate` dataclass and `static_candidate_score`

**Files:**
- Create: `zerx/candidates.py`
- Test: `tests/test_candidates.py`

**Interfaces:**
- Consumes: `zerx.policy.ParsedAction`, `zerx.policy.parse_action` (existing); `zerx.types.ActionName` (existing).
- Produces: `Candidate` frozen dataclass (`raw_response: str`, `parsed: Optional[ParsedAction]`, `static_score: float`); `static_candidate_score(candidate_raw: str, parsed: Optional[ParsedAction]) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_candidates.py`:

```python
from zerx.candidates import Candidate, static_candidate_score
from zerx.policy import ParsedAction
from zerx.types import Action, ActionName


def test_static_candidate_score_zero_for_unparsed_candidate():
    assert static_candidate_score("garbage", None) == 0.0


def test_static_candidate_score_full_for_clean_non_reset_parse():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    assert static_candidate_score('{"action": "ACTION1"}', parsed) == 1.0


def test_static_candidate_score_penalizes_repaired_output():
    clean = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    repaired = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=True)
    assert static_candidate_score("x", repaired) < static_candidate_score("x", clean)


def test_static_candidate_score_penalizes_reset_action():
    reset = ParsedAction(action=Action(name=ActionName.RESET), repaired=False)
    non_reset = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    assert static_candidate_score("x", reset) < static_candidate_score("x", non_reset)


def test_static_candidate_score_never_negative():
    reset_repaired = ParsedAction(action=Action(name=ActionName.RESET), repaired=True)
    assert static_candidate_score("x", reset_repaired) >= 0.0


def test_candidate_is_a_frozen_dataclass_with_expected_fields():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    c = Candidate(raw_response="raw", parsed=parsed, static_score=1.0)
    assert c.raw_response == "raw"
    assert c.parsed is parsed
    assert c.static_score == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_candidates.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'zerx.candidates'`.

- [ ] **Step 3: Create `zerx/candidates.py`**

```python
"""Multi-candidate generation and deterministic scoring/selection --
exp-140-vlm-refinement infrastructure (STRATEGY.md 3.2). Off by default:
nothing here is called unless Config.candidate_count > 1 (see
zerx/policy.py's decide()). Never calls decide() itself and never touches
the environment -- pure candidate generation/scoring/selection over
backend.generate() responses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional

from zerx.config import Config
from zerx.model_backend import ModelBackend
from zerx.policy import ParsedAction, parse_action
from zerx.types import ActionName


@dataclass(frozen=True)
class Candidate:
    raw_response: str
    parsed: Optional[ParsedAction]
    static_score: float


def static_candidate_score(candidate_raw: str, parsed: Optional[ParsedAction]) -> float:
    """Deterministic scoring, STRATEGY.md 3.2's factors adapted to a
    single-action ParsedAction -- see
    docs/superpowers/plans/2026-08-05-exp-140-vlm-refinement.md's "Design
    notes" for the full rationale of which factors were kept, merged, or
    dropped.
    """
    if parsed is None:
        return 0.0
    score = 1.0
    if parsed.repaired:
        score -= 0.2
    if parsed.action.name == ActionName.RESET:
        score -= 0.5
    return max(0.0, score)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_candidates.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add zerx/candidates.py tests/test_candidates.py
git commit -m "feat(candidates): add Candidate type and deterministic static_candidate_score"
```

---

### Task 3: `generate_candidates`

**Files:**
- Modify: `zerx/candidates.py`
- Test: `tests/test_candidates.py`

**Interfaces:**
- Consumes: `zerx.model_backend.ModelBackend` (existing, `.generate(prompt: str) -> str`), `zerx.types.ActionName` (existing).
- Produces: `generate_candidates(backend: ModelBackend, prompt: str, legal_actions: FrozenSet[ActionName], count: int) -> List[Candidate]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_candidates.py`:

```python
from zerx.candidates import generate_candidates
from zerx.model_backend import FakeModelBackend

LEGAL = frozenset({ActionName.RESET, ActionName.ACTION1, ActionName.ACTION5, ActionName.ACTION6})


def test_generate_candidates_calls_backend_exactly_count_times():
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'] * 3)
    candidates = generate_candidates(backend, "prompt", LEGAL, count=3)
    assert backend.call_count == 3
    assert len(candidates) == 3


def test_generate_candidates_records_parse_failure_without_crashing():
    backend = FakeModelBackend(
        responses=['{"action": "ACTION1"}', "garbage", '{"action": "ACTION5"}']
    )
    candidates = generate_candidates(backend, "prompt", LEGAL, count=3)
    assert candidates[0].parsed is not None
    assert candidates[1].parsed is None
    assert candidates[1].static_score == 0.0
    assert candidates[2].parsed is not None


def test_generate_candidates_stores_raw_response_and_score():
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'])
    candidates = generate_candidates(backend, "prompt", LEGAL, count=1)
    assert candidates[0].raw_response == '{"action": "ACTION1"}'
    assert candidates[0].static_score == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_candidates.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_candidates'`.

- [ ] **Step 3: Add `generate_candidates` to `zerx/candidates.py`**

Append to `zerx/candidates.py`:

```python
def generate_candidates(
    backend: ModelBackend,
    prompt: str,
    legal_actions: FrozenSet[ActionName],
    count: int,
) -> List[Candidate]:
    """Calls backend.generate(prompt) `count` times, parses each response,
    scores each deterministically. Never calls an arbiter -- that's a
    separate, explicitly optional step (see select_candidate()). A
    response that fails to parse gets parsed=None and a 0.0 score but does
    not stop generation of the remaining candidates.
    """
    candidates: List[Candidate] = []
    for _ in range(count):
        raw = backend.generate(prompt)
        try:
            parsed = parse_action(raw, legal_actions)
        except Exception:
            parsed = None
        candidates.append(
            Candidate(
                raw_response=raw,
                parsed=parsed,
                static_score=static_candidate_score(raw, parsed),
            )
        )
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_candidates.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add zerx/candidates.py tests/test_candidates.py
git commit -m "feat(candidates): add generate_candidates to call a backend N times and score each response"
```

---

### Task 4: `select_best_candidate`

**Files:**
- Modify: `zerx/candidates.py`
- Test: `tests/test_candidates.py`

**Interfaces:**
- Produces: `select_best_candidate(candidates: List[Candidate]) -> Optional[Candidate]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_candidates.py`:

```python
from zerx.candidates import select_best_candidate


def test_select_best_candidate_picks_highest_score():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    low = Candidate(raw_response="a", parsed=None, static_score=0.0)
    high = Candidate(raw_response="b", parsed=parsed, static_score=1.0)
    assert select_best_candidate([low, high]) is high


def test_select_best_candidate_breaks_ties_by_earliest_candidate():
    first = Candidate(raw_response="a", parsed=None, static_score=0.5)
    second = Candidate(raw_response="b", parsed=None, static_score=0.5)
    assert select_best_candidate([first, second]) is first


def test_select_best_candidate_empty_list_returns_none():
    assert select_best_candidate([]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_candidates.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_best_candidate'`.

- [ ] **Step 3: Add `select_best_candidate` to `zerx/candidates.py`**

Append to `zerx/candidates.py`:

```python
def select_best_candidate(candidates: List[Candidate]) -> Optional[Candidate]:
    """Deterministic selection -- no LLM arbiter. This is what actually
    gets used when arbiter_on is False (the default, and the only
    supported mode for this track). Ties break toward the earliest
    candidate in generation order: Python's max() only replaces its
    running best on a strictly greater key, so the first item with the
    maximum score wins.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.static_score)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_candidates.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add zerx/candidates.py tests/test_candidates.py
git commit -m "feat(candidates): add select_best_candidate deterministic selection"
```

---

### Task 5: `select_candidate` arbiter hook

**Files:**
- Modify: `zerx/candidates.py`
- Test: `tests/test_candidates.py`

**Interfaces:**
- Consumes: `zerx.config.Config.arbiter_on` (existing, already defaults to `False`), `zerx.model_backend.ModelBackend`.
- Produces: `select_candidate(candidates: List[Candidate], config: Config, arbiter: Optional[ModelBackend] = None) -> Optional[Candidate]` — the entry point `decide()` will call in Task 6.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_candidates.py`:

```python
from zerx.candidates import select_candidate
from zerx.config import Config


def test_select_candidate_never_calls_arbiter_when_arbiter_on_false():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    candidates = [Candidate(raw_response="a", parsed=parsed, static_score=1.0)]
    arbiter = FakeModelBackend(responses=["0"])
    picked = select_candidate(candidates, Config(arbiter_on=False), arbiter=arbiter)
    assert picked is candidates[0]
    assert arbiter.call_count == 0


def test_select_candidate_never_calls_arbiter_when_none_provided():
    parsed = ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False)
    candidates = [Candidate(raw_response="a", parsed=parsed, static_score=1.0)]
    picked = select_candidate(candidates, Config(arbiter_on=True), arbiter=None)
    assert picked is candidates[0]


def test_select_candidate_consults_arbiter_when_arbiter_on_true():
    low = Candidate(
        raw_response="a",
        parsed=ParsedAction(action=Action(name=ActionName.ACTION5), repaired=False),
        static_score=1.0,
    )
    high = Candidate(
        raw_response="b",
        parsed=ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False),
        static_score=1.0,
    )
    arbiter = FakeModelBackend(responses=["1"])
    picked = select_candidate([low, high], Config(arbiter_on=True), arbiter=arbiter)
    assert picked is high
    assert arbiter.call_count == 1


def test_select_candidate_falls_back_to_deterministic_when_arbiter_output_invalid():
    only = Candidate(
        raw_response="a",
        parsed=ParsedAction(action=Action(name=ActionName.ACTION1), repaired=False),
        static_score=1.0,
    )
    arbiter = FakeModelBackend(responses=["not an int"])
    picked = select_candidate([only], Config(arbiter_on=True), arbiter=arbiter)
    assert picked is only


def test_select_candidate_empty_list_returns_none():
    assert select_candidate([], Config(), arbiter=None) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_candidates.py -v`
Expected: FAIL with `ImportError: cannot import name 'select_candidate'`.

- [ ] **Step 3: Add the arbiter hook to `zerx/candidates.py`**

Append to `zerx/candidates.py`:

```python
def _build_arbiter_prompt(candidates: List[Candidate]) -> str:
    lines = "\n".join(
        f"{i}: {c.raw_response!r} (static_score={c.static_score:.2f})"
        for i, c in enumerate(candidates)
    )
    return (
        "Multiple candidate actions were generated for the same game state. "
        "Pick the single best one.\n"
        f"{lines}\n\n"
        "Respond with exactly one integer: the index of the best candidate."
    )


def _select_with_arbiter(candidates: List[Candidate], arbiter: ModelBackend) -> Optional[Candidate]:
    valid = [c for c in candidates if c.parsed is not None]
    if not valid:
        return None
    try:
        raw = arbiter.generate(_build_arbiter_prompt(valid))
        index = int(raw.strip())
    except Exception:
        return None
    if 0 <= index < len(valid):
        return valid[index]
    return None


def select_candidate(
    candidates: List[Candidate],
    config: Config,
    arbiter: Optional[ModelBackend] = None,
) -> Optional[Candidate]:
    """Entry point decide() calls (Task 6). Deterministic by default
    (select_best_candidate). Only consults `arbiter` when
    config.arbiter_on is True AND an arbiter backend is actually supplied
    -- both conditions required, so passing an arbiter with the flag off
    is still a true no-op. STRATEGY.md 3.2: the arbiter is the
    lowest-priority, most speculative part of this track's scope, so on
    any arbiter failure (bad output, exception, no valid candidates) this
    falls back to the deterministic pick rather than raising.
    """
    if not candidates:
        return None
    if config.arbiter_on and arbiter is not None:
        picked = _select_with_arbiter(candidates, arbiter)
        if picked is not None:
            return picked
    return select_best_candidate(candidates)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_candidates.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add zerx/candidates.py tests/test_candidates.py
git commit -m "feat(candidates): add select_candidate arbiter hook, off by default"
```

---

### Task 6: Wire `candidate_count > 1` into `decide()`

**Files:**
- Modify: `zerx/policy.py`
- Test: `tests/test_policy_decide.py`

**Interfaces:**
- Consumes: `zerx.candidates.generate_candidates`, `zerx.candidates.select_candidate` (Tasks 3/5, imported locally inside `decide()` — see "Design notes" above for why).
- Produces: no new public interface — `decide()`'s existing signature and default (`candidate_count == 1`) behavior are unchanged; a new internal branch handles `candidate_count > 1`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_policy_decide.py`:

```python
def test_decide_multi_candidate_calls_backend_candidate_count_times():
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'] * 3)
    decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(candidate_count=3),
        backend=backend,
        actions_taken=0,
    )
    assert backend.call_count == 3


def test_decide_multi_candidate_uses_model_source_when_a_candidate_parses():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(candidate_count=2),
        backend=FakeModelBackend(
            responses=['{"action": "ACTION5"}', '{"action": "ACTION1"}']
        ),
        actions_taken=0,
    )
    assert decision.source == "model"
    assert decision.action.name in (ActionName.ACTION1, ActionName.ACTION5)


def test_decide_multi_candidate_falls_back_when_all_candidates_unparseable():
    decision, _ = decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(candidate_count=2),
        backend=FakeModelBackend(responses=["garbage", "also garbage"]),
        actions_taken=0,
    )
    assert decision.source == "fallback_deterministic"


def test_decide_multi_candidate_prefers_higher_scored_non_reset_candidate():
    legal = frozenset({ActionName.RESET, ActionName.ACTION1, ActionName.ACTION5})
    decision, _ = decide(
        frame=_frame([[0, 0], [0, 0]], legal=legal),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(candidate_count=2),
        backend=FakeModelBackend(
            responses=['{"action": "RESET"}', '{"action": "ACTION1"}']
        ),
        actions_taken=0,
    )
    assert decision.action.name == ActionName.ACTION1


def test_decide_default_candidate_count_still_calls_backend_exactly_once():
    """Regression guard: candidate_count's default (1) must take the
    original, untouched single-call path -- every other test in this file
    already exercises Config() with no candidate_count override, so this
    just makes the call-count invariant explicit.
    """
    backend = FakeModelBackend(responses=['{"action": "ACTION1"}'])
    decide(
        frame=_blank_frame(),
        history=(),
        memory=MemoryState(),
        dead_signatures=DeadSignatureTracker(),
        config=Config(),
        backend=backend,
        actions_taken=0,
    )
    assert backend.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\pytest.exe tests/test_policy_decide.py -v`
Expected: the 4 new `multi_candidate` tests FAIL (backend called once instead of N times / wrong source), `test_decide_default_candidate_count_still_calls_backend_exactly_once` PASSES already (no code change needed for it, it documents existing behavior).

- [ ] **Step 3: Add the branch to `decide()` in `zerx/policy.py`**

Replace this block in `decide()` (currently around line 214):

```python
    try:
        raw = backend.generate(build_prompt(perception, new_memory, candidates))
        parsed = parse_action(raw, legal_actions)
    except Exception:
        parsed = None
```

with:

```python
    if config.candidate_count > 1:
        try:
            from zerx.candidates import generate_candidates, select_candidate

            prompt = build_prompt(perception, new_memory, candidates)
            model_candidates = generate_candidates(
                backend, prompt, legal_actions, config.candidate_count
            )
            best = select_candidate(model_candidates, config)
            parsed = best.parsed if best is not None else None
        except Exception:
            parsed = None
    else:
        try:
            raw = backend.generate(build_prompt(perception, new_memory, candidates))
            parsed = parse_action(raw, legal_actions)
        except Exception:
            parsed = None
```

The `else` branch is byte-identical to the code it replaces — only a new `if` branch and one `else:` line were added.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\pytest.exe tests/test_policy_decide.py -v`
Expected: all PASS, including every pre-existing test in the file (default `Config()` behavior unchanged).

- [ ] **Step 5: Run the full suite to check for regressions**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add zerx/policy.py tests/test_policy_decide.py
git commit -m "feat(policy): route decide() through multi-candidate generation when candidate_count > 1"
```

---

### Task 7: Full suite verification, `docs/HANDOFF.md` status update, push

**Files:**
- Modify: `docs/HANDOFF.md` (the "Parallel work split" table, one line)

- [ ] **Step 1: Run the full test suite one final time**

Run: `.venv\Scripts\pytest.exe tests/ -q`
Expected: all pass. Record the exact pass count (baseline 136 + this track's new tests).

- [ ] **Step 2: Confirm the feature is truly off by default**

Run: `.venv\Scripts\pytest.exe tests/test_policy_decide.py tests/test_config.py tests/test_candidates.py -v`
Expected: all pass; specifically re-confirm every pre-existing `test_policy_decide.py` test (none of which set `candidate_count`) still passes unmodified.

- [ ] **Step 3: Update `docs/HANDOFF.md`'s "Parallel work split" table**

In `docs/HANDOFF.md`, find the table row:

```markdown
| 3 | `exp-140-vlm-refinement` (candidate/arbiter infra, off by default) | `feat/exp-140-vlm-refinement` | `docs/superpowers/plans/parallel-day3/person-3-exp-140.md` |
```

Add a status note after it (one line, do not rewrite the table):

```markdown

Track 3 status (2026-08-05): done. `zerx/candidates.py` (Candidate,
static_candidate_score, generate_candidates, select_best_candidate,
select_candidate w/ off-by-default arbiter hook) + `Config.candidate_count`
(default 1) + a new `decide()` branch gated on `candidate_count > 1`. Full
suite green, default behavior unchanged. See
`docs/superpowers/plans/2026-08-05-exp-140-vlm-refinement.md`.
```

- [ ] **Step 4: Commit the handoff update**

```bash
git add docs/HANDOFF.md
git commit -m "docs(handoff): record exp-140-vlm-refinement track status"
```

- [ ] **Step 5: Push to the track branch**

```bash
git push origin feat/exp-140-vlm-refinement
```

Expected: push succeeds, branch is `feat/exp-140-vlm-refinement`, no push to `master`.

---

## Self-review notes

- **Spec coverage:** multi-candidate generation (Task 3), deterministic static scorer (Task 2), config-gated arbiter hook (Task 5), `Config.candidate_count` field (Task 1), `decide()` wiring behind a new additive branch (Task 6), off-by-default regression guard (Tasks 6/7) — all of `person-3-exp-140.md`'s "Interfaces you're producing" and "Tests" sections are covered. Frame descriptor, click-failure-radius, confidence prompting, and multi-action queues are deliberately not built, per that file's "Explicitly out of scope."
- **Placeholder scan:** no TBD/TODO; every step has real code.
- **Type consistency:** `Candidate.static_score` (not `.score`) used consistently across Tasks 2–6, matching `person-3-exp-140.md`'s specified field name; `generate_candidates`/`select_best_candidate`/`select_candidate` signatures match their first-defined form in every later usage (Task 6's `decide()` call sites).

## Known follow-up (not built in this track)

Flagged by the final whole-branch review — real gaps, deliberately out of
this track's scope (infrastructure only, per
`docs/superpowers/plans/parallel-day3/README.md`'s "what done does NOT
require"), recorded here so a future ablation-prep track has a starting
point instead of rediscovering them:

- **No candidate-trace survival.** `decide()`'s multi-candidate branch keeps
  only the winning `Candidate.parsed`; the full `List[Candidate]` (every
  raw response, parse outcome, and score) is discarded once
  `select_candidate` returns. `STRATEGY.md` §3.2's own precondition for
  ever enabling the arbiter is "collect candidate traces" first — that
  collection doesn't exist yet. A future track should thread something
  like an inert-by-default `Decision.candidate_scores` tuple through to
  `agent/my_agent.py`'s `upstream.reasoning` logging, alongside the
  existing `source`/`repaired`/`config_hash` fields.
- **No sampling control.** `ModelBackend.generate(prompt: str) -> str` has
  no seed/temperature/max-tokens parameter. Requesting several candidates
  under "identical seed/temperature/token budgets" (this track's own
  ladder-entry promotion bar, `STRATEGY.md` §7) isn't possible yet, and a
  greedy-decoding server would make `generate_candidates` produce N
  byte-identical candidates -- pure latency cost, zero diversity. Adding
  this was correctly out of scope here (the brief mandated not touching
  the `ModelBackend` Protocol), but it's a real prerequisite for a real
  ablation run.
- **`arbiter_on=True` is a silent no-op through `decide()` today.**
  `zerx/policy.py`'s multi-candidate branch calls
  `select_candidate(model_candidates, config)` with no `arbiter` argument,
  so `zerx/candidates.py`'s arbiter guard (`config.arbiter_on and arbiter
  is not None`) never passes no matter what `Config.arbiter_on` is set to.
  Setting `ZERX_ARBITER_ON=true` with `ZERX_CANDIDATE_COUNT>1` in a real
  run produces zero arbiter calls and no warning. Wiring an actual arbiter
  `ModelBackend` into `decide()`'s call site is future-track scope, not
  this one (see `person-3-exp-140.md`: "the LLM-arbiter path can be a
  real, tested, config-gated hook... don't over invest in tuning it").
