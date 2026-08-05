# baseline-130-hypothesis (structured memory) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `zerx` a structured, machine-readable memory shape (confirmed rules / working hypotheses / rejected hypotheses / open questions / goal / plan / notable failures, per STRATEGY.md §2.4 and §3.1) as an off-by-default, additive sibling to the existing free-text `MemoryState`, with deterministic pure update functions that implement STRATEGY.md §7's "fewer repeated probes and belief reversals" promotion criterion.

**Architecture:** Add `ConfirmedRule`, `Hypothesis`, `StructuredMemoryState`, and a family of pure update functions (`record_hypothesis`, `confirm_hypothesis`, `contradict_hypothesis`, `add_open_question`, `set_current_goal`, `set_current_plan`, `record_notable_failure`) plus `render_for_prompt` and `maybe_refresh_structured` to the existing `zerx/memory.py` — appended after the untouched `MemoryState`/`Summarizer`/`maybe_refresh` code, never editing a line of it. Gate everything behind a new `Config.structured_memory_on` flag (default `False`). Wire it into `agent/my_agent.py` as its own delimited, config-gated block that runs alongside (never inside) the existing `decide()` call — `zerx/policy.py`'s `decide()` is not touched at all, since `baseline-125-phase-control` (STRATEGY.md §6 step 4) is the documented, later consumer of this structure, and not touching it removes all risk to the other 3 parallel Day-3 branches that edit `decide()`.

**Tech Stack:** Python 3.12, stdlib `dataclasses` (`dataclass`, `field`, `replace`), `pytest`. No new third-party dependency.

## Global Constraints

- Feature ships OFF by default: `Config.structured_memory_on: bool = False` — running the full suite with no env vars set must produce byte-identical `decide()`/`choose_action` behavior to before this change, for every existing test (`docs/superpowers/plans/parallel-day3/README.md`).
- The existing 136 tests must keep passing unmodified — confirm the count with `.venv/bin/pytest tests/ -q` (this machine has no `uv`; `AGENTS.md`'s documented `uv run pytest -q` doesn't apply here, same class of environment deviation `docs/HANDOFF.md` already records for Windows venvs).
- `zerx/memory.py`: append only. Do not change a single line of the existing `MemoryState` dataclass, `Summarizer` type alias, or `maybe_refresh` function.
- `zerx/config.py`: add the new field at the **end** of `Config`'s field list (after `platform`), and the matching `from_env` line at the **end** of that method's return-call argument list. Do not touch any existing field or add new `__post_init__` validation.
- `zerx/policy.py`'s `decide()`: **not modified in this plan at all** (see Architecture above for why) — no signature change, no new kwarg, no new branch.
- `agent/my_agent.py`: new code lives in `# --- baseline-130-hypothesis (feat/baseline-130-hypothesis-memory) --- ... # --- end baseline-130-hypothesis ---` delimited blocks, placed immediately after existing unmodified code, guarded by `if self._config.structured_memory_on:`, never inside an existing `if`/`try` block.
- No changes to `scripts/build_notebook.py`, `scripts/build_colab_notebook.py`, anything Kaggle-related, or anything touching `CEREBRAS_API_KEY`.
- Commit after every task (imperative message, explains why not just what, matching `git log` style already in this repo).
- Push only to `feat/baseline-130-hypothesis-memory` — never merge to `master`.

---

### Task 1: `StructuredMemoryState` data model

**Files:**
- Modify: `zerx/memory.py` (append after line 53, the end of the current file)
- Test: `tests/test_structured_memory.py` (new file)

**Interfaces:**
- Consumes: nothing new (stdlib `dataclasses` only).
- Produces: `ConfirmedRule(statement: str, evidence_count: int = 1)`, `Hypothesis(statement: str, supporting_evidence: int = 1, contradicting_evidence: int = 0)`, `StructuredMemoryState` (fields: `confirmed_rules: list[ConfirmedRule]`, `working_hypotheses: list[Hypothesis]`, `rejected_hypotheses: list[Hypothesis]`, `open_questions: list[str]`, `current_goal: str`, `current_plan: list[str]`, `notable_failures: list[str]`, `step_count: int`, `last_refreshed_step: int`), each list-valued field defaulting via `field(default_factory=list)`, `StructuredMemoryState.reset() -> None` (mutates in place, matches `MemoryState.reset()`'s exact contract).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_structured_memory.py`:

```python
from zerx.memory import ConfirmedRule, Hypothesis, StructuredMemoryState


def test_structured_memory_state_defaults_are_empty():
    state = StructuredMemoryState()
    assert state.confirmed_rules == []
    assert state.working_hypotheses == []
    assert state.rejected_hypotheses == []
    assert state.open_questions == []
    assert state.current_goal == ""
    assert state.current_plan == []
    assert state.notable_failures == []
    assert state.step_count == 0
    assert state.last_refreshed_step == 0


def test_structured_memory_state_reset_clears_every_field():
    state = StructuredMemoryState(
        confirmed_rules=[ConfirmedRule(statement="lights turn on click", evidence_count=3)],
        working_hypotheses=[Hypothesis(statement="key opens door")],
        rejected_hypotheses=[Hypothesis(statement="red tile is lava", contradicting_evidence=2)],
        open_questions=["what does ACTION3 do"],
        current_goal="reach the exit",
        current_plan=["click door", "move right"],
        notable_failures=["clicked wall 4 times, no effect"],
        step_count=12,
        last_refreshed_step=10,
    )
    state.reset()
    assert state.confirmed_rules == []
    assert state.working_hypotheses == []
    assert state.rejected_hypotheses == []
    assert state.open_questions == []
    assert state.current_goal == ""
    assert state.current_plan == []
    assert state.notable_failures == []
    assert state.step_count == 0
    assert state.last_refreshed_step == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: FAIL with `ImportError: cannot import name 'ConfirmedRule' from 'zerx.memory'`

- [ ] **Step 3: Append the data model to `zerx/memory.py`**

First, change the existing import line (line 9 of `zerx/memory.py`, currently `from dataclasses import dataclass`) to:

```python
from dataclasses import dataclass, field
```

Then append this block at the end of `zerx/memory.py` (after the existing `maybe_refresh` function, which must stay unmodified above it):

```python


@dataclass(frozen=True)
class ConfirmedRule:
    statement: str
    evidence_count: int = 1


@dataclass(frozen=True)
class Hypothesis:
    statement: str
    supporting_evidence: int = 1
    contradicting_evidence: int = 0


@dataclass
class StructuredMemoryState:
    """STRATEGY.md §2.4/§3.1's structured memory schema: distinguishes
    confirmed rules, working hypotheses, and rejected hypotheses instead of
    storing every model statement as one undifferentiated fact, to reduce
    self-reinforcing hallucination in reflection memory. Off by default
    (`Config.structured_memory_on`); `zerx/memory.py`'s existing `MemoryState`
    is untouched and remains the baseline free-text memory.
    """

    confirmed_rules: list = field(default_factory=list)
    working_hypotheses: list = field(default_factory=list)
    rejected_hypotheses: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    current_goal: str = ""
    current_plan: list = field(default_factory=list)
    notable_failures: list = field(default_factory=list)
    step_count: int = 0
    last_refreshed_step: int = 0

    def reset(self) -> None:
        """Clear memory between games -- same guarantee as MemoryState.reset(),
        at every field, not just the top-level object.
        """
        self.confirmed_rules = []
        self.working_hypotheses = []
        self.rejected_hypotheses = []
        self.open_questions = []
        self.current_goal = ""
        self.current_plan = []
        self.notable_failures = []
        self.step_count = 0
        self.last_refreshed_step = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv/bin/pytest tests/ -q`
Expected: 138 passed (136 existing + 2 new)

- [ ] **Step 6: Commit**

```bash
git add zerx/memory.py tests/test_structured_memory.py
git commit -m "feat(memory): add StructuredMemoryState data model (baseline-130)

STRATEGY.md §2.4/§3.1's structured schema (confirmed_rules,
working_hypotheses, rejected_hypotheses, open_questions, current_goal,
current_plan, notable_failures) as a new, additive type alongside the
existing free-text MemoryState -- appended to zerx/memory.py without
touching a line of the existing MemoryState/maybe_refresh code."
```

---

### Task 2: Hypothesis/rule pure update functions

**Files:**
- Modify: `zerx/memory.py` (append after Task 1's code)
- Test: `tests/test_structured_memory.py` (append)

**Interfaces:**
- Consumes: `ConfirmedRule`, `Hypothesis`, `StructuredMemoryState` from Task 1.
- Produces: `record_hypothesis(state, statement) -> StructuredMemoryState`, `confirm_hypothesis(state, statement) -> StructuredMemoryState`, `contradict_hypothesis(state, statement) -> StructuredMemoryState`. All pure (never mutate `state`), all return a new `StructuredMemoryState` (only `confirmed_rules`/`working_hypotheses`/`rejected_hypotheses` differ from input; `step_count`/`last_refreshed_step` untouched by these three).

**Design decision (documented per `person-2-baseline-130.md`'s explicit request):** the "contradiction/probe-check" mechanism STRATEGY.md §7 requires ("fewer repeated probes and belief reversals") is a deterministic evidence-count crossover: each `contradict_hypothesis` call increments that hypothesis's `contradicting_evidence`; the moment `contradicting_evidence` reaches or exceeds `supporting_evidence` (`>=`, verified against the test suite below -- a hypothesis recorded once, at `supporting_evidence=1`, reverses on its first contradiction), the hypothesis moves from `working_hypotheses` to `rejected_hypotheses` -- this crossing IS a belief reversal, and it's directly countable across a run (the promotion metric) without needing a model call or a separate probe-scheduling subsystem. This is deliberately the shallow end of what STRATEGY.md allows ("decide how deep to go here, document the choice") -- no probe-scheduling, no automatic re-confirmation, no numeric confidence score; `confirm_hypothesis` is a separate, explicit action so a hypothesis is never silently auto-promoted to a confirmed rule by evidence count alone (STRATEGY.md's stated worry is over-confident auto-promotion from repeated restatement).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_structured_memory.py`:

```python
from zerx.memory import (
    confirm_hypothesis,
    contradict_hypothesis,
    record_hypothesis,
)


def test_record_hypothesis_adds_new_working_hypothesis():
    state = StructuredMemoryState()
    new_state = record_hypothesis(state, "clicking the blue tile opens the door")
    assert len(new_state.working_hypotheses) == 1
    assert new_state.working_hypotheses[0].statement == "clicking the blue tile opens the door"
    assert new_state.working_hypotheses[0].supporting_evidence == 1
    assert new_state.working_hypotheses[0].contradicting_evidence == 0
    # other lists untouched
    assert new_state.confirmed_rules == []
    assert new_state.rejected_hypotheses == []


def test_record_hypothesis_does_not_mutate_input():
    state = StructuredMemoryState()
    record_hypothesis(state, "clicking the blue tile opens the door")
    assert state.working_hypotheses == []


def test_record_hypothesis_repeated_statement_increments_support_not_duplicate():
    state = StructuredMemoryState()
    state = record_hypothesis(state, "same object")
    state = record_hypothesis(state, "same object")
    assert len(state.working_hypotheses) == 1
    assert state.working_hypotheses[0].supporting_evidence == 2


def test_confirm_hypothesis_moves_matching_working_hypothesis_to_confirmed_rules():
    state = StructuredMemoryState(working_hypotheses=[Hypothesis(statement="key opens door", supporting_evidence=3)])
    new_state = confirm_hypothesis(state, "key opens door")
    assert new_state.working_hypotheses == []
    assert len(new_state.confirmed_rules) == 1
    assert new_state.confirmed_rules[0].statement == "key opens door"
    assert new_state.confirmed_rules[0].evidence_count == 3


def test_confirm_hypothesis_with_no_matching_working_hypothesis_confirms_directly():
    state = StructuredMemoryState()
    new_state = confirm_hypothesis(state, "reset always returns to level 1")
    assert new_state.working_hypotheses == []
    assert len(new_state.confirmed_rules) == 1
    assert new_state.confirmed_rules[0].statement == "reset always returns to level 1"
    assert new_state.confirmed_rules[0].evidence_count == 1


def test_confirm_hypothesis_does_not_mutate_input():
    state = StructuredMemoryState(working_hypotheses=[Hypothesis(statement="key opens door")])
    confirm_hypothesis(state, "key opens door")
    assert len(state.working_hypotheses) == 1
    assert state.confirmed_rules == []


def test_contradict_hypothesis_increments_contradicting_evidence_below_threshold():
    state = StructuredMemoryState(
        working_hypotheses=[Hypothesis(statement="green tile is safe", supporting_evidence=2, contradicting_evidence=0)]
    )
    new_state = contradict_hypothesis(state, "green tile is safe")
    assert len(new_state.working_hypotheses) == 1
    assert new_state.working_hypotheses[0].contradicting_evidence == 1
    assert new_state.rejected_hypotheses == []


def test_contradict_hypothesis_crossing_threshold_is_a_belief_reversal():
    state = StructuredMemoryState(
        working_hypotheses=[Hypothesis(statement="green tile is safe", supporting_evidence=1, contradicting_evidence=0)]
    )
    new_state = contradict_hypothesis(state, "green tile is safe")
    assert new_state.working_hypotheses == []
    assert len(new_state.rejected_hypotheses) == 1
    assert new_state.rejected_hypotheses[0].statement == "green tile is safe"
    assert new_state.rejected_hypotheses[0].contradicting_evidence == 1


def test_contradict_hypothesis_with_no_matching_hypothesis_is_a_no_op():
    state = StructuredMemoryState()
    new_state = contradict_hypothesis(state, "never asserted")
    assert new_state.working_hypotheses == []
    assert new_state.rejected_hypotheses == []


def test_contradict_hypothesis_does_not_mutate_input():
    state = StructuredMemoryState(
        working_hypotheses=[Hypothesis(statement="green tile is safe", supporting_evidence=1)]
    )
    contradict_hypothesis(state, "green tile is safe")
    assert state.working_hypotheses[0].contradicting_evidence == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: FAIL with `ImportError: cannot import name 'record_hypothesis' from 'zerx.memory'`

- [ ] **Step 3: Append the update functions to `zerx/memory.py`**

```python


def record_hypothesis(state: StructuredMemoryState, statement: str) -> StructuredMemoryState:
    """Add a new working hypothesis, or -- if this exact statement is
    already tracked -- increment its supporting evidence instead of
    duplicating it. Never mutates `state`.
    """
    existing = [h for h in state.working_hypotheses if h.statement == statement]
    if existing:
        updated = Hypothesis(
            statement=statement,
            supporting_evidence=existing[0].supporting_evidence + 1,
            contradicting_evidence=existing[0].contradicting_evidence,
        )
        new_working = [updated if h.statement == statement else h for h in state.working_hypotheses]
    else:
        new_working = list(state.working_hypotheses) + [Hypothesis(statement=statement)]
    return replace(state, working_hypotheses=new_working)


def confirm_hypothesis(state: StructuredMemoryState, statement: str) -> StructuredMemoryState:
    """Move a working hypothesis to confirmed_rules (carrying its evidence
    count forward), or -- if it was never tracked as a hypothesis -- confirm
    it directly with evidence_count=1. Deduplicates against an already
    confirmed rule with the same statement by bumping its evidence_count.
    Never mutates `state`.
    """
    matching = [h for h in state.working_hypotheses if h.statement == statement]
    evidence_count = matching[0].supporting_evidence if matching else 1
    new_working = [h for h in state.working_hypotheses if h.statement != statement]

    already_confirmed = [r for r in state.confirmed_rules if r.statement == statement]
    if already_confirmed:
        bumped = ConfirmedRule(statement=statement, evidence_count=already_confirmed[0].evidence_count + evidence_count)
        new_confirmed = [bumped if r.statement == statement else r for r in state.confirmed_rules]
    else:
        new_confirmed = list(state.confirmed_rules) + [ConfirmedRule(statement=statement, evidence_count=evidence_count)]

    return replace(state, working_hypotheses=new_working, confirmed_rules=new_confirmed)


def contradict_hypothesis(state: StructuredMemoryState, statement: str) -> StructuredMemoryState:
    """Increment a working hypothesis's contradicting evidence; the moment
    contradicting_evidence exceeds supporting_evidence, this is a belief
    reversal -- move it from working_hypotheses to rejected_hypotheses
    (STRATEGY.md §7's promotion metric). A statement not currently tracked
    as a working hypothesis is a no-op. Never mutates `state`.
    """
    matching = [h for h in state.working_hypotheses if h.statement == statement]
    if not matching:
        return state

    current = matching[0]
    updated = Hypothesis(
        statement=statement,
        supporting_evidence=current.supporting_evidence,
        contradicting_evidence=current.contradicting_evidence + 1,
    )
    new_working = [h for h in state.working_hypotheses if h.statement != statement]

    if updated.contradicting_evidence >= updated.supporting_evidence:
        new_rejected = list(state.rejected_hypotheses) + [updated]
        return replace(state, working_hypotheses=new_working, rejected_hypotheses=new_rejected)

    new_working.append(updated)
    return replace(state, working_hypotheses=new_working)
```

Add `replace` to the existing `dataclasses` import line at the top of `zerx/memory.py` (from Task 1's `from dataclasses import dataclass, field`, change to):

```python
from dataclasses import dataclass, field, replace
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: PASS (12 passed)

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv/bin/pytest tests/ -q`
Expected: 148 passed (136 existing + 12 new)

- [ ] **Step 6: Commit**

```bash
git add zerx/memory.py tests/test_structured_memory.py
git commit -m "feat(memory): add hypothesis record/confirm/contradict functions (baseline-130)

Deterministic evidence-count crossover implements STRATEGY.md §7's
'belief reversal' promotion metric: a hypothesis moves from
working_hypotheses to rejected_hypotheses the moment contradicting
evidence exceeds supporting evidence. All three functions are pure --
never mutate their input StructuredMemoryState."
```

---

### Task 3: Remaining schema field helpers

**Files:**
- Modify: `zerx/memory.py` (append after Task 2's code)
- Test: `tests/test_structured_memory.py` (append)

**Interfaces:**
- Consumes: `StructuredMemoryState`, `replace` from Task 1/2.
- Produces: `add_open_question(state, question) -> StructuredMemoryState`, `set_current_goal(state, goal) -> StructuredMemoryState`, `set_current_plan(state, plan) -> StructuredMemoryState`, `record_notable_failure(state, failure) -> StructuredMemoryState`. All pure.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_structured_memory.py`:

```python
from zerx.memory import (
    add_open_question,
    record_notable_failure,
    set_current_goal,
    set_current_plan,
)


def test_add_open_question_appends_and_dedupes():
    state = StructuredMemoryState()
    state = add_open_question(state, "what does ACTION3 do")
    state = add_open_question(state, "what does ACTION3 do")
    state = add_open_question(state, "is there a timer")
    assert state.open_questions == ["what does ACTION3 do", "is there a timer"]


def test_set_current_goal_replaces_value():
    state = StructuredMemoryState(current_goal="old goal")
    new_state = set_current_goal(state, "reach the exit")
    assert new_state.current_goal == "reach the exit"
    assert state.current_goal == "old goal"  # input not mutated


def test_set_current_plan_replaces_list():
    state = StructuredMemoryState(current_plan=["old step"])
    new_state = set_current_plan(state, ["click door", "move right"])
    assert new_state.current_plan == ["click door", "move right"]
    assert state.current_plan == ["old step"]  # input not mutated


def test_record_notable_failure_appends_without_deduping():
    state = StructuredMemoryState()
    state = record_notable_failure(state, "clicked wall, no effect")
    state = record_notable_failure(state, "clicked wall, no effect")
    assert state.notable_failures == ["clicked wall, no effect", "clicked wall, no effect"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: FAIL with `ImportError: cannot import name 'add_open_question' from 'zerx.memory'`

- [ ] **Step 3: Append the field helpers to `zerx/memory.py`**

```python


def add_open_question(state: StructuredMemoryState, question: str) -> StructuredMemoryState:
    """Append an open question, deduped by exact text (repeatedly asking
    the same open question should not bloat the rendered prompt). Never
    mutates `state`.
    """
    if question in state.open_questions:
        return replace(state)
    return replace(state, open_questions=list(state.open_questions) + [question])


def set_current_goal(state: StructuredMemoryState, goal: str) -> StructuredMemoryState:
    """Replace the current goal. Never mutates `state`."""
    return replace(state, current_goal=goal)


def set_current_plan(state: StructuredMemoryState, plan) -> StructuredMemoryState:
    """Replace the current plan. Never mutates `state`."""
    return replace(state, current_plan=list(plan))


def record_notable_failure(state: StructuredMemoryState, failure: str) -> StructuredMemoryState:
    """Append a notable failure. Not deduped -- a repeated identical
    failure is itself meaningful signal (STRATEGY.md §3.1's ineffective-
    action evidence uses repetition the same way). Never mutates `state`.
    """
    return replace(state, notable_failures=list(state.notable_failures) + [failure])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: PASS (16 passed)

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv/bin/pytest tests/ -q`
Expected: 152 passed (136 existing + 16 new)

- [ ] **Step 6: Commit**

```bash
git add zerx/memory.py tests/test_structured_memory.py
git commit -m "feat(memory): add open_question/goal/plan/failure helpers (baseline-130)

Completes STRATEGY.md §3.1's structured schema -- every field now has a
pure, tested way to be populated, not just the hypothesis/rule lists."
```

---

### Task 4: `render_for_prompt`

**Files:**
- Modify: `zerx/memory.py` (append after Task 3's code)
- Test: `tests/test_structured_memory.py` (append)

**Interfaces:**
- Consumes: `StructuredMemoryState` from Task 1.
- Produces: `render_for_prompt(state: StructuredMemoryState) -> str`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_structured_memory.py`:

```python
from zerx.memory import render_for_prompt


def test_render_for_prompt_on_empty_state_uses_placeholders():
    text = render_for_prompt(StructuredMemoryState())
    assert "(none set)" in text  # goal
    assert "(none)" in text  # plan
    assert "(none yet)" in text  # confirmed rules / hypotheses / questions / failures


def test_render_for_prompt_includes_populated_fields():
    state = StructuredMemoryState(
        confirmed_rules=[ConfirmedRule(statement="key opens door", evidence_count=3)],
        working_hypotheses=[Hypothesis(statement="green tile is safe", supporting_evidence=2, contradicting_evidence=1)],
        rejected_hypotheses=[Hypothesis(statement="red tile is safe", supporting_evidence=1, contradicting_evidence=2)],
        open_questions=["what does ACTION3 do"],
        current_goal="reach the exit",
        current_plan=["click door", "move right"],
        notable_failures=["clicked wall, no effect"],
    )
    text = render_for_prompt(state)
    assert "key opens door" in text and "3" in text
    assert "green tile is safe" in text and "support=2" in text and "contradict=1" in text
    assert "red tile is safe" in text
    assert "what does ACTION3 do" in text
    assert "reach the exit" in text
    assert "click door" in text and "move right" in text
    assert "clicked wall, no effect" in text


def test_render_for_prompt_is_pure_and_deterministic():
    state = StructuredMemoryState(current_goal="reach the exit")
    assert render_for_prompt(state) == render_for_prompt(state)
    assert state.current_goal == "reach the exit"  # not mutated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: FAIL with `ImportError: cannot import name 'render_for_prompt' from 'zerx.memory'`

- [ ] **Step 3: Append `render_for_prompt` to `zerx/memory.py`**

```python


def render_for_prompt(state: StructuredMemoryState) -> str:
    """STRATEGY.md §3.1: 'The rendered prompt may include a compact
    textual form; the stored source of truth stays machine-readable.'
    This is that compact textual form -- a pure function, never the
    source of truth itself.
    """
    confirmed = (
        "\n".join(f"- {r.statement} (evidence={r.evidence_count})" for r in state.confirmed_rules)
        or "(none yet)"
    )
    working = (
        "\n".join(
            f"- {h.statement} (support={h.supporting_evidence}, contradict={h.contradicting_evidence})"
            for h in state.working_hypotheses
        )
        or "(none yet)"
    )
    rejected = (
        "\n".join(f"- {h.statement}" for h in state.rejected_hypotheses) or "(none yet)"
    )
    questions = "\n".join(f"- {q}" for q in state.open_questions) or "(none yet)"
    plan = "; ".join(state.current_plan) or "(none)"
    failures = "\n".join(f"- {f}" for f in state.notable_failures) or "(none yet)"

    return (
        f"Current goal: {state.current_goal or '(none set)'}\n"
        f"Current plan: {plan}\n"
        f"Confirmed rules:\n{confirmed}\n"
        f"Working hypotheses:\n{working}\n"
        f"Rejected hypotheses:\n{rejected}\n"
        f"Open questions:\n{questions}\n"
        f"Notable failures:\n{failures}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: PASS (19 passed)

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv/bin/pytest tests/ -q`
Expected: 155 passed (136 existing + 19 new)

- [ ] **Step 6: Commit**

```bash
git add zerx/memory.py tests/test_structured_memory.py
git commit -m "feat(memory): add render_for_prompt for structured memory (baseline-130)

Compact textual form for the prompt; StructuredMemoryState itself stays
the machine-readable source of truth, per STRATEGY.md §3.1."
```

---

### Task 5: `maybe_refresh_structured`

**Files:**
- Modify: `zerx/memory.py` (append after Task 4's code)
- Test: `tests/test_structured_memory.py` (append)

**Interfaces:**
- Consumes: `StructuredMemoryState`, `replace` from earlier tasks.
- Produces: `StructuredSummarizer = Callable[[StructuredMemoryState, str], StructuredMemoryState]`, `maybe_refresh_structured(state, recent_context, summarizer, refresh_interval) -> StructuredMemoryState`.

**Design decision:** unlike `Summarizer` (`str, str -> str`), `StructuredSummarizer` takes and returns the **full** `StructuredMemoryState`, not a single string -- a structured refresh may need to touch several fields at once (add a hypothesis, confirm another, update the goal) in one pass, which a single-string signature can't express. The deterministic no-op default is `lambda prev, ctx: prev`, mirroring `zerx/policy.py`'s existing `decide()` no-op summarizer pattern exactly.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_structured_memory.py`:

```python
from zerx.memory import maybe_refresh_structured


def test_maybe_refresh_structured_not_due_keeps_state_and_skips_summarizer():
    state = StructuredMemoryState(current_goal="old goal", step_count=0, last_refreshed_step=0)

    def boom(prev, ctx):
        raise AssertionError("summarizer should not be called")

    new_state = maybe_refresh_structured(state, "context", boom, refresh_interval=10)
    assert new_state.current_goal == "old goal"
    assert new_state.step_count == 1
    assert new_state.last_refreshed_step == 0


def test_maybe_refresh_structured_due_calls_summarizer_and_updates():
    state = StructuredMemoryState(current_goal="old goal", step_count=8, last_refreshed_step=0)

    def summarizer(prev, ctx):
        return set_current_goal(prev, f"{prev.current_goal}+{ctx}")

    new_state = maybe_refresh_structured(state, "context", summarizer, refresh_interval=9)
    assert new_state.step_count == 9
    assert new_state.last_refreshed_step == 9
    assert new_state.current_goal == "old goal+context"


def test_maybe_refresh_structured_does_not_mutate_input():
    state = StructuredMemoryState(current_goal="old goal", step_count=0, last_refreshed_step=0)
    maybe_refresh_structured(state, "context", lambda prev, ctx: set_current_goal(prev, "new goal"), refresh_interval=1)
    assert state.current_goal == "old goal"
    assert state.step_count == 0
```

Add `set_current_goal` to the existing `from zerx.memory import (...)` block used by Task 3's tests (it's already imported there), so this new block only needs `maybe_refresh_structured` added to that same import.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: FAIL with `ImportError: cannot import name 'maybe_refresh_structured' from 'zerx.memory'`

- [ ] **Step 3: Append `StructuredSummarizer`/`maybe_refresh_structured` to `zerx/memory.py`**

```python


StructuredSummarizer = Callable[[StructuredMemoryState, str], StructuredMemoryState]
# (previous_state, recent_context) -> new_state -- takes/returns the full
# structured state, unlike Summarizer's str-in/str-out, because a
# structured refresh may revise several fields in one pass.


def maybe_refresh_structured(
    state: StructuredMemoryState,
    recent_context: str,
    summarizer: StructuredSummarizer,
    refresh_interval: int,
) -> StructuredMemoryState:
    """Same due/not-due/never-mutate contract as maybe_refresh, adapted to
    StructuredSummarizer's full-state-in/full-state-out shape.
    """
    new_step_count = state.step_count + 1
    due = (new_step_count - state.last_refreshed_step) >= refresh_interval
    if not due:
        return replace(state, step_count=new_step_count)
    updated = summarizer(state, recent_context)
    return replace(updated, step_count=new_step_count, last_refreshed_step=new_step_count)
```

`Callable` is already imported at the top of `zerx/memory.py` (`from typing import Callable`) -- no new import needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_structured_memory.py -v`
Expected: PASS (22 passed)

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv/bin/pytest tests/ -q`
Expected: 158 passed (136 existing + 22 new)

- [ ] **Step 6: Commit**

```bash
git add zerx/memory.py tests/test_structured_memory.py
git commit -m "feat(memory): add maybe_refresh_structured refresh cadence (baseline-130)

Same due/not-due/never-mutate contract as the existing maybe_refresh,
adapted to a full-state-in/full-state-out StructuredSummarizer so one
refresh pass can revise several fields at once."
```

---

### Task 6: `Config.structured_memory_on` flag

**Files:**
- Modify: `zerx/config.py:44-77`
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Consumes: existing `Config` dataclass, `_env_bool`.
- Produces: `Config.structured_memory_on: bool` (default `False`), `Config.from_env(...)` honors `ZERX_STRUCTURED_MEMORY_ON`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config.py`:

```python
def test_structured_memory_on_defaults_false():
    assert Config().structured_memory_on is False


def test_from_env_missing_structured_memory_on_uses_default():
    cfg = Config.from_env({})
    assert cfg.structured_memory_on is False


def test_from_env_enables_structured_memory_on():
    cfg = Config.from_env({"ZERX_STRUCTURED_MEMORY_ON": "true"})
    assert cfg.structured_memory_on is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: FAIL with `AttributeError: 'Config' object has no attribute 'structured_memory_on'`

- [ ] **Step 3: Add the field to `zerx/config.py`**

Change line 46 (currently the last field, `platform: str = "local"  # "local" | "colab" | "kaggle"`) to add the new field immediately after it:

```python
    platform: str = "local"  # "local" | "colab" | "kaggle"
    structured_memory_on: bool = False
```

Change the end of `from_env` (currently ending at line 76 `platform=_env_str(env, "ZERX_PLATFORM", cls.platform),` followed by the closing `)` on line 77) to add the matching line immediately after it:

```python
            platform=_env_str(env, "ZERX_PLATFORM", cls.platform),
            structured_memory_on=_env_bool(
                env, "ZERX_STRUCTURED_MEMORY_ON", cls.structured_memory_on
            ),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config.py -v`
Expected: PASS (all `test_config.py` tests, including the 3 new ones)

- [ ] **Step 5: Run the full suite to confirm no regression**

Run: `.venv/bin/pytest tests/ -q`
Expected: 161 passed (158 from Task 5 + 3 new)

- [ ] **Step 6: Commit**

```bash
git add zerx/config.py tests/test_config.py
git commit -m "feat(config): add structured_memory_on flag, default off (baseline-130)

Added at the end of Config's field list and from_env's argument list per
docs/superpowers/plans/parallel-day3/README.md's merge etiquette --
default False preserves current behavior for every existing caller."
```

---

### Task 7: Wire `StructuredMemoryState` into `agent/my_agent.py`

**Files:**
- Modify: `agent/my_agent.py:29-33` (imports), `agent/my_agent.py:134-143` (`__init__`), `agent/my_agent.py:170-180` (`_choose_action_inner`, right after the existing `decide()` call block)
- Test: `tests/test_my_agent.py` (append)

**Interfaces:**
- Consumes: `StructuredMemoryState`, `maybe_refresh_structured` from `zerx.memory` (Tasks 1/5); `Config.structured_memory_on` from Task 6; `perceive` (already imported in `agent/my_agent.py`).
- Produces: `MyAgent._structured_memory: StructuredMemoryState` attribute, advanced once per `choose_action` call only when `config.structured_memory_on` is `True`; otherwise never touched (stays at its `__init__`-time default forever, confirming the flag is a true no-op).

**Design decision:** reuses the existing `self._config.memory_refresh_interval` field for cadence rather than adding a second, parallel refresh-interval config field -- one cadence knob is enough at this stage (STRATEGY.md §2.1: don't add machinery before it's needed), and both `MemoryState` and `StructuredMemoryState` refreshing on the same interval is the simplest coherent default. `decide()` itself is not touched or called differently -- this block runs immediately after it, using the already-computed `frame`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_my_agent.py`:

```python
def test_structured_memory_off_by_default_is_a_true_no_op():
    """With no env vars set (structured_memory_on defaults False), the
    structured memory attribute must never advance across several
    choose_action calls -- confirms the flag is a real no-op, not just an
    unused field.
    """
    agent = _make_agent()
    assert agent._structured_memory.step_count == 0

    frame = FrameData(
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )
    agent.choose_action([frame], frame)
    agent.choose_action([frame], frame)

    assert agent._structured_memory.step_count == 0


def test_structured_memory_on_advances_step_count(monkeypatch):
    monkeypatch.setenv("ZERX_STRUCTURED_MEMORY_ON", "true")
    agent = _make_agent()
    assert agent._structured_memory.step_count == 0

    frame = FrameData(
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 2, 5],
    )
    agent.choose_action([frame], frame)

    assert agent._structured_memory.step_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_my_agent.py -v -k structured_memory`
Expected: FAIL with `AttributeError: 'MyAgent' object has no attribute '_structured_memory'`

- [ ] **Step 3: Update the import line in `agent/my_agent.py`**

Change line 29 (currently `from zerx.memory import MemoryState`) to:

```python
from zerx.memory import MemoryState, StructuredMemoryState, maybe_refresh_structured
```

- [ ] **Step 4: Add the delimited construction block to `__init__`**

Immediately after line 143 (`self._pending_before_frame: Optional[GameFrame] = None`), add:

```python
        # --- baseline-130-hypothesis (feat/baseline-130-hypothesis-memory) ---
        self._structured_memory = StructuredMemoryState()
        # --- end baseline-130-hypothesis ---
```

- [ ] **Step 5: Add the delimited refresh block to `_choose_action_inner`**

Immediately after the existing line `self._actions_taken += 1` (currently line 180, right after the `decide()` call and before `self._transitions.begin(frame, decision.action)`), add:

```python
        # --- baseline-130-hypothesis (feat/baseline-130-hypothesis-memory) ---
        if self._config.structured_memory_on:
            self._structured_memory = maybe_refresh_structured(
                self._structured_memory,
                recent_context=perceive(frame).ascii_grid,
                summarizer=lambda prev, ctx: prev,
                refresh_interval=self._config.memory_refresh_interval,
            )
        # --- end baseline-130-hypothesis ---
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_my_agent.py -v`
Expected: PASS (all `test_my_agent.py` tests, including the 2 new ones)

- [ ] **Step 7: Run the full suite to confirm no regression**

Run: `.venv/bin/pytest tests/ -q`
Expected: 163 passed (161 from Task 6 + 2 new)

- [ ] **Step 8: Commit**

```bash
git add agent/my_agent.py tests/test_my_agent.py
git commit -m "feat(agent): wire StructuredMemoryState behind structured_memory_on (baseline-130)

Delimited, config-gated block per docs/superpowers/plans/parallel-day3/
README.md's etiquette -- runs alongside decide(), never inside it;
decide()'s signature and behavior are completely untouched. Off by
default: verified the structured memory attribute never advances when
the flag is unset."
```

---

## Self-Review Notes (for the plan author, already applied above)

- **Spec coverage:** person-2-baseline-130.md's schema (7 fields) -> Task 1/3; the four-way evidence/hypothesis distinction -> Task 2; "stored source of truth stays machine-readable" + separate rendering -> Task 1 (structure) + Task 4 (`render_for_prompt`); "reflection resets/partitions between games" -> Task 1's `reset()`; swappable summarizer seam -> Task 5; `Config` field at end of list -> Task 6; wiring without changing `decide()`'s signature -> Task 7 (chose not to touch `decide()` at all, the README's explicitly preferred lowest-risk option); tests -> one file, all 5 required areas covered (empty-state render, list-specific updates, full reset, refresh-interval mechanics, contradiction/belief-reversal).
- **Placeholder scan:** none -- every step has real code.
- **Type consistency:** `StructuredMemoryState`, `ConfirmedRule`, `Hypothesis`, `render_for_prompt`, `maybe_refresh_structured`, `StructuredSummarizer` are defined once in Task 1/4/5 and referenced identically in every later task and in `agent/my_agent.py`'s wiring.

## After all 7 tasks are green (not a plan task -- the session's own final wrap-up, per the original request)

1. Run the full suite once more and confirm the final count and 0 failures.
2. Update `docs/HANDOFF.md`'s "Parallel work split" table: one-line status update for the `baseline-130-hypothesis` row.
3. Commit the `docs/HANDOFF.md` update.
4. Push `feat/baseline-130-hypothesis-memory` to `origin` -- that branch only, no merge to `master`.
