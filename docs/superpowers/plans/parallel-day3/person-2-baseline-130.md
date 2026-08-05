# Person 2 — `baseline-130-hypothesis` (structured memory)

**Read `README.md` in this directory first** — shared context, branch
table, and the shared-file etiquette rules this file assumes.

**Your branch:** `feat/baseline-130-hypothesis-memory` (already exists on
the remote, forked from `master`, tests green at fork time).

## What you're building

From `STRATEGY.md` §2.4, §3.1 (lines 73–88), and the ladder entry in §7:

> `baseline-130-hypothesis` — Structured claims (§2.4/§3.1's schema) and
> contradiction/probe checks. Promote when: fewer repeated probes and
> belief reversals.

The gap: `zerx/memory.py`'s current `MemoryState` (built Day 1) is
deliberately a single free-text `summary: str` field — correct for the
baseline, but §2.4 is explicit about why it's not the end state: "Memory
must not store every model statement as fact." The target schema
(STRATEGY.md §3.1, verbatim):

```json
{
  "confirmed_rules": [],
  "working_hypotheses": [],
  "rejected_hypotheses": [],
  "open_questions": [],
  "current_goal": "",
  "current_plan": [],
  "notable_failures": []
}
```

This is the single biggest track of the 4 — you're changing the shape of
a core type other things touch, not just adding a new file. Read
`zerx/memory.py` in full before starting; understand `maybe_refresh`'s
current contract (pure function, returns a new `MemoryState`, never
mutates the input, calls an injected `Summarizer` callable) before you
change it.

## Design constraints from STRATEGY.md (not optional, read carefully)

- **§2.4's four-way evidence/hypothesis distinction is the whole point**:
  observed transition (raw fact) → confirmed rule (repeated/discriminating
  evidence) → working hypothesis (plausible, unverified) → rejected
  hypothesis (contradicted). Don't collapse these into one list with a
  status field if you can help it — the separation is what "reduces
  self-reinforcing hallucination," per STRATEGY.md's own stated purpose.
- **"The rendered prompt may include a compact textual form; the stored
  source of truth stays machine-readable."** (§3.1) — i.e. `MemoryState`
  itself must be structured (dataclasses/typed lists), and a *separate*
  rendering function turns it into prompt text. Don't store a blob of
  pre-rendered text as the source of truth.
- **"Reflection resets or correctly partitions between games."** — same
  discipline as the existing `MemoryState.reset()`; a structured
  `MemoryState` needs the same guarantee, at every field, not just the
  top-level object.
- **"Both deterministic summarization and model-generated reflection stay
  viable experiments — the design must not couple the rest of the
  pipeline to one summarizer implementation."** — the existing
  `Summarizer` callable pattern (`Callable[[str, str], str]`) already
  achieves this for free-text; your structured version needs an analogous
  seam. Its exact shape (does the summarizer now take/return structured
  data instead of strings? a different callable signature entirely?) is
  a real design decision left to you — STRATEGY.md specifies the target
  *data shape*, not the summarizer's new call signature. Document your
  choice and reasoning in your plan file.

## Interfaces you're producing

Redesign `zerx/memory.py`'s `MemoryState` (or introduce a new structured
type alongside it — your call, but if you keep the name `MemoryState`,
every existing caller of it needs updating, which is a bigger blast
radius; consider whether a new `StructuredMemoryState` type, used only
when a new `Config.structured_memory_on` flag is on, is actually the
lower-risk path given `agent/my_agent.py` and `zerx/policy.py`'s
`decide()`/`build_prompt()` currently construct/consume `MemoryState`
directly — read those call sites before deciding).

Suggested shape (adapt as needed — this is the JSON schema translated to
Python, not a literal mandate on class names):

```python
@dataclass(frozen=True)
class ConfirmedRule:
    statement: str
    evidence_count: int


@dataclass(frozen=True)
class Hypothesis:
    statement: str
    supporting_evidence: int
    contradicting_evidence: int


@dataclass
class StructuredMemoryState:
    confirmed_rules: list[ConfirmedRule] = field(default_factory=list)
    working_hypotheses: list[Hypothesis] = field(default_factory=list)
    rejected_hypotheses: list[Hypothesis] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    current_goal: str = ""
    current_plan: list[str] = field(default_factory=list)
    notable_failures: list[str] = field(default_factory=list)
    step_count: int = 0
    last_refreshed_step: int = 0

    def reset(self) -> None: ...


def render_for_prompt(state: StructuredMemoryState) -> str:
    """The compact textual form STRATEGY.md §3.1 describes -- structured
    data stays the source of truth, this is a pure rendering function."""
    ...
```

`maybe_refresh`'s equivalent for the structured version: same
"advance step count, refresh on interval, never mutate input, return a
new state" contract as the existing `maybe_refresh` — but now the
injected callable needs to produce structured updates (which fields
change on refresh, and how) rather than a single new summary string.
Decide the callable's signature yourself; whatever you pick, keep a
no-op/deterministic default implementation available (mirroring how
`zerx/policy.py`'s `decide()` currently passes `lambda prev, ctx: prev`
as a deterministic no-op summarizer) so this stays testable without a
real model, same as everything else in this codebase.

## Config field

Add to `zerx/config.py` (per `README.md`'s etiquette — end of field list,
default preserves current behavior):

```python
structured_memory_on: bool = False
```

Plus the matching `from_env` line, at the end of that method's argument
list.

## Wiring

If you introduce a new `StructuredMemoryState` type rather than replacing
`MemoryState`, wiring is additive: `agent/my_agent.py`'s `MyAgent.__init__`
gets a new, config-gated `self._structured_memory` alongside the existing
`self._memory`, in your own delimited block per `README.md`. Whether
`decide()`/`build_prompt()` need to *consume* it (e.g. render it into the
prompt) is part of your scope, but per `README.md`'s rule, do this without
changing `decide()`'s existing signature/behavior when the flag is off —
if `build_prompt()` needs a new optional parameter, same one-new-kwarg
rule applies.

## Tests

New file `tests/test_structured_memory.py` (or extend
`tests/test_memory.py` if you kept the same type name) covering:
- Empty structured state renders sensibly (`render_for_prompt`).
- Recording a confirmed rule / hypothesis / rejection updates the right
  list, not the others.
- `reset()` clears every field, not just some.
- The refresh-interval mechanics (due / not due / never mutates input) —
  same test shapes as the existing `test_memory.py`, adapted to the
  structured shape.
- If you implement any contradiction/probe-check logic (§7's "Promote
  when: fewer repeated probes and belief reversals" implies some
  mechanism for detecting a belief reversal — decide how deep to go here,
  document the choice), test it.

Confirm the full existing suite (136 tests) still passes with your
feature off, and that `tests/test_memory.py`'s existing tests are
untouched if you kept `MemoryState` as a separate, still-functioning type.

## Explicitly out of scope

- `baseline-125-phase-control` (EXPLORE/VERIFY/EXECUTE/RECOVER) — STRATEGY.md
  §6 step 4 explicitly sequences this *after* `baseline-130` because it
  needs the hypothesis structure to be meaningful. Building the structure
  is your job; consuming it for phase control is a separate, later track.
- A real model-generated reflection summarizer — the injected-callable
  seam is your job; wiring a real Gemma call through it happens once
  `GemmaModelBackend` is exercised for real (Day 2 territory, already
  partially done, not yours to extend here).
- Anything from `baseline-115`, `exp-140`, or `exp-150`.
