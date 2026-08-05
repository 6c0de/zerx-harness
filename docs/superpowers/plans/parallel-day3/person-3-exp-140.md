# Person 3 — `exp-140-vlm-refinement` (candidate/arbiter infrastructure, off by default)

**Read `README.md` in this directory first** — shared context, branch
table, and the shared-file etiquette rules this file assumes.

**Your branch:** `feat/exp-140-vlm-refinement` (already exists on the
remote, forked from `master`, tests green at fork time).

## What you're building

From `STRATEGY.md` §3.2 and the ladder entry in §7:

> `exp-140-vlm-refinement` — Murad/Forge VLM-informed: multi-candidate,
> arbiter, confidence, click-failure-radius, frame-descriptor ablations.
> Promote when: a stated hypothesis is confirmed with matched
> seed/temperature/token budgets — not "more machinery" by default.

**Read §3.2 in full before starting; it matters more than usual here.**
The section is explicit that Murad/Forge VLM's own *winning, submitted*
profile disabled every one of these features
(`LLM_ACTION_CANDIDATES=1, LLM_CANDIDATE_ARBITER=0, LLM_CLICK_FAILURE_RADIUS=0,
LLM_CONFIDENCE_PROMPT=0, LLM_INCLUDE_FRAME_DESCRIPTOR=0`). Your job is
**not** to make any of these active by default or to argue for turning
them on — it's to build the *infrastructure* that would let someone run a
real matched-budget ablation later and get a real answer, exactly as
STRATEGY.md's "Promote when" column requires. Every feature you build
must default to the disabled/single-candidate/no-arbiter state.

Per §3.2's per-feature verdicts, here's what's actually in scope for this
track vs. explicitly not:

- **Multiple candidates + static scoring** — in scope. §3.2: "not in
  baseline. Later hypothesis-driven experiment... test candidate count 1
  vs 2/3 under identical seed/temperature/tokens." This is the core of
  what you're building.
- **LLM arbiter** — in scope, but narrowly: §3.2 says "implement a
  deterministic candidate scorer, collect candidate traces, only add an
  LLM arbiter if traces show recurring cases where valid candidates exist
  but deterministic selection picks wrong." Build the deterministic
  scorer for real; the LLM-arbiter *path* can be a real, tested,
  config-gated hook (so someone can flip it on later), but don't over
  invest in tuning it — it's explicitly the last resort per STRATEGY.md's
  own ordering.
- **Confidence prompting** — §3.2 explicitly says don't use self-reported
  confidence as a control threshold initially; if you touch this at all,
  it's collecting/recording it for later analysis, never gating behavior
  on it. Low priority for your time.
- **Frame descriptor** — §3.2: "`zerx/perception.py`'s ASCII grid +
  labeled-object table is already a more interpretable descriptor than a
  generic stats blob." **Out of scope** — don't build this.
- **Click-failure radius** — §3.2: "prefer component-aware failure memory
  (already Zerx's `DeadSignatureTracker`)... a small radius may remain a
  future experiment, not the primary mechanism." **Out of scope.**

So concretely: build (1) multi-candidate generation support in the
`ModelBackend` protocol/`decide()` path, (2) a deterministic static
candidate scorer, (3) a config-gated (off) arbiter hook. Skip frame
descriptor and click-failure-radius entirely; confidence collection only
if you have time left over.

## Interfaces you're producing

**`zerx/model_backend.py`** — the `ModelBackend` protocol currently has a
single `.generate(prompt: str) -> str`. Multi-candidate generation needs a
way to request N responses. Two reasonable approaches (pick one, document
why in your plan file):
(a) add a new optional method to the protocol,
`.generate_candidates(prompt: str, n: int) -> list[str]`, with a default
mixin/base implementation that just calls `.generate()` n times for
backends that don't override it (so `FakeModelBackend`/`CerebrasDevBackend`/
`GemmaModelBackend` don't all need bespoke changes), or
(b) keep `.generate()` as the only protocol method and do the "call N
times" loop in whatever new orchestration function you write instead,
never inside `decide()` itself.
(b) is very likely the lower-conflict-risk choice since it doesn't touch
the shared `ModelBackend` Protocol/`FakeModelBackend` at all — strongly
prefer it unless you have a specific reason not to.

**New file, e.g. `zerx/candidates.py`** (name your own choice):

```python
@dataclass(frozen=True)
class Candidate:
    raw_response: str
    parsed: Optional["ParsedAction"]  # from zerx.policy, None if parse failed
    static_score: float


def generate_candidates(
    backend: ModelBackend, prompt: str, count: int
) -> list[Candidate]:
    """Calls backend.generate(prompt) `count` times, parses each response,
    scores each deterministically. Never calls an arbiter -- that's a
    separate, explicitly optional step."""
    ...


def static_candidate_score(candidate_raw: str, parsed: Optional["ParsedAction"]) -> float:
    """Deterministic scoring per STRATEGY.md 3.2's factors: parse
    validity, plan length (n/a at baseline -- single action only, see
    AGENTS.md's 'no multi-action queues'), reset penalty, click
    plausibility, usable-action presence. Adapt the factor list to what's
    actually observable from a single-action ParsedAction; STRATEGY.md's
    list originally described Murad/Forge VLM's multi-action-plan scoring,
    which doesn't fully apply here -- use your judgment and document which
    factors you kept/dropped and why."""
    ...


def select_best_candidate(candidates: list[Candidate]) -> Optional[Candidate]:
    """Deterministic selection -- no LLM arbiter. This is what actually
    gets used when arbiter_on is False (the default, and the only
    supported mode for this track)."""
    ...
```

**Arbiter hook** — `Config.arbiter_on` already exists (Task 3, Day 1) and
already defaults to `False`. Confirm it's still wired as a true no-op
today (nothing currently reads it at all — verify this by grepping before
you start). Build a config-gated hook point (e.g. an
`arbiter: Optional[ModelBackend] = None` parameter somewhere in your new
candidate-selection flow, never in `decide()`'s own signature per
`README.md`'s rule) that, when `arbiter_on` is True and an arbiter backend
is provided, calls it to pick between candidates instead of
`select_best_candidate`'s deterministic logic. Keep this thin — it's
explicitly the lowest-priority, most speculative part of your scope.

## Config fields

Add to `zerx/config.py` (end of field list, defaults preserve current
single-candidate behavior):

```python
candidate_count: int = 1
# arbiter_on already exists (Task 3) -- confirm, don't re-add
```

Plus the matching `from_env` line(s), at the end of that method's
argument list. `candidate_count` should reject values `< 1` — if you add
validation, follow the existing pattern in `Config.__post_init__` (the
`budget_soft_cap` positivity check from Day 1's final review is the most
recent precedent).

## Wiring into `agent/my_agent.py` / `zerx/policy.py`

This is the track most likely to want an actual `decide()` change (to
route through multi-candidate generation instead of the single
`backend.generate()` call when `candidate_count > 1`). Follow `README.md`'s
rule strictly: **do not restructure `decide()`'s existing single-candidate
path.** Add the multi-candidate path as a new, separate branch — e.g.
`if config.candidate_count > 1: <your new path> else: <existing code,
completely unchanged>`. This keeps your diff to `zerx/policy.py` a clean
insertion, not a rewrite, which matters a lot for mergeability against
the other 3 branches editing the same file's surrounding context.

## Tests

New file `tests/test_candidates.py` covering:
- `generate_candidates` calls the backend exactly `count` times (use
  `FakeModelBackend` with `count` scripted responses).
- A response that fails to parse gets `parsed=None` and a low/zero score,
  but doesn't crash candidate generation for the others.
- `static_candidate_score` — at least one test per factor you actually
  implemented, showing it moves the score in the expected direction.
- `select_best_candidate` picks the highest-scored candidate; ties broken
  deterministically (document your tie-break rule); empty list returns
  `None` without raising.
- If you built the arbiter hook: a test with `arbiter_on=True` and a
  `FakeModelBackend` arbiter proving it gets consulted; a test with
  `arbiter_on=False` (default) proving it's never called (e.g. assert the
  arbiter backend's `call_count == 0`).
- Integration: `decide()` with `candidate_count=1` (default) produces
  byte-identical behavior to before your change, for every existing
  `tests/test_policy_decide.py` scenario — this is your most important
  regression guard, since it's proof the "off by default" contract holds.

## Explicitly out of scope

- Frame descriptor, click-failure-radius (§3.2 says don't build these —
  see above).
- Actually running a real ablation against Colab/Kaggle — infrastructure
  only, per `README.md`'s "what done does NOT require."
- Multi-action plans/queues — `AGENTS.md`'s hard rule, not just this
  track's scope boundary; never build this regardless of what Murad/Forge
  VLM's notebook does.
- Anything from `baseline-115`, `baseline-130`, or `exp-150`.
