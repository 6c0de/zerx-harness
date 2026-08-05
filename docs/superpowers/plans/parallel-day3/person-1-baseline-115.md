# Person 1 — `baseline-115-exact-state-memory`

**Read `README.md` in this directory first** — shared context, branch
table, and the shared-file etiquette rules this file assumes.

**Your branch:** `feat/baseline-115-exact-state-memory` (already exists on
the remote, forked from `master`, tests green at fork time).

## What you're building

From `STRATEGY.md` §3.1 ("State-specific ineffective-action memory," lines
93–106) and the ladder entry in §7:

> `baseline-115-exact-state-memory` — Exact `(state, action)` ineffective
> suppression alongside the existing graded structural signatures.
> Promote when: fewer repeated no-ops without new false suppressions.

The gap this fills: `zerx/heuristics.py`'s `DeadSignatureTracker` (already
built) grades *structural* signatures — `(color, size-bucket)`, scoped
game-wide, never a hard veto. What's missing is a *narrower, more
confident* layer: when the agent proposes the exact same action against
the exact same grid state it has already tried, and that exact attempt
already produced zero visible change and zero score delta, **suppress**
(not down-rank) proposing that literal (state, action) pair again. Unlike
the structural tracker, this one CAN be a hard suppression, because the
state is identical, not just similar — STRATEGY.md's own reasoning: "if
the state is truly identical, the outcome is already known with high
confidence."

## Interfaces you're producing

**New file: `zerx/exact_state_memory.py`**

Record shape, verbatim from `STRATEGY.md` §3.1:

```text
state_signature
action_signature
attempt_count
visible_change
level_delta
later_disconfirmed
```

Design it as a small, focused module mirroring `zerx/heuristics.py`'s
`DeadSignatureTracker` in spirit (same kind of "tracker with
record/query/reset" shape) but with the different suppression semantics
described above. Suggested (not mandatory — use your judgment, this is
less exhaustively pre-specified than Day 1's plan):

```python
@dataclass(frozen=True)
class ExactStateRecord:
    state_signature: str
    action_signature: str
    attempt_count: int
    visible_change: bool
    level_delta: int
    later_disconfirmed: bool


class ExactStateMemory:
    def record_outcome(
        self,
        state_signature: str,
        action_signature: str,
        visible_change: bool,
        level_delta: int,
    ) -> None: ...

    def is_suppressed(self, state_signature: str, action_signature: str) -> bool:
        """True only when this exact pair has been tried and produced no
        visible change and no level delta -- and has not since been
        `later_disconfirmed` (see below)."""
        ...

    def reset(self) -> None:
        """Clear between games -- exact-state evidence from one game must
        never leak into the next, same discipline as MemoryState.reset()
        and TransitionLedger.reset()."""
        ...
```

**`later_disconfirmed`** exists because STRATEGY.md is explicit this must
stay *graded, not permanent* in spirit even though it's a hard suppression
per-pair (§2.5/§5's "graded, not hard" preference — re-read §3.1's closing
paragraph: "Suppression should still not be a permanent universal ban on
the underlying action/object type — only on that exact (state, action)
pair"). Decide for yourself what "disconfirmed" means operationally (e.g.
does a suppression ever expire, or is `later_disconfirmed` purely a
diagnostic field recorded but not currently acted on?) and document your
choice in your track's plan file — this is a real design decision
STRATEGY.md leaves to the implementer, not an oversight to guess past.

**What computes `state_signature`?** You need a stable hash of a
`GameFrame`'s grid — `zerx/transitions.py`'s `_grid_hash` already does
exactly this (`hashlib.sha256` of the flattened grid, truncated to 16
chars) but it's a private (`_`-prefixed) function. Either import it
directly (acceptable — it's pure and stable) or, cleaner: check whether
promoting it to a small public helper (e.g. `zerx/transitions.py` exposing
`grid_hash(frame: GameFrame) -> str` alongside the existing `_grid_hash`,
with `_grid_hash` becoming a thin wrapper, or just renaming without the
underscore) is worth it — if you do this, it's a one-line, purely additive
change to `zerx/transitions.py` and won't conflict with anyone else's
branch (no other track touches that file). Your call; either approach is
fine, just don't duplicate the hashing logic.

`action_signature` — a stable string for an `Action` (e.g.
`f"{action.name.value}:{action.x},{action.y}"` for `ACTION6`, just
`action.name.value` otherwise). Simple, deterministic, no need to
overthink it.

## Config field

Add to `zerx/config.py` (per `README.md`'s etiquette — at the end of the
field list, `False` default):

```python
exact_state_suppression_on: bool = False
```

Add the matching line to `from_env`'s return call, at the end of its
argument list (`ZERX_EXACT_STATE_SUPPRESSION_ON` env var name, following
the existing `_env_bool` pattern already in that file).

## Wiring into `agent/my_agent.py`

Two things need to happen, both in `_choose_action_inner`, both in your
own delimited block per `README.md`'s etiquette:

1. **Feed outcomes back**, right where `DeadSignatureTracker.record_outcome`
   already gets called (after `self._transitions.finalize(frame)`). You
   need the BEFORE frame's grid hash and the action that was taken — both
   already available in that block (`self._pending_before_frame`,
   `self._pending_decision.action`, and `record.effective`/`record.changed_pixels`
   from the `TransitionRecord`). Unlike `DeadSignatureTracker` (which only
   feeds back when `target_object_label is not None`), this should feed
   back for **every** action, not just heuristic-sourced ones — exact-state
   suppression applies regardless of whether the action came from the
   model or a heuristic.
2. **Consult it before accepting an action.** This is the part that needs
   `decide()` awareness without changing its signature — see `README.md`'s
   rule: either (a) do this as a POST-check in `_choose_action_inner`
   after `decide()` returns (if `config.exact_state_suppression_on` and
   the returned decision's action is suppressed for the current state,
   fall back to `_deterministic_fallback`-style behavior — you may need to
   duplicate or reuse a small piece of that logic since it's currently
   private to `zerx/policy.py`), or (b) thread it into `decide()` via the
   one-new-optional-kwarg rule in `README.md`. Pick whichever is cleaner
   once you're looking at the real code — document which you chose and
   why in your plan file.

## Tests

Cover, in `tests/test_exact_state_memory.py`:
- New signature has no suppression.
- Recording an ineffective outcome (`visible_change=False, level_delta=0`)
  makes `is_suppressed` true for that exact pair.
- A *different* action against the same state, or the same action against
  a *different* state, is NOT suppressed.
- `reset()` clears everything.
- Whatever `later_disconfirmed` semantics you chose, tested.

Plus integration coverage in `tests/test_my_agent.py`-style (either extend
that file or add a focused new one) proving: with
`exact_state_suppression_on=True` and a scripted repeat of the identical
(state, action) pair, the second `choose_action` call does not propose
the suppressed action; with the flag `False` (default), behavior is
unchanged from before your change.

## Explicitly out of scope

- Level/game-scoped suppression variants (STRATEGY.md §3.1 mentions this
  as a documented future refinement, not baseline-115 scope).
- Touching `DeadSignatureTracker` itself — it stays exactly as-is,
  yours is a separate, narrower layer alongside it, not a replacement.
- Anything from `baseline-120`/`125`/`130` or the `exp-` tracks.
