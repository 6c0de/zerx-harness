# Day 3 parallel work split — read this first, all 4 people

Four people, four machines, four Claude Code sessions (Sonnet 5), same
GitHub repo (`https://github.com/6c0de/zerx-harness`), working
**simultaneously** on four independent tracks from `STRATEGY.md`'s §7
experiment ladder. This file is shared context every person's prompt
assumes you've read. Each person also gets their own file
(`person-N-*.md`) with their exact scope.

## Why these 4 tracks and not others

`STRATEGY.md` §7's ladder has dependencies: `baseline-125-phase-control`
needs `baseline-130-hypothesis` landed first (§6 step 4's own note); all
of `exp-200`/`exp-210`/`exp-220` need `baseline-130`'s structured memory
landed first (§4's "Deferred as isolated experiments" note). The 4 tracks
below were picked specifically because **none of them depends on another
landing first** — they can genuinely run in parallel without one person
blocking another:

| # | Track | Owner | Branch |
|---|---|---|---|
| 1 | `baseline-115-exact-state-memory` | — | `feat/baseline-115-exact-state-memory` |
| 2 | `baseline-130-hypothesis` (structured memory) | — | `feat/baseline-130-hypothesis-memory` |
| 3 | `exp-140-vlm-refinement` (candidate/arbiter infra, off by default) | — | `feat/exp-140-vlm-refinement` |
| 4 | `exp-150-duck-tools` Variants A+B (segmentation + fixed tools) | — | `feat/exp-150-duck-tools-ab` |

All 4 branches already exist on the remote, forked from `master` at the
commit that finished Day 1 + Day 2 (136/136 tests passing, `baseline-100`
recorded — see `docs/HANDOFF.md`). Fill in the "Owner" column yourselves
before starting.

## Read before starting (every person, no exceptions)

1. `AGENTS.md` — the binding operating contract. In particular: "Required
   control flow," "Never, under any circumstance" list, "Configuration
   and reproducibility," "Testing gates."
2. `STRATEGY.md` in full (456 lines — all 4 tracks are specified
   somewhere in it; read the whole thing, not just your section, since
   §2's principles and §6's control loop apply to everyone).
3. `docs/HANDOFF.md` — current state, what's done, what's not.
4. Your own `person-N-*.md` file in this directory.

## The one rule that makes 4 simultaneous branches mergeable

**Every feature ships OFF by default and additive-only in shared files.**
This isn't a style preference — it's the mechanism that keeps 4 people
editing the same codebase from producing an unmergeable mess. Concretely:

### `zerx/config.py`

Add your new `Config` field(s) at the **end** of the existing field list
(after `platform`), not interleaved anywhere else. Default value must
make your feature inert (`False` for an on/off flag, a value that
preserves current behavior otherwise). Add the matching `from_env(...)`
line at the end of that method's return-call argument list too. This
means when 4 branches each add one field at the end, merging is a
"stack the new lines" operation, not a real conflict.

**Do not** touch any existing field, remove anything, or reorder the
class. **Do not** add validation logic to `Config.__post_init__` unless
your track specifically requires a hard safety invariant (like the
existing `cerebras_dev`+`kaggle` guard) — prefer keeping new fields
inert-by-default instead of adding new validation branches, since
validation logic is exactly the kind of code that's easy to conflict on.

### `zerx/policy.py`'s `decide()`

**Do not change `decide()`'s function signature.** If your track's logic
needs to run inside the decision loop, either:
- implement it as a pure function in your own new module that `decide()`
  doesn't need to call at all (most of the 4 tracks work this way — see
  your own file), or
- if you genuinely need a new hook inside `decide()`, add exactly **one**
  new keyword-only parameter with a default of `None` at the very end of
  the parameter list, and gate all new behavior behind
  `if config.<your_flag> and <param> is not None:`. Never touch existing
  parameters, branches, or their order.

### `agent/my_agent.py`

If your track needs to observe transitions or feed something back (like
the existing `DeadSignatureTracker.record_outcome` call already does),
add your own clearly-delimited block using a comment banner:

```python
# --- <your-track-name> (feat/<your-branch>) ---
...your addition...
# --- end <your-track-name> ---
```

immediately after the existing, unmodified code — never inside an
existing `if`/`try` block. Guard the whole block with
`if self._config.<your_flag>:` so it's a true no-op when your feature is
off. This makes your addition a clean, independently-revertable hunk.

### Any new module you create

No etiquette needed — it's your own file, nobody else touches it. Prefer
a new file over extending an existing shared one whenever the choice
exists (e.g. `zerx/exact_state_memory.py` rather than piling onto
`zerx/heuristics.py`).

## Definition of done (every track)

- A short plan exists (`docs/superpowers/plans/YYYY-MM-DD-<your-track>.md`,
  written via the `superpowers:writing-plans` skill — your scope is
  already fully specified in your `person-N-*.md` file and in
  `STRATEGY.md`, so you can go straight to `writing-plans` without a
  separate brainstorming session).
- TDD throughout (`superpowers:test-driven-development`); run it
  yourself via `superpowers:subagent-driven-development` or
  `superpowers:executing-plans`, your choice.
- New tests pass; the **full existing suite still passes** (136 tests as
  of this handoff — confirm the count hasn't silently dropped).
- Your feature is OFF by default — running the full suite with no env
  vars set must produce byte-identical `decide()` behavior to before your
  change, for every existing test.
- No changes to `scripts/build_notebook.py`, `scripts/build_colab_notebook.py`,
  anything Kaggle-related, or anything touching `CEREBRAS_API_KEY` — out
  of scope for all 4 tracks.
- Commit messages follow the existing style in `git log` (imperative,
  explains why not just what).
- Push to **your own branch only** (`feat/...` from the table above) —
  **do not merge to `master` yourself.** One person (see `INTEGRATION.md`)
  merges all 4 branches in sequence once everyone's done, resolving the
  (expected, mechanical) `Config`/`decide()` conflicts by hand and running
  the full suite after each merge.
- Update `docs/HANDOFF.md`'s "Parallel work split" table with your actual
  completion status when done — a one-line status update, not a rewrite.

## What "done" does NOT require

Real Colab/Kaggle validation of your feature. All 4 tracks are local,
model-free work (matches the project's `FakeModelBackend` testing
discipline throughout Day 1/2) — proving your feature's logic is correct
and off-by-default-safe locally is the bar. Actually measuring whether it
*improves* anything (STRATEGY.md §7's "Promote when" column) is a
separate, later step once `baseline-115`/`baseline-130`/etc. are all
merged and someone runs a real ablation — don't block on that.

## Questions / ambiguity

Each `person-N-*.md` file is written to the same standard of rigor as
Day 1's implementation plan — real interfaces, real STRATEGY.md
citations, no invented scope. If your Claude Code session hits a genuine
ambiguity STRATEGY.md doesn't resolve, that's expected (these are less
exhaustively pre-specified than Day 1's 15-task plan) — use your own
judgment consistent with §2's principles, note the decision in your
track's plan file, and flag it in your status update. Don't block on it.
