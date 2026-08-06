# Integration — `feat/baseline-120-followups` + `feat/policy-prompt-legal-budget` + `integration/baseline-120-colab-ready` → `master`

**One person/session does this**, same rule as every prior integration in
this project (`docs/superpowers/plans/parallel-day3/INTEGRATION.md`,
`docs/superpowers/plans/parallel-baseline-120/INTEGRATION.md`). Don't have
two sessions merge simultaneously.

This plan was written by a session that built and fully verified
`feat/baseline-120-followups` (trace capture, live+replay visualizer,
diagnostic logging, README) and then, at the human owner's request,
investigated two teammate branches to prepare this integration — but did
**not** execute the merge itself. That is this plan's job, in a fresh
session with no memory of the branch's build history. Everything you need
is below or linked; do not assume you remember anything from "before."

## Verified starting point (do not re-derive, re-verify if in doubt)

- **`origin/master`:** `b405d3b19f2b7b5c87d5bf541813835cbb6c6c0e`
  (`docs(handoff): consolidated baseline-120 integration summary` — the
  `baseline-120-reki-core` 4-track integration from earlier this project).
  All three branches below fork from **exactly this commit** — confirmed
  via `git merge-base`, not assumed.
- **`origin/feat/baseline-120-followups`:** `bceaeaaa3854bf4cec1274e92e8575485d6968d2`.
  Fully built, task-reviewed (7 tasks, one needed a fix round), final
  whole-branch-reviewed (3 Important findings found and fixed), and then
  live-debugged with the human owner against a real `cerebras_dev` run
  (3 more small fixes: the ARC-AGI-3 color palette, `Decision.model_error`
  diagnostic logging, an HTTP-error-body-capturing fix, and a
  reasoning-panel newline-rendering fix). Full unfiltered suite green at
  **353 passed, 0 failed** as of this branch's tip. Design spec:
  `docs/superpowers/specs/2026-08-06-baseline-120-followups-design.md`.
  Implementation plan: `docs/superpowers/plans/2026-08-06-baseline-120-followups.md`.
- **`origin/feat/policy-prompt-legal-budget`:** `09db286bd4b7e31964d5bb62341b5828d447d14d`.
  4 commits on top of `b405d3b`: `e402a0d fix(policy): surface legal
  actions and budget signal in build_prompt`, `c964ea0 fix(kaggle,policy):
  repair submission-breaking packaging and wire progress signal`,
  `8a6bc05 docs(handoff): record open item 6 as partially resolved with
  A/B evidence`, `09db286 docs(handoff): branch-wide resolution check for
  all 20 audit findings`. Also carries a new `docs/audits/2026-08-06-full-repository-audit.md`
  (763 lines — read it for context on what "20 audit findings" means
  before touching this branch's content). **This is the exact fix for the
  `build_prompt()` legal-actions gap** `docs/HANDOFF.md` has documented
  since the `baseline-120` dev-lane sweep, and which this session's own
  human owner independently, live reproduced against a real `cerebras_dev`
  run while debugging `feat/baseline-120-followups` (see
  `docs/HANDOFF.md`'s `feat/baseline-120-followups` status section) —
  the model proposed a syntactically valid `ACTION6` that was correctly
  rejected because `ls20` never has `ACTION6` legal, and had no way to
  know that from the prompt. This branch fixes exactly that.
- **`origin/integration/baseline-120-colab-ready`:** `4a1fda18271a677da19a6e8dcc18f917ab5707f0`.
  Itself a 2-way integration branch: merges `origin/feat/policy-prompt-legal-budget`
  (at an **earlier** point — commit `8a6bc05`, missing
  `feat/policy-prompt-legal-budget`'s current tip commit `09db286`, see
  "Known branch-state nuance" below) and a `feat/baseline-120-8bit-quant-parity`
  branch (not present as a separate ref on `origin` — already fully
  absorbed via its own merge commit `5654c65`, nothing further to do with
  it), plus its own Colab-specific commits: `7429419 fix(build): make
  Makefile portable to Windows`, `96063b9 fix(colab): switch model
  quantization from 4-bit bitsandbytes to 8-bit fp8`, `2b8d0fb
  docs(handoff): record Colab/Kaggle 8-bit fp8 quantization parity
  decision`, `4a1fda1 fix(colab): clone and slim the ARC-AGI-3-Agents
  framework before use`. Trying to get a Colab run working end-to-end,
  independent of anything `feat/baseline-120-followups` added (it forked
  before that work started).

### Known branch-state nuance — confirm still true before merging

`integration/baseline-120-colab-ready` does **not** contain
`feat/policy-prompt-legal-budget`'s current tip (`09db286`) — verified via
`git merge-base --is-ancestor 09db286 origin/integration/baseline-120-colab-ready`
returning false at plan-writing time. If the teammates have since pushed
more commits to either branch, **re-run this check** (and everything else
in this "Verified starting point" section) before trusting this plan's
specifics — re-derive, don't assume this document is still accurate the
moment new commits land.

## Why this order

Smallest / most independently-understood first, same logic as every prior
`INTEGRATION.md` in this project:

1. **`feat/baseline-120-followups`** — the merging session's author (this
   plan's author) has full, verified context on every line of this
   branch; zero unknowns. Also the smallest real conflict surface with
   the other two (see below).
2. **`feat/policy-prompt-legal-budget`** — introduces the one substantial,
   confirmed code conflict (`zerx/policy.py`, exact resolution given
   below). Merging it before `integration/baseline-120-colab-ready`
   matters: `colab-ready` already contains most of this branch via its
   own earlier merge, so once `policy-prompt-legal-budget` is merged into
   `master`/the integration branch, `colab-ready`'s shared history is
   already present and git's merge will apply only `colab-ready`'s own
   unique diff cleanly, instead of re-resolving the same `zerx/policy.py`
   conflict twice.
3. **`integration/baseline-120-colab-ready`** — biggest, most dependent
   (depends on `policy-prompt-legal-budget`'s content already being
   present), merges last.

## Preconditions (do these first, in a fresh session — do not skip)

1. `git fetch --all --prune`.
2. Re-verify every SHA and every `git merge-base` claim in "Verified
   starting point" above — this plan could be stale by the time you run
   it.
3. Individually verify each of the 3 branches is green **on its own**,
   the same discipline every prior integration in this project used —
   don't trust this plan's "353 passed" figure or the branches' own
   commit messages without running the suite yourself:
   ```bash
   git checkout --detach origin/feat/baseline-120-followups
   .venv/Scripts/pytest.exe tests/ -q          # expect 353 passed, 0 failed
   git checkout --detach origin/feat/policy-prompt-legal-budget
   .venv/Scripts/pytest.exe tests/ -q          # record the actual count
   git checkout --detach origin/integration/baseline-120-colab-ready
   .venv/Scripts/pytest.exe tests/ -q          # record the actual count
   ```
   The full unfiltered suite includes a 25-game live-engine sweep; per
   this project's own established finding (`docs/HANDOFF.md`), once a
   real backend-selection fix is present it completes in ~20-40s, not the
   ~20 minutes it took before that fix existed on `master` — if any of
   these three checkouts takes far longer than that, don't assume it's
   just slow; investigate before proceeding.
4. Read `docs/audits/2026-08-06-full-repository-audit.md` (only present
   on `feat/policy-prompt-legal-budget` and `integration/baseline-120-colab-ready`
   at this point) for context on what the "20 audit findings" work
   actually covers — this plan does not summarize it.

## Interface / shared-code freeze point

`zerx/policy.py`'s `Decision` dataclass and `build_prompt()`'s function
body are **not** touched by both sides at once — confirmed by diffing
each branch against `b405d3b` independently:

- `feat/baseline-120-followups` adds `Decision.raw_response` and
  `Decision.model_error` (both `Optional[str] = None`, additive,
  trailing fields) and touches only the **call sites** inside `decide()`
  that invoke `backend.generate(...)`/`build_prompt(...)` — never
  `build_prompt()`'s own body or signature, never `Decision`'s field list
  beyond appending.
- `feat/policy-prompt-legal-budget` adds two new parameters to
  `build_prompt()` (`legal_actions: FrozenSet[ActionName] = frozenset()`,
  `budget: Optional[BudgetSignal] = None`, both defaulted — non-breaking)
  and rewrites `build_prompt()`'s body to render them, plus a
  `_MAX_PROMPT_OBJECTS` truncation guard. It also touches the same
  `decide()` call sites `feat/baseline-120-followups` touches — this is
  the one real conflict, resolved exactly below. It does **not** touch
  `Decision`'s field list at all (confirmed: `grep -n "class Decision"`
  on the diff finds nothing).

Because the two sides' changes are on the same lines but touch
non-overlapping *concerns* (this branch: capture `model_error`/
`raw_response`; that branch: which `build_prompt()` args to pass), the
resolution is a straightforward union of both edits, not a judgment call
about which side "wins."

## Procedure

For each branch:

```bash
git checkout master
git pull --ff-only
git checkout -b integration/baseline-120-followups   # only on the FIRST branch; reuse it after
git merge origin/<branch-name> --no-edit
```

### 1. Merge `feat/baseline-120-followups`

```bash
git merge origin/feat/baseline-120-followups --no-edit
```

**Expected: clean, no conflicts.** `master` at `b405d3b` has none of this
branch's files modified by anyone else yet at this point in the sequence.

```bash
.venv/Scripts/pytest.exe tests/ -q      # expect 353 passed, 0 failed
```

### 2. Merge `feat/policy-prompt-legal-budget`

```bash
git merge origin/feat/policy-prompt-legal-budget --no-edit
```

**Expected conflicts, with exact resolutions:**

#### `zerx/policy.py` — real conflict, resolve by union

Both sides edited `decide()`'s model-call section. The conflict will
appear around the `if config.candidate_count > 1:` block. Resolve to
exactly this (combining `feat/baseline-120-followups`'s `model_error`
capture with `feat/policy-prompt-legal-budget`'s new `build_prompt()`
arguments — do not pick one side, both changes are needed):

```python
    raw_response: Optional[str] = None
    model_error: Optional[str] = None
    if config.candidate_count > 1:
        try:
            from zerx.candidates import generate_candidates, select_candidate

            prompt = build_prompt(perception, new_memory, candidates, legal_actions, budget)
            model_candidates = generate_candidates(
                backend, prompt, legal_actions, config.candidate_count
            )
            best = select_candidate(model_candidates, config)
            parsed = best.parsed if best is not None else None
        except Exception as exc:
            parsed = None
            model_error = f"{type(exc).__name__}: {exc}"
    else:
        try:
            raw_response = backend.generate(
                build_prompt(perception, new_memory, candidates, legal_actions, budget)
            )
            parsed = parse_action(raw_response, legal_actions)
        except Exception as exc:
            parsed = None
            model_error = f"{type(exc).__name__}: {exc}"
```

`build_prompt()`'s own function body/signature and `Decision`'s field
list are each touched by only one side (see "Interface / shared-code
freeze point" above) — those hunks should auto-resolve with no conflict.
If either shows a conflict anyway, that means someone touched something
this plan didn't predict — stop and read both sides' intent before
resolving by hand, don't guess.

**After resolving**, grep-verify the union actually landed:
```bash
grep -n "model_error" zerx/policy.py           # must appear (from followups)
grep -n "legal_actions, budget" zerx/policy.py  # must appear twice (from policy-prompt-legal-budget)
```

#### `agent/my_agent.py` — expected low risk, verify anyway

The two branches touch different regions of this file:
`feat/policy-prompt-legal-budget` changes `_to_game_frame` (maps
`FrameData.levels_completed` onto `GameFrame.score`, fixing a real,
previously-documented "score is always 0" limitation) and a comment near
the exact-state-memory outcome-recording block, both **before** the
`decide()` call in `_choose_action_inner`. `feat/baseline-120-followups`
adds `self.trace_recorder` in `__init__` and the `model_error` warning
log **after** the `decide()` call returns. These shouldn't conflict, but
don't assume — if git reports no conflict, still read the merged file's
`_choose_action_inner` end to end once to confirm both changes are
present and coherent.

**Separately, worth investigating (not necessarily fixing) while you're
in this file:** `feat/policy-prompt-legal-budget` also adds a
`_shapes_match`-equivalent shape-change guard to `zerx/transitions.py`'s
`_diff()` (see that branch's `zerx/transitions.py` diff). `agent/my_agent.py`
already has its **own** private `_shapes_match` helper, added earlier
(commit history: `baseline-115-exact-state-memory`) specifically because
"`zerx/transitions.py` is shared infrastructure and is not modified for
this fix; the guard lives here instead" (see that function's docstring).
Now that `zerx/transitions.py` itself has shape-change handling, check
whether `agent/my_agent.py`'s local `_shapes_match` and its call site are
still doing useful, non-redundant work, or whether they're now dead code
duplicating logic `zerx/transitions.py` handles itself. Don't touch this
without understanding both sides' exact behavior first — flag it in your
own status notes if you find real duplication, but fixing it is out of
this integration's required scope unless it's actually broken.

#### `tests/test_policy_decide.py` — expected low risk

`feat/policy-prompt-legal-budget` inserts its new tests mid-file (after
`test_build_prompt_without_candidates_says_so`, before
`test_decide_budget_favoring_execution_triggers_heuristic_even_when_heuristic_first_off`).
`feat/baseline-120-followups` appends its `model_error`/`raw_response`
tests at the very end of the file. Different regions — expect a clean
auto-merge; if git flags a conflict anyway, keep both sides' test
functions, don't drop either.

#### `docs/HANDOFF.md` — expected, mechanical, per this project's own precedent

Both branches append substantial new content. Keep every side's content
during the merge itself (per every prior `INTEGRATION.md` in this
project: "don't blindly take 'ours' or 'theirs'"); the **integration
owner does the final consolidated rewrite** after all 3 branches are in,
same as every previous integration round — see step 4 below. Don't spend
time perfecting `HANDOFF.md`'s prose during this individual merge step,
just don't lose content.

**After resolving all conflicts:**

```bash
.venv/Scripts/pytest.exe tests/ -q
```

Must show more tests than step 1's count and zero failures. If
`zerx/policy.py`'s merge is wrong, expect failures in `tests/test_policy_decide.py`
specifically (both the pre-existing `build_prompt`/`decide` tests and the
new `model_error` tests) — that's your signal to re-check the resolution
above, not to weaken a test.

**Extra verification specific to this branch's fix** (per this session's
own live reproduction of the bug this branch fixes): re-run the same
kind of dev-lane sanity check `docs/HANDOFF.md`'s `baseline-120`
experiment record and this session's own live debugging both used —
confirm `build_prompt()` now actually includes the legal action names:

```python
from zerx.perception import PerceptionResult
from zerx.policy import build_prompt
from zerx.memory import MemoryState
from zerx.types import ActionName

prompt = build_prompt(
    PerceptionResult(ascii_grid="0", objects=()),
    MemoryState(),
    legal_actions=frozenset({ActionName.ACTION1, ActionName.RESET}),
)
assert "ACTION1" in prompt and "ACTION2" not in prompt
```

### 3. Merge `integration/baseline-120-colab-ready`

```bash
git merge origin/integration/baseline-120-colab-ready --no-edit
```

Since `colab-ready` already contains most of `feat/policy-prompt-legal-budget`'s
content (via its own earlier merge), and that content is now already
present in your integration branch from step 2, expect this merge to be
**much smaller and lower-conflict** than step 2 — git should recognize
the shared history and apply mostly `colab-ready`'s own unique diff:
`Makefile` (Windows portability), `scripts/build_colab_notebook.py` and
`tests/test_build_colab_notebook.py` (quantization/framework-slimming
changes), plus whatever `docs/HANDOFF.md` content is genuinely unique to
this branch.

**Known nuance to watch for:** this branch is missing
`feat/policy-prompt-legal-budget`'s tip commit (`09db286`, the "20 audit
findings" resolution doc) — see "Known branch-state nuance" above. Since
you merge `feat/policy-prompt-legal-budget` (including `09db286`) in
step 2 *before* this step, `09db286`'s content should already be present
in your integration branch, and this merge should not need to
re-introduce or conflict over it. If it does conflict here, that's a
signal the branches have diverged further than this plan accounted for —
stop and re-read both sides' current state rather than forcing a
resolution blind.

**`docs/HANDOFF.md` reconciliation note:** this session's own
investigation found `colab-ready`'s `docs/HANDOFF.md` is *not* simply
`feat/policy-prompt-legal-budget`'s version plus more — it's
substantially condensed/rewritten in places (confirmed: diffing
`feat/policy-prompt-legal-budget`'s `HANDOFF.md` against `colab-ready`'s
shows ~500 lines changed, mostly deletions, not a clean append). Don't
resolve this merge conflict by mechanically keeping every line from both
sides without reading them — some of `colab-ready`'s edits may be
deliberate simplifications of `policy-prompt-legal-budget`'s content, not
just additions. Read both versions' intent before resolving, same
standard as any other real conflict in this plan.

```bash
.venv/Scripts/pytest.exe tests/ -q
```

Must show more tests than step 2's count (verify `tests/test_build_colab_notebook.py`
and `tests/test_kaggle_bundle_importable.py`'s new/changed tests
specifically pass) and zero failures.

## After all 3 are merged into `integration/baseline-120-followups`

1. Run the full suite once more; sanity-check the total test count makes
   sense (sum of each branch's own reported new-test count, matching the
   discipline every prior integration in this project used — don't just
   trust a number without adding it up).
2. Grep-confirm no duplicate/dead logic slipped through:
   ```bash
   grep -n "_shapes_match" agent/my_agent.py zerx/transitions.py
   ```
   (see the `agent/my_agent.py` section above — decide whether this is
   worth a follow-up note, not necessarily a fix in this integration).
3. Confirm `zerx/backends/cerebras_dev.py`'s lazy-import pattern (added
   by `feat/policy-prompt-legal-budget`, moving `from zerx.backends.cerebras_dev
   import CerebrasDevBackend` from module scope into the one branch of
   `select_backend()` that needs it, to keep the Kaggle bundle importable
   without the `cerebras_dev` module) didn't get reverted or duplicated
   by any merge — `grep -n "CerebrasDevBackend" zerx/model_backend.py`
   should show exactly one import, inside `select_backend()`, not at
   module scope.
4. Confirm `STRATEGY.md` is untouched by this entire integration —
   `git diff master origin/master -- STRATEGY.md` before you started,
   compared to your integration branch's `STRATEGY.md`, should be
   identical. None of the 3 source branches touch it (verified at
   plan-writing time); if your merged branch shows a diff there anyway,
   something unexpected happened — stop and investigate before
   proceeding, per this project's standing "only the integration owner
   edits `STRATEGY.md`, and only after real Colab numbers exist" rule
   (`AGENTS.md`, `STRATEGY.md` §7.1, and this session's own explicit
   instruction from the human owner never to write a
   keep/revert/investigate verdict before that run happens).
5. Merge into `master`:
   ```bash
   git checkout master
   git merge integration/baseline-120-followups --no-edit
   .venv/Scripts/pytest.exe tests/ -q
   ```
   **Do not push to `origin/master` without the human owner's explicit
   confirmation** — same standing rule this whole project has followed
   for every integration so far. Present the final test count, the list
   of what's now in `master` that wasn't before, and wait for a clear go
   before `git push origin master`.
6. **Only now**, write the consolidated `docs/HANDOFF.md` update —
   replacing the 3 branches' (well, effectively 2 distinct efforts' —
   `feat/baseline-120-followups` and the `feat/policy-prompt-legal-budget`
   / `integration/baseline-120-colab-ready` pair) individual status
   entries with one coherent section. This is also the natural place to
   record, explicitly, that the `build_prompt()` legal-actions gap
   (`docs/HANDOFF.md`'s known-failures item 6) is now fixed — cite the
   actual commit — and that the visualizer/trace tooling this session
   built is what let a human directly watch and diagnose the dev-lane
   Cerebras run this round, closing the loop `docs/HANDOFF.md`'s Colab-
   postponement note from the `baseline-120` integration round described
   as the reason the visualizer needed to exist in the first place. Do
   **not** write a `STRATEGY.md` keep/revert/investigate verdict as part
   of this — that still waits on the real Colab Gemma-4-31B-it run,
   unchanged from every earlier instruction in this project's history.
7. State the actual next action in `docs/HANDOFF.md`'s "Exact next
   action" — likely candidates, informed by everything now on `master`:
   re-running the Cerebras dev-lane sweep with the fixed prompt (should
   no longer fall back to `ACTION6` guesses on games that don't support
   it) and/or finally attempting the real Colab Gemma-4-31B-it run this
   project has postponed since the `baseline-120` integration, now that
   both of its stated prerequisites (visualizer, `build_prompt()` fix)
   are done. This is a recommendation to write down, not something to
   start automatically.
8. The 3 source branches can stay or be deleted once `master` is
   confirmed merged and pushed — human owner's call, not automatic, same
   as every prior integration round.

## Rollback

Same standing rule as every prior integration in this project: prefer
`git revert` of the specific merge commit over a broader reset if a
regression surfaces post-merge. Do not force-push, delete branches, or
run any other destructive git operation without the human owner's
explicit request.
