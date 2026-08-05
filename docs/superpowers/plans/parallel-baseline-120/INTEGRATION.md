# Integration — merging all 4 `baseline-120` tracks back into `master`

**One person does this** (whoever's coordinating), same rule as
`docs/superpowers/plans/parallel-day3/INTEGRATION.md`. Don't have
multiple people merge simultaneously.

## Preconditions

All 4 branches (`feat/baseline-120-backend-wiring`,
`feat/baseline-120-eval-harness`, `feat/baseline-120-local-regression`,
`feat/baseline-120-colab-validation`) pushed to the remote, each track's
own full local suite green, each track's `docs/HANDOFF.md` status line
updated. Track 4's real Colab run (Part B of its own file) should be
complete before you merge it — merging Track 4 with only Part A done is
acceptable if the human owner explicitly decides to land the
infrastructure now and run the experiment in a follow-up session, but
that decision must be stated explicitly in `docs/HANDOFF.md`, not implied.

## Interface freeze point

Frozen from the start of this plan (`README.md`'s "Interface / data
contracts" section): `zerx.model_backend.select_backend(config: Config) -> ModelBackend`
and `eval.run_ablation.run_games(config, game_ids, max_steps=200) -> List[ExperimentRecord]`.
If either signature changed during implementation, note it here before
merging anything that depends on it.

## Procedure

Merge **one branch at a time**, in this order — smallest/least-invasive
first, most dependent last, same logic as Day 3's own integration:

1. `feat/baseline-120-local-regression` (Track 3) — new test file only, no
   shared-file touches beyond a `docs/HANDOFF.md` line. Zero dependency on
   anything else in this stage.
2. `feat/baseline-120-backend-wiring` (Track 1) — small, foundational:
   `zerx/model_backend.py`'s new `select_backend`, `agent/my_agent.py`'s
   one-line fix, `zerx/config.py`'s one new field.
3. `feat/baseline-120-eval-harness` (Track 2) — depends on Track 1 for its
   real-integration test to exercise the actual (fixed) backend path
   end-to-end.
4. `feat/baseline-120-colab-validation` (Track 4) — biggest, carries the
   real experiment record and status-doc updates; merges last.

For each branch:

```bash
git checkout master
git pull --ff-only
git checkout -b integration/baseline-120   # only on the FIRST branch; reuse it after
git merge feat/<branch-name> --no-edit
```

**Expected conflicts, and why they should be minimal this round:** unlike
Day 3 (which had a real 3-way `zerx/config.py` stack and a `zerx/policy.py`
branch-addition), this round only Track 1 touches `zerx/config.py` and
only Track 1 touches `agent/my_agent.py` — so if everyone followed the
ownership matrix in `README.md`, you should see **no conflicts at all**
in those files (each is edited by exactly one track). If a conflict does
show up in `zerx/config.py` or `agent/my_agent.py`, that means someone
edited a file they didn't own this round — stop, read what changed and
why (their `person-N-*.md` file explains their intent), and resolve by
hand; don't blindly take "ours" or "theirs." A conflict in
`docs/HANDOFF.md` is expected and mechanical (all 4 tracks add a status
line) — keep every track's line, then do your own final consolidated
rewrite of that file's `baseline-120` section after all 4 are in (see
"After all 4 are merged" below).

After each merge:

```bash
.venv/bin/pytest tests/ -q      # Mac/Linux
# or: .venv\Scripts\pytest.exe tests/ -q   # Windows
```

Must show **more tests than before** and **zero failures**. Fix before
merging the next branch — don't stack a broken merge under another one.

**After merging Track 1 specifically**, run this extra check before
proceeding to Track 2:

```bash
grep -n "GemmaModelBackend(self._config.model_revision)" agent/my_agent.py
```

This must return nothing — confirming the old hardcoded construction is
actually gone, not just that a new function was added alongside it.

**After merging Track 2**, re-run its real-engine test explicitly and
confirm it now exercises the actual fixed backend path (Track 1 having
already landed):

```bash
.venv/bin/pytest tests/test_run_ablation.py -v
```

## After all 4 are merged into `integration/baseline-120`

1. Run the full suite once more; confirm the total test count is sensible
   (261 baseline + Track 1's ~7 new tests + Track 2's new tests + Track
   3's new tests + Track 4's new tests).
2. Confirm `zerx/config.py`'s new `gemma_base_url` field is present
   exactly once, defaulting to the original hardcoded URL — grep for it,
   don't just trust the merge.
3. Run Track 3's full 25-game crash-safety sweep one more time against the
   fully-merged `integration/baseline-120` branch (not just each track's
   individual branch) — this is the first point where all 4 tracks'
   changes are combined, and it's the cheapest, fastest full-stack check
   before treating the stage as mergeable.
4. Confirm `agent/my_agent.py` still constructs `MyAgent` cleanly and
   `tests/test_my_agent.py` (Day 3's existing file, untouched by this
   stage) still passes — worth an explicit look even though no track this
   round should have touched it.
5. Merge `integration/baseline-120` into `master`:
   ```bash
   git checkout master
   git merge integration/baseline-120 --no-edit
   .venv/bin/pytest tests/ -q
   git push origin master
   ```
6. **Only now**, as the integration owner, write the consolidated update
   to `docs/HANDOFF.md` (replacing the 4 tracks' individual one-line
   updates with a single coherent section, same as Day 3's own
   integration step did) and to `STRATEGY.md` (recording the real
   `baseline-120` outcome against its §7 ladder entry — this is the one
   edit to `STRATEGY.md` this whole plan authorizes, and only after
   Track 4's real numbers exist).
7. State the actual next action in `docs/HANDOFF.md`'s "Exact next
   action" — this depends entirely on what Track 4's real run showed
   (`keep`: the next unstarted ladder rung becomes the candidate, per
   `STRATEGY.md` §7's dependency rules already encoded in `AGENTS.md`'s
   sequencing constraints; `investigate`/`revert`: name the specific
   follow-up needed before any later rung is attempted) — do not
   pre-decide this; it's a decision for the human owner informed by the
   real result, exactly as `docs/HANDOFF.md`'s existing precedent already
   treats `baseline-120` itself.
8. The 4 `feat/baseline-120-*` branches can stay or be deleted once
   `integration/baseline-120` is confirmed merged and pushed — human
   owner's call, not automatic.

## If a track didn't finish or has real failures

Don't block the other 3 on it. Merge the ones that are actually done and
green; leave the incomplete branch for a later round. Note this in
`docs/HANDOFF.md` rather than silently dropping it. If Track 4's Part B
(the real Colab run) specifically isn't done, land Parts 1–3 plus Track
4's Part A (notebook/schema work) and record `baseline-120`'s status as
"infrastructure complete, experiment not yet run" rather than either
`MERGED` or silently omitting it — see `IMPLEMENTATION.md`'s status
dictionary.

## Rollback

If a regression is found post-merge, prefer `git revert` of the specific
merge commit responsible over a broader reset — every track's change this
round is additive or a narrowly-scoped, test-covered fix (Track 1's
one-line `agent/my_agent.py` change being the most behavior-changing of
the four), so a clean revert should be possible without touching the
other 3 tracks' work. Do not force-push, delete branches, or run any
destructive `git` operation as part of this — the human owner decides
those, not this runbook.

## Kaggle / offline-package validation vs. real submission — kept separate

This integration produces no Kaggle-related change at all (`README.md`'s
"Kaggle / external gate," `IMPLEMENTATION.md`'s "External / Kaggle gate"
row). If a future stage adds offline package/schema validation ahead of
an actual submission, that validation step and the real submission action
must remain two distinct, separately-approved steps — a submission is
only "done" once Kaggle reports a terminal status with a receipt/score
recorded in the repository, per `AGENTS.md`'s Kaggle gate; this stage
produces neither.
