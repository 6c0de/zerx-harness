# Developer 3 — Local regression & fallback-loop investigation

**Read `README.md` in this directory first** — shared context, especially
"A concrete, empirical finding this plan is built on," which is exactly
the behavior this track investigates in depth.

- **Track:** Local, no-GPU regression sweep + fallback-loop root-cause investigation
- **Base master SHA:** `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb`
- **Branch:** `feat/baseline-120-local-regression` (create from base SHA above)

## Purpose and expected outcome

This is the one track in this plan with **no dependency on any other
track** — see `README.md`'s dependency graph. You use the harness exactly
as it exists on `master` today (including its current backend bug, which
you treat as a known, already-flagged condition, not something you fix —
Track 1 owns that fix).

Two deliverables:

1. **A crash-safety sweep across all 25 public games** — proving (or
   disproving) that `baseline-100`/`baseline-110`'s "no regressions"
   promotion criterion actually holds against the *full* public game set,
   not just the `ls20`+`vc33` pair every prior session has used. Nobody in
   this project has run this before — `docs/HANDOFF.md` states plainly
   that "no track has yet been exercised against an actual game," and this
   session's own direct test only covered 2 of the 25 games.
2. **A root-cause investigation of the exact behavior this session
   observed**: 50 steps on `ls20`+`vc33`, 0 levels completed, every action
   `ACTION6`, no coordinate variation. Determine whether this is (a) fully
   explained by the missing backend wiring alone (i.e., once Track 1's fix
   lands, does real model reasoning break the loop, or is there a second,
   independent bug in the heuristic/fallback path itself that would still
   produce a stuck loop even with a working model backend?), and (b) if a
   second bug exists, root-cause it per `superpowers:systematic-debugging`
   before deciding whether to fix it.

## Commands to run before starting

```bash
git fetch origin
git checkout -b feat/baseline-120-local-regression 8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb
.venv/bin/pytest tests/ -q                          # confirm 261 passed, 0 failed
.venv/bin/python scripts/play_local.py --list       # confirm you can reach the 25-game list (read-only, no credentials)
```

## Files you own this round

- New file: `tests/test_real_game_regression.py` (or a name you prefer of
  similar shape — pick one and use it consistently across your plan file,
  tests, and status update).
- `docs/HANDOFF.md` — one-line status addition only.

## Conditionally: `zerx/heuristics.py` or `agent/my_agent.py`

Only if your investigation confirms a genuine, fixable bug (see below).
If the root cause turns out to live in a file another track owns this
round (`agent/my_agent.py` is Track 1's this round), **do not edit it
yourself** — document the finding precisely (file, line, mechanism) in
your status update and your plan file, and let the integration owner
sequence the fix with the owning track. If the root cause is in
`zerx/heuristics.py` (untouched by any other track this round), you may
fix it directly, with its own regression test.

## Do not touch

`zerx/model_backend.py`, `zerx/config.py`, `eval/run_ablation.py`,
`scripts/build_colab_notebook.py`, `agent/my_agent.py` (unless the
conditional case above applies, and even then coordinate first).

## What "investigate" means concretely

Reproduce this session's exact finding first, so you're working from a
confirmed starting point rather than a secondhand description:

```bash
.venv/bin/python scripts/play_local.py --game ls20,vc33 --max-steps 50
```

Then, using `superpowers:systematic-debugging`'s discipline (form a
hypothesis, find the smallest reproduction, verify by reading the actual
code path rather than guessing):

- Read `zerx/policy.py`'s `decide()` fallback chain closely: when
  `backend.generate()` fails every call, `parsed` stays `None`, so
  `decide()` falls to `if candidates and ActionName.ACTION6 in legal_actions:`
  — the **same top-ranked click candidate** every time, unless
  `rank_click_candidates`'s ranking itself changes between calls.
- Check whether `DeadSignatureTracker.record_outcome` is actually being
  called for these fallback-sourced decisions. Read
  `agent/my_agent.py`'s `_choose_action_inner`: the outcome-feedback call
  is gated on `self._pending_decision.target_object_label is not None`.
  Trace whether `Decision(source="fallback_heuristic", ...)` sets
  `target_object_label` (read `zerx/policy.py`'s `decide()` return value
  for that branch directly rather than assuming) — if it does, the
  penalty-recording path should be live; if some other decision source in
  the loop doesn't set it, that source's outcomes never reach
  `DeadSignatureTracker`, and the same candidate could keep winning
  indefinitely. This is a real, specific hypothesis to verify against the
  actual code, not an assumption to write up unverified.
- Separately check `zerx/heuristics.py`'s `rank_click_candidates` and
  `DeadSignatureTracker.penalty`: even if outcomes ARE being recorded,
  confirm the penalty actually changes the ranking order enough to matter
  within 50 steps, versus being too weak to visibly affect which candidate
  ranks first (`penalty_step: float = 0.35` — read the actual ranking
  math, don't guess whether that's enough to reorder a small candidate
  set).
- Determine whether this is fully explained by "no real model ever runs"
  (i.e., a heuristic-only fallback loop with a small, static candidate set
  and a real game that doesn't visibly change under `ACTION6` repeats is
  *expected* to look exactly like what was observed) — in which case this
  is not a bug, it is confirmation that Track 1's fix is the actual and
  complete remedy, and you should say so plainly rather than manufacture
  a second finding.

## Tests

`tests/test_real_game_regression.py`:

- **Crash-safety sweep:** for all 25 public games (from
  `arc.get_environments()`, matching `scripts/play_local.py`'s own
  `--list`/`--game` resolution pattern), construct `MyAgent` with
  `Config.from_env()`'s current (unfixed, hardcoded-`GemmaModelBackend`)
  behavior — this deliberately tests the harness *as it exists on this
  branch's base*, not a hypothetical fixed version — and run a bounded
  step count. Choose and document your actual step cap in your plan file
  (e.g. 20–30 steps per game keeps 25 games' wall-clock cost reasonable;
  the existing `verify-local` precedent uses 50 steps for 2 games).
  Assert: no unhandled exception escapes for any game, and each game
  reaches either a terminal `GameState` or the step cap. This is a real
  integration test against the live local engine — expect it to be
  slower than the rest of the suite; consider marking it so it can be
  skipped from the fast default run if your measured wall-clock cost is
  too high for every-commit use (your call — document the trade-off you
  chose, e.g. a `pytest.mark.slow_local_engine` marker, consistent with
  this repo's existing precedent of marking opt-in tests, like
  `pytest -m cerebras_live`, rather than always running them by default).
- **Fallback-loop characterization test(s):** whatever your investigation
  concludes, encode it as a test. If you find the root cause is fully
  explained by the missing backend (no second bug), write a test that
  documents and locks in the *current, known* behavior on `ls20` (e.g.
  "with `Config(backend='fake')` and no scripted responses, 20 consecutive
  `choose_action` calls against a real game never raise, and the returned
  actions are drawn only from `fallback_heuristic`/`fallback_deterministic`/`fallback_random`/`reset`
  sources" — read `.reasoning["source"]` off the returned `GameAction`,
  which `agent/my_agent.py` already sets). If you find a second, genuine
  bug in `zerx/heuristics.py`, write a failing test first (TDD), then fix
  it, matching this project's `superpowers:test-driven-development`
  discipline throughout.

## Verification commands

```bash
.venv/bin/pytest tests/ -q
.venv/bin/pytest tests/test_real_game_regression.py -v
```

## Expected outputs

- New `tests/test_real_game_regression.py`, with the crash-safety sweep
  and the fallback-loop characterization/regression test(s) described
  above.
- A written finding (in your own
  `docs/superpowers/plans/2026-08-05-baseline-120-local-regression.md`
  plan file) stating plainly: is the stuck-loop behavior fully explained
  by the missing backend wiring, or is there a second bug — and if the
  latter, exactly where, with the specific code path cited.
- If a second bug was confirmed and fixed: the fix itself, isolated and
  test-covered, in whichever file the ownership matrix and your own
  coordination note actually permits.

## Artifact and log locations

None beyond the test file and your plan-file writeup — this track
produces no run-time artifacts of its own.

## Performance / runtime bounds

Document the actual measured wall-clock time of your 25-game sweep in
your plan file (run it, don't estimate) — this number matters for whether
future sessions run it by default or opt-in.

## Edge cases

- A game that `arc.make()` can't construct (mirrors Track 2's same
  concern) — skip with a clear message, consistent with
  `scripts/play_local.py`'s own behavior, don't fail the whole sweep over
  one unavailable game.
- A game that completes (`GameState.WIN`) well before your step cap —
  that's a success case for the sweep, not a special case to handle
  differently.

## Failure-mode behavior

If the crash-safety sweep finds even one game that raises an unhandled
exception through `MyAgent.choose_action` (which `AGENTS.md`'s outer
`try/except` in `choose_action` should prevent, but this is exactly the
kind of thing that discipline needs empirical verification, not just
trust) — that is a real, reportable regression. Document it precisely
(game id, step number, exception type/message) rather than only noting
"one game failed."

## Definition of done

- Crash-safety sweep implemented, run, and its result (clean, or specific
  failures) documented.
- Fallback-loop investigation concluded with a clear, evidenced finding.
- If a second bug was found: fixed, with its own regression test, in a
  file you're actually permitted to touch (coordinate if not).
- `docs/HANDOFF.md` one-line status update.
- Your own plan file written before coding.

## PR checklist

- [ ] All 25 public games covered by the crash-safety sweep (not a subset).
- [ ] Measured (not estimated) wall-clock cost documented.
- [ ] Fallback-loop root cause stated explicitly, with the specific code path cited.
- [ ] If a bug was fixed: TDD (failing test committed before the fix), and the fix is in a file this track is permitted to own.
- [ ] Full suite green, count reported in PR description.
- [ ] No edits outside "Files you own this round" (or a documented, coordinated exception).

## Handoff format

Update `docs/HANDOFF.md` with branch, commit SHA, the crash-safety sweep's
pass/fail summary across all 25 games, measured wall-clock cost, and one
sentence stating your fallback-loop root-cause conclusion (this directly
informs whether Track 1's fix alone is sufficient, which Track 4 needs to
know before writing the final experiment record).

## Merge preconditions

Full suite green. Since this track has no code dependency on any other
track, it's safe to merge first per `INTEGRATION.md`'s order regardless of
the other 3 tracks' progress.

## Rollback approach

Your test file is new and additive; if the crash-safety sweep proves too
slow or flaky for regular runs after merge, the safe fix is adding/adjusting
an opt-in marker (matching this repo's `cerebras_live`-style precedent),
not reverting the whole track — only revert outright if a genuine defect
in the test itself (not the sweep's cost) is found.
