# IMPLEMENTATION.md — `baseline-120-reki-core` real-game validation

Living tracker. Update your own track's rows as you progress; the
integration owner updates the cross-track rows (Integration gate,
External/Kaggle gate, Final master SHA, Blockers that span tracks).

- **Stage:** `baseline-120-reki-core` real-game validation (4-track split)
- **Plan date:** 2026-08-05
- **Base master SHA:** `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb`
- **Plan directory:** `docs/superpowers/plans/parallel-baseline-120/`

## Status dictionary

`PLANNED` · `IN PROGRESS` · `BLOCKED` · `READY FOR REVIEW` · `MERGED` ·
`VALIDATED` · `REJECTED/ROLLED BACK`

All 4 tracks start `PLANNED` — none of this stage's code exists yet as of
this plan's writing (verified: `git status --short` clean, no
`feat/baseline-120-*` branch exists locally or on `origin` at plan-writing
time).

## Tracks

| # | Track | Owner (fill in) | Branch | Depends on | Status |
|---|---|---|---|---|---|
| 1 | Backend selection wiring | — | `feat/baseline-120-backend-wiring` | none | `PLANNED` |
| 2 | Real-game eval harness | — | `feat/baseline-120-eval-harness` | Track 1 (interface only, not physically) | `PLANNED` |
| 3 | Local regression & investigation | — | `feat/baseline-120-local-regression` | none | `PLANNED` |
| 4 | Colab execution + experiment record | — | `feat/baseline-120-colab-validation` | Track 1 + Track 2 (real integration) | `PLANNED` |

## Expected PRs

One PR per track, targeting `master` via the `integration/baseline-120`
branch (same pattern as Day 3's `integration/day3`) — see `INTEGRATION.md`.
No track merges directly to `master`.

## Test gates

| Gate | Requirement | Status |
|---|---|---|
| Pre-stage baseline | `.venv/bin/pytest tests/ -q` → 261 passed, 0 failed | `VALIDATED` (confirmed 2026-08-05 on base SHA) |
| Track 1 | New `tests/test_backend_selection.py` + `tests/test_config.py` addition green | `PLANNED` |
| Track 2 | Extended `tests/test_run_ablation.py` green, including one real-engine test | `PLANNED` |
| Track 3 | New `tests/test_real_game_regression.py` green, 25-game crash sweep clean | `PLANNED` |
| Track 4 | Extended `tests/test_build_colab_notebook.py` green | `PLANNED` |
| Post-merge full suite | All of the above stacked, 0 failed | `PLANNED` |

## Benchmark gates

| Gate | Requirement | Status |
|---|---|---|
| Local crash-safety sweep (Track 3) | All 25 public games, bounded steps, 0 unhandled exceptions | `PLANNED` |
| `cerebras_dev` full sweep (Track 4, via real harness) | Documented game sample, real `select_backend`-wired harness, results recorded as a labeled dev-lane proxy — token budget confirmed ample, preview-lifecycle risk accepted, model-identity mismatch still applies | `PLANNED` |
| Colab real-game run (Track 4) | Same documented game sample as the Cerebras sweep, real `google/gemma-4-31B-it`, real RHAE captured — this is the authoritative result | `PLANNED` |
| Proposed acceptance threshold (see `README.md` — explicitly a proposal, not a repository decision) | At least 1 sampled game reaches `GameState.WIN` with the real backend | `PLANNED` |

## Integration gates

See `INTEGRATION.md` for the full runbook. Summary:

| Step | Status |
|---|---|
| All 4 branches pushed, each track's own suite green | `PLANNED` |
| Merged into `integration/baseline-120` in order (Track 3 → 1 → 2 → 4) | `PLANNED` |
| Full suite green after each merge | `PLANNED` |
| `integration/baseline-120` merged into `master` | `PLANNED` |
| `docs/HANDOFF.md` + `STRATEGY.md` updated by integration owner | `PLANNED` |

## External / Kaggle gate

Explicitly separate from this stage — see `README.md`'s "Kaggle /
external gate" section. Not started, not scheduled by this plan, requires
a distinct human-owner approval per `AGENTS.md`'s Kaggle gate.

| Item | Status |
|---|---|
| Kaggle Day 1 smoke submission (unmodified starter) | Not started (carried over from Day 1, still open per `docs/HANDOFF.md`) |
| First full Kaggle compatibility run (per `AGENTS.md`'s Day 3 schedule target) | Not started |

## Blockers

None recorded at plan-writing time. Add rows here as they arise during
implementation — each blocker should name the track, the specific file/
decision it's blocked on, and who can unblock it.

## Evidence links

- Base-commit test run: `.venv/bin/pytest tests/ -q` → `261 passed in 1.19s` (this session, 2026-08-05, on `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb`).
- Empirical "before" measurement: `scripts/play_local.py --game ls20,vc33 --max-steps 50` → `Aggregate scorecard score: 0.0`, both games `levels_completed=0`, `state=GameState.NOT_FINISHED` (this session, 2026-08-05).
- 25-public-game list: `scripts/play_local.py --list` output (this session, 2026-08-05).

## Final master SHA

_(fill in once `integration/baseline-120` is merged to `master` and pushed)_

## Results (fill in after completion)

- Final test count:
- Crash-safety sweep result:
- `baseline-120` conclusion (`keep`/`revert`/`investigate`):
- Next stage recommended:
