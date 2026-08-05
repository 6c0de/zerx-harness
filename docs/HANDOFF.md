# Project handoff

Copy this template's structure for each handoff entry (newest first, or one
file per handoff under `docs/handoffs/` if the history grows long — not
needed yet). See `docs/TEAM_WORKFLOW.md` for the 5-day schedule this feeds
into.

- Updated at: 2026-08-05
- Current owner: (local session, Claude Code — integration)
- Next owner: whoever picks `baseline-120-reki-core` validation (see "Exact
  next action" below) — not auto-started
- Branch: `master` (all 4 Day 3 parallel tracks merged in, sequentially,
  via `integration/day3`, per `INTEGRATION.md`)
- Commit: `f3e9e573498ad58df154be3fe5a5b667f51a6513` (merge of
  `integration/day3` into `master`, pushed to `origin/master`)
- Experiment ID: `baseline-100` (recorded, but flagged `investigate` — see
  `docs/superpowers/experiments/baseline-100.md`, per-game outcome wasn't
  captured by the notebook, only environment/setup validation)
- Config ID/hash: n/a — no scored experiment record exists yet
- Sprint day (1–5): Day 3 complete (4 parallel tracks merged); Day 3's
  Kaggle compatibility run (per `AGENTS.md`'s 5-day schedule) is still open
  — see "Exact next action"

## Objective

Day 1 (local-skeleton plan, 15 tasks), Day 2 (Colab Gemma-4-31B load,
`baseline-100`), and Day 3's 4 parallel tracks (`baseline-115`,
`baseline-130`, `exp-140`, `exp-150-duck-tools-ab`) are all complete and
merged into `master`. See "Parallel work split" below for per-track detail
and the integration record.

## Completed changes (Day 1 + Day 2, now on `master`)

**Day 1** — full model-free `zerx/` package (types, config, perception,
heuristics, memory, budget, model backend protocol, dev-only Cerebras
backend, secret scanner, JSON policy parsing + `decide()` orchestrator,
evidence-first transition ledger) plus a thin `agent/my_agent.py` harness
adapter wired to the real upstream ARC-AGI-3-Kaggle-Starter API. `choose_action`
has a top-level exception boundary. `scripts/build_notebook.py` bundles
`zerx/*.py` into the Kaggle submission notebook (never `zerx/backends/`),
gated by a secret-scan build check. Full detail:
`docs/superpowers/experiments/baseline-000.md`.

**Day 2** — `zerx/model_backend.py`'s `GemmaModelBackend` implemented for
real as an injectable vLLM OpenAI-compatible HTTP client (same pattern as
`zerx/backends/cerebras_dev.py`). `scripts/build_colab_notebook.py`
generates a Colab dev notebook (`notebooks/colab_gemma_smoke.ipynb`,
gitignored) satisfying `AGENTS.md`'s Colab gate. **Six real, independently
diagnosed infrastructure bugs were found and fixed** while actually
running this on a live Colab A100, each backed by the real error log, not
guessed:

1. `pip install vllm==0.11.0` predates Gemma 4's release by ~5 months and
   can't parse its `rope_scaling` config — bumped to `0.26.0`.
2. The Kaggle Models UI's model path
   (`google/gemma-4/Transformers/gemma-4-31b-it`) is NOT a valid Hugging
   Face Hub repo id — the real, loadable one (confirmed live against
   `huggingface.co/google/gemma-4-31B-it`) is `google/gemma-4-31B-it`
   (capital B).
3. Plain `pip install vllm` ignores the actual driver's CUDA version —
   switched to `uv pip install --torch-backend=auto`.
4. `--torch-backend=auto` alone wasn't enough — Colab's pre-existing torch
   install gets left in place and mismatches vLLM's freshly-installed CUDA
   extension; added `--reinstall` to force replacement (per vLLM's own
   docs: "install vLLM with a fresh new environment").
5. Confirmed the real attached GPU is an **A100-SXM4-80GB** (not 40GB —
   the earlier `nvidia-smi` reading was correct for a *different* Colab
   runtime shape before "High-RAM" was selected), so 4-bit quantization is
   unnecessary — bf16 (~61.4GB weights) fits with headroom. The notebook's
   default is still 4-bit (sized for a 40GB SKU); switch to bf16
   (`--quantization`/`--load-format` flags removed) for future runs
   confirmed on an 80GB card.
6. `scripts/build_colab_notebook.py` must be regenerated **after**
   committing a fix, not before/during the same command chain — otherwise
   the notebook's embedded `COMMIT_SHA` points at the previous commit,
   silently shipping a stale notebook. This session hit that exact
   mistake once; downstream sessions should regenerate as a separate,
   final step after any fix commit.

## Tests executed and results

`.venv/bin/pytest tests/ -q` (this integration was run on macOS; Windows
teammates use `.venv\Scripts\pytest.exe tests/ -q` — see
`docs/superpowers/experiments/baseline-000.md`'s "Windows-native
environment deviations"):

- `master` pre-integration: 136 passed, 0 failed.
- After merging `feat/baseline-115-exact-state-memory`: 159 passed.
- After merging `feat/exp-150-duck-tools-ab`: 203 passed.
- After merging `feat/exp-140-vlm-refinement`: 230 passed.
- After merging `feat/baseline-130-hypothesis-memory`: **261 passed, 0
  failed** — matches the pre-merge estimate (136 baseline + each track's
  own reported new-test count) with no dropped or duplicated tests.
- `master` post-merge (after fast-forwarding to `integration/day3`): 261
  passed, 0 failed. Pushed to `origin/master`.

Every new `Config` flag added by the 4 tracks
(`exact_state_suppression_on`, `duck_objects_on`, `candidate_count`,
`structured_memory_on`) still defaults to its inert value after all
merges — verified by grep against `zerx/config.py` after each merge step,
not just assumed.

## Colab state

- Account owner: (session owner, human)
- Notebook: `notebooks/colab_gemma_smoke.ipynb`, generated by
  `scripts/build_colab_notebook.py` (gitignored, regenerate locally)
- Git commit checked out (in the Colab run): `89126ecf3ea40e203567d5203669cc47ac35874c`
- GPU/backend profile: NVIDIA A100-SXM4-80GB, bf16, vLLM 0.26.0, model
  `google/gemma-4-31B-it`
- Status/results location: setup succeeded (real model reachable, no
  exception through the save-results cell); per-game play outcome not
  captured — see `docs/superpowers/experiments/baseline-100.md`'s "Known
  gap" section. **Re-running with the results-capture fix is a good
  candidate first task for whichever of the 4 tracks below finishes
  first**, since it's small and independent of all 4 tracks' actual scope.

## Cerebras development state

Not started. No `CEREBRAS_API_KEY` exists in this environment; no live
Cerebras call has ever been made — every `zerx/backends/cerebras_dev.py`
test injects a fake `http_post` or a literal string.

## Kaggle state

Not started. No `make submit`, no Kaggle CLI call, no notebook push.
**Still open** from Day 1's exit condition — needs explicit owner approval
before it happens, per `AGENTS.md`'s Kaggle gate. Not blocking the 4
parallel tracks below (none of them touch Kaggle).

## Known failures or risks (carried over, still real)

1. `zerx/backends/cerebras_dev.py`'s `platform` kwarg defaults to `"local"`
   and is never wired to the real `Config.platform` — inert today (nothing
   constructs `CerebrasDevBackend` outside its own tests). **Whichever
   track adds a backend-selection factory must forward
   `platform=config.platform` explicitly.**
2. No true rate-limit backoff in `CerebrasDevBackend.generate()`'s retry
   loop — inert until a live Cerebras test exists.
3. `parse_action(None, ...)` raises `AttributeError`, inert because
   `decide()` wraps the only real call site in `try/except Exception`.
4. `history` is computed in `agent/my_agent.py` and passed to
   `decide()`/`perceive()`, but `perceive()` ignores it — deliberate
   interface stability for future movement-delta perception.
5. `baseline-100`'s per-game outcome wasn't captured (see Colab state
   above) — small, independent follow-up.

## Parallel work split (Day 3, starting 2026-08-05)

Four people, each running their own Claude Code (Sonnet 5) session on
their own machine, all pushing to the same GitHub repo. Full prompts,
scope boundaries, and the shared-file etiquette that keeps 4 simultaneous
branches mergeable are written out in full, ready to paste, one file per
person, under `docs/superpowers/plans/parallel-day3/`:

| # | Track (STRATEGY.md ladder) | Branch | Status |
|---|---|---|---|
| 1 | `baseline-115-exact-state-memory` | `feat/baseline-115-exact-state-memory` | **merged to master** — 159/159 own-branch tests, `exact_state_suppression_on=False` default |
| 2 | `baseline-130-hypothesis` (structured memory) | `feat/baseline-130-hypothesis-memory` | **merged to master** — 167/167 own-branch tests, `structured_memory_on=False` default |
| 3 | `exp-140-vlm-refinement` (candidate/arbiter infra) | `feat/exp-140-vlm-refinement` | **merged to master** — 163/163 own-branch tests, `candidate_count=1` default |
| 4 | `exp-150-duck-tools` Variants A+B | `feat/exp-150-duck-tools-ab` | **merged to master** — 180/180 own-branch tests, `duck_objects_on=False` default |

All 4 tracks were chosen because none depends on another landing first
(unlike `baseline-125`, which needs `baseline-130` done, or `exp-200`+,
which needs `baseline-130` done) — see `STRATEGY.md` §7's table and the
"Deferred as isolated experiments" note in §4.

**Integration — done, 2026-08-05.** Followed
`docs/superpowers/plans/parallel-day3/INTEGRATION.md` exactly: merged into
`integration/day3` one branch at a time, in the specified order
(`baseline-115` → `exp-150-duck-tools-ab` → `exp-140-vlm-refinement` →
`baseline-130-hypothesis-memory`, smallest/least-invasive shared-file
touch first), full suite run after each merge (see "Tests executed and
results" above for the running count), then fast-forwarded `master` to
`integration/day3` and pushed.

Conflicts were exactly what `INTEGRATION.md` predicted — mechanical,
confined to `zerx/config.py` and `tests/test_config.py`, one "both sides
added lines in the same place" per merge (`exact_state_suppression_on` +
`duck_objects_on`, then `+ candidate_count`, then `+ structured_memory_on`
— all unique names, no field collision). Resolved by keeping every side's
added lines, in every case; no logic conflicts and no ambiguity requiring
a stop. `zerx/policy.py`'s `decide()` only changed on the `exp-140`
branch, as a single new `if config.candidate_count > 1:` branch with no
signature change — merged clean, no conflict. `agent/my_agent.py` picked
up both `baseline-115`'s and `baseline-130`'s independent, config-gated
comment-banner blocks — auto-merged clean, verified by reading the diff
directly rather than trusting the auto-merge.

## Exact next action

1. Per `STRATEGY.md` §7's ladder, the natural next step is validating
   `baseline-120-reki-core` (reflection + click proposals + soft failure
   memory) against real games — everything built in the 4 merged tracks
   is currently local, model-free, and unwired/off-by-default; no track
   has yet been exercised against an actual game. This is a
   recommendation for the human owner to schedule, not something this
   session is starting automatically.
2. Kaggle Day 1 smoke submission is still open — get explicit approval
   before running it, independent of the parallel tracks.
3. `baseline-100`'s results-capture gap (see Colab state above) — small,
   pick it up whenever convenient.
4. The 4 `feat/...` branches used for Day 3 are fully merged into
   `master` (see "Parallel work split" above) — safe to delete once the
   human owner confirms, not deleted automatically.

## Uncommitted or external artifacts

None tracked or required. `.venv/`, `vendor/ARC-AGI-3-Agents/`,
`environment_files/`, `notebooks/*.ipynb`, and `.superpowers/` (SDD
session ledger/briefs/reports) all exist locally and are gitignored — not
source of truth. No credentials of any kind exist in this environment (no
`CEREBRAS_API_KEY`, no Kaggle token) — every test that would need one
injects a fake instead.
