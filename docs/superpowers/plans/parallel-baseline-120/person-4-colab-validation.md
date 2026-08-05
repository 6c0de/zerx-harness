# Developer 4 — Colab execution, experiment record, and status docs

**Read `README.md` in this directory first** — shared context, the frozen
interface contracts, the dependency graph (your real, final deliverable is
gated on Track 1+2; your prep work is not), and the proposed acceptance
threshold, which is explicitly a proposal, not a repository decision.

- **Track:** Colab Gemma execution + `baseline-120` experiment record + status docs
- **Base master SHA:** `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb`
- **Branch:** `feat/baseline-120-colab-validation` (create from base SHA above)

## Purpose and expected outcome

`scripts/build_colab_notebook.py` currently generates a **smoke test**
notebook: it loads Gemma, plays exactly one game (`ls20`, 50 steps), and
its `save_results_cell` only records environment/setup metadata — not the
actual per-game outcome. `docs/superpowers/experiments/baseline-100.md`
already documents this exact gap in detail ("Known gap — per-game play
outcome not captured") and concludes `investigate`, not `keep`.

Your job is to turn this from a one-game smoke test into
`baseline-120`'s real validation run: extend the notebook to play a
documented sample of public games, capture their real outcomes (including
RHAE via `arc.get_scorecard()`, same mechanism Track 2's `run_games`
uses), and write the actual experiment record with a real
keep/revert/investigate conclusion — the first genuinely scored
`baseline-*` experiment this repository will have.

**Read the dependency graph in `README.md` before starting.** Your
notebook-structure and results-schema work does not depend on Track 1 or
2. Your final, authoritative Colab run and the written experiment record
depend on Track 1's `select_backend` actually existing (so `cerebras_dev`
is reachable through the real harness, not just standalone) and benefit
from Track 2's `run_games` (so the record format is consistent with
future ablation work). Say so plainly in your status updates rather than
claiming a premature "done."

**Cerebras dev-lane is the primary fast-iteration path for this track,
not an optional side-check.** The human owner has confirmed: (a) token
budget for `gemma-4-31b` on Cerebras is ample, not a rate/quota
constraint this stage needs to ration; (b) Cerebras's "preview" lifecycle
status (per `AGENTS.md`'s Cerebras development boundary — the model "may
be discontinued on short notice") is an accepted risk for this project,
not a reason to limit dev-lane use. This does **not** change anything
about the hard rule it sits next to: Cerebras is still never part of the
Kaggle submission runtime, and any behavior selected via Cerebras must
still be reproduced on the exact Gemma-4-31B backend in Colab before it
counts as validated (`AGENTS.md`, `STRATEGY.md` §2.6/§4) — a full,
ample-quota Cerebras sweep makes the *Colab* run cheaper and lower-risk
(catches prompt/logic/harness problems before spending Colab wall-clock),
it does not replace it. Model-identity mismatch (Cerebras's own
weight-only quantization vs. the exact deployed Gemma weights) is a
separate, unrelated fact from the quota/preview question and still
applies regardless of budget.

## Commands to run before starting

```bash
git fetch origin
git checkout -b feat/baseline-120-colab-validation 8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb
.venv/bin/pytest tests/ -q                         # confirm 261 passed, 0 failed
.venv/bin/python scripts/build_colab_notebook.py   # regenerate the notebook from your branch, confirms the script still runs cleanly before you change it
```

## Files you own this round

- `scripts/build_colab_notebook.py` — extend `smoke_game_cell` and
  `save_results_cell` (see below). Do not touch `PINNED_INSTALL`,
  `checkout_cell`, `env_print_cell`, or `start_vllm_cell` — those cells
  already work and are out of scope.
- New file: `docs/superpowers/experiments/baseline-120.md`.
- `tests/test_build_colab_notebook.py` — extend with tests for your new
  cell content (existing file, but no other track touches it this round).
- `docs/HANDOFF.md` — status update, and (only once your real run is
  complete) the "Exact next action" section.
- `notebooks/kernel-metadata.json` — **do not touch**; that file is
  Kaggle-submission-related and explicitly out of scope (see `README.md`'s
  "Kaggle / external gate").

## Do not touch

`zerx/model_backend.py`, `agent/my_agent.py`, `zerx/config.py`,
`eval/run_ablation.py`, `scripts/build_notebook.py` (the **Kaggle**
notebook builder — different file from the one you own), `STRATEGY.md`
(only the integration owner edits this, after your real numbers exist —
see `README.md`'s ownership matrix).

## Part A — work you can do immediately, no dependency

1. **Extend `smoke_game_cell`'s game list.** Currently hardcoded to a
   single game (`ls20`). Change it to a documented sample — this session's
   direct verification confirms 25 public games exist
   (`su15, bp35, wa30, tn36, cd82, g50t, ka59, cn04, dc22, vc33, re86,
   sp80, lf52, ft09, sb26, tr87, m0r0, r11l, sc25, ls20, tu93, lp85, s5i5,
   ar25, sk48`). `AGENTS.md`/`STRATEGY.md`'s "repeated seeds/configurations"
   and "per-game regressions" language argues for more than the existing
   2-game (`ls20`+`vc33`) precedent — pick a sample size you can justify
   given Colab's ~9-hour ceiling (`AGENTS.md`'s Kaggle-gate section notes
   this same ceiling; a Colab session doesn't have that exact limit, but a
   31B model's per-decision latency is real and multi-game runs at
   `max-steps=200` each add up) and document your reasoning and final list
   in `docs/superpowers/experiments/baseline-120.md`, not just in code.
2. **Extend `save_results_cell`** to capture, per game: `state`,
   `levels_completed`, `actions` (i.e. `agent.frames[-1]` /
   `agent.action_counter`, matching `scripts/play_local.py`'s own
   per-game summary line — this is the exact fix
   `docs/superpowers/experiments/baseline-100.md` already asked for) plus
   the RHAE-style score from `arc.get_scorecard()`'s `EnvironmentScorecard`
   for that game (same source Track 2's `run_games` reads — see
   `README.md`'s frozen interface for the exact object shape:
   `EnvironmentScorecard.environments`, each a `EnvironmentScoreList` with
   `.score`/`.actions`/`.levels_completed`, matched by `.id`). Save the
   full per-game breakdown to the JSON written to Drive, not just an
   aggregate.
3. **A fast, GPU-free prompt/parse sanity check, independent of Track 1/2:**
   `zerx/backends/cerebras_dev.py`'s `CerebrasDevBackend` is directly
   constructible today without going through the (currently broken)
   `agent/my_agent.py` wiring — write a small, standalone script (not
   part of the Kaggle/Colab notebook, e.g. a scratch script you keep local
   or a short new test-adjacent script if you find it genuinely reusable)
   that constructs `CerebrasDevBackend` directly and calls
   `zerx/policy.py`'s `build_prompt`/`parse_action` against it, using
   `CEREBRAS_API_KEY` from your own environment (never commit it, never
   put it in a notebook cell, per `AGENTS.md`'s Cerebras development
   boundary — this rule is unaffected by quota/tier). This step only
   exercises the prompt/JSON-schema/parse path in isolation (no real game
   loop, no heuristics, no memory) — it's a cheap sanity check before Part
   B's full sweep, not a substitute for it. Required this round (token
   budget is confirmed ample) — if you genuinely cannot get a key, say so
   explicitly in your status update rather than silently skipping it.

## Part B — gated on Track 1 (Track 2 improves it, doesn't block it)

4. **Full `cerebras_dev` sweep across your documented game sample, via the
   real harness.** Once Track 1's `select_backend` exists (merge or pull
   their branch directly — see `README.md`'s dependency graph on
   fetching a track's branch ahead of the official merge order), run your
   full game sample through the real `agent/my_agent.py`/`decide()` loop
   with `Config(backend="cerebras_dev", platform="local")` — either via
   `scripts/play_local.py` with `ZERX_BACKEND=cerebras_dev` set, or via
   Track 2's `run_games` if it's available to you yet. This is the real
   deliverable Part A's step 3 was only a cheap preview of: the actual
   click-candidates/reflection-memory/soft-failure-evidence loop, running
   against real games, with a real (if not Kaggle-identical) model doing
   the reasoning — fast and inexpensive relative to Colab, and now the
   primary way this stage gets its first real signal on whether
   `baseline-120`'s core loop does anything better than the confirmed
   0.0/`ACTION6`-loop reference. Record these results too (same fields as
   the Colab run below) — they inform, but do not substitute for, the
   Colab conclusion.
5. Regenerate the notebook (`.venv/bin/python scripts/build_colab_notebook.py`)
   after Track 1's fix has landed, and change `smoke_game_cell`'s backend
   setup to actually exercise the fixed path — at minimum confirm
   `ZERX_BACKEND=gemma_local` still resolves correctly through Track 1's
   `select_backend` (it should, since `gemma_local` already mapped
   correctly even before the fix — see `README.md`'s empirical finding).
6. **Run the actual notebook on Colab** (A100 or L4, per `AGENTS.md`'s
   Colab gate) reproducing the **same** game sample step 4 just ran on
   Cerebras (`AGENTS.md`'s hard rule: anything indicated by a Cerebras
   result must be reproduced on the exact Gemma-4-31B backend before it
   counts) with the real `google/gemma-4-31B-it` backend. Download the
   results JSON from Drive.
7. **Write `docs/superpowers/experiments/baseline-120.md`**, following
   the exact field structure `docs/superpowers/experiments/baseline-100.md`
   already established (date, base commit, model, GPU, precision, backend,
   game(s), environment setup outcome, per-game results, and a
   `keep`/`revert`/`investigate` conclusion per `STRATEGY.md` §7.1 — do
   not skip the conclusion field, and do not write `keep` unless the
   evidence genuinely supports it per `README.md`'s proposed threshold or
   your own better-justified replacement for it). Include both the
   `cerebras_dev` sweep (step 4, labeled explicitly as a dev-lane proxy
   result, never as the `baseline-120` score itself) and the real Colab
   Gemma run (step 6, the authoritative number) side by side, so a reader
   can see whether the two agreed — a real disagreement between them
   (e.g. Cerebras shows progress, Gemma doesn't) is itself worth recording
   as a model-identity-mismatch data point, not something to paper over.

## Tests

`tests/test_build_colab_notebook.py` (extend):

- The generated notebook's `smoke_game_cell` source contains your full
  game sample (not just `ls20`).
- The generated notebook's `save_results_cell` source references
  `arc.get_scorecard()` (or however you actually implement the capture —
  assert against your real implementation, not this description) and no
  longer contains only the old environment-metadata-only result dict.
- No test requires an actual Colab/GPU environment — you're testing the
  *generator script's output*, exactly like the existing tests in this
  file already do (they check generated cell source text, not live
  execution).

## Verification commands

```bash
.venv/bin/pytest tests/ -q
.venv/bin/pytest tests/test_build_colab_notebook.py -v
.venv/bin/python scripts/build_colab_notebook.py   # regenerate after every code change, per docs/HANDOFF.md's known gotcha: commit first, THEN regenerate, or the embedded COMMIT_SHA goes stale
```

## Expected outputs

- Extended `scripts/build_colab_notebook.py`, generating a notebook that
  plays your documented game sample and captures real per-game results.
- New `docs/superpowers/experiments/baseline-120.md` with real numbers
  (once Part B is complete) and an explicit conclusion.
- `docs/HANDOFF.md` updated with the real outcome and a revised "Exact
  next action" pointing at whatever `STRATEGY.md` §7 rung is genuinely
  next given your result (do not pre-decide this now — it depends on
  what the real run shows).

## Artifact and log locations

- Generated notebook: `notebooks/colab_gemma_smoke.ipynb` (gitignored,
  regenerate locally — same as today).
- Raw Colab results JSON: Google Drive (`/content/drive/MyDrive/...`,
  same pattern as `baseline-100`) — download a copy and reference its
  content in `docs/superpowers/experiments/baseline-120.md` rather than
  only linking to Drive, so the record survives independent of Drive
  access.

## Performance / runtime bounds

A 31B model's per-decision latency plus your chosen game sample's step
count determines total Colab wall-clock time — estimate this before
starting the real run (e.g. from `baseline-100`'s single-game timing, if
recorded) and record your estimate vs. actual in the experiment doc. If
your sample risks exceeding a reasonable single Colab session, reduce the
sample rather than let the run silently exceed Colab's own session limits
— document that trade-off if you make it.

## Edge cases

- A game where the model produces exclusively malformed/unparseable JSON
  for the entire run — this is itself a valid, recordable outcome (high
  `invalid_outputs`/`repairs` in the experiment record), not a run to
  discard.
- A game that completes near-instantly (`GameState.WIN` in very few
  actions) — record it as-is; do not treat an unexpectedly high RHAE
  (capped at 115 per `arc_agi`'s own formula) as suspicious without
  investigating, but do note it plainly if it looks like an outlier.

## Failure-mode behavior

If the vLLM server fails to start or the model fails to load on Colab
(both have happened before, per `docs/HANDOFF.md`'s "Six real,
independently diagnosed infrastructure bugs" from Day 2) — that is an
`investigate` or environment-blocker outcome for
`docs/superpowers/experiments/baseline-120.md`, not a reason to silently
fall back to a smaller/easier claim. Record the actual failure, matching
this repository's existing discipline (`baseline-100.md`'s own honest
"environment/packaging validated, pipeline result unmeasured" conclusion
is the model to follow when something doesn't fully work).

## Definition of done

- Part A complete and tested regardless of Track 1/2's status, including
  the standalone Cerebras prompt/parse sanity check (step 3).
- Part B complete once Track 1 is available: the full `cerebras_dev`
  sweep (step 4) run and recorded, a real Colab run executed (step 6),
  `docs/superpowers/experiments/baseline-120.md` written with both
  results and a real conclusion, `docs/HANDOFF.md` updated.
- Your own `docs/superpowers/plans/2026-08-05-baseline-120-colab-validation.md`
  plan file, written before coding.

## PR checklist

- [ ] Game sample documented and justified (not an arbitrary, unexplained count).
- [ ] `save_results_cell` captures real per-game outcome + RHAE, not just environment metadata.
- [ ] Full `cerebras_dev` sweep (step 4) run against the real harness and its results recorded, clearly labeled as a dev-lane proxy, before the Colab run.
- [ ] Colab run reproduces the same game sample the Cerebras sweep used.
- [ ] `docs/superpowers/experiments/baseline-120.md` has a real, evidenced `keep`/`revert`/`investigate` conclusion — not a placeholder — and states the conclusion using the Colab (Gemma) result, not the Cerebras proxy result.
- [ ] `docs/HANDOFF.md`'s "Exact next action" reflects the real outcome, not a pre-written assumption.
- [ ] Full local suite green (notebook-generation tests only — no live-Colab test exists or is expected).
- [ ] No edits outside "Files you own this round," and `STRATEGY.md` is untouched by this branch (integration owner's job).

## Handoff format

Update `docs/HANDOFF.md` with: branch, commit SHA, real Colab run's GPU
type/dtype/model revision (mirroring `baseline-100.md`'s fields exactly),
game sample and result summary for both the `cerebras_dev` sweep and the
Colab Gemma run, and the conclusion (`keep`/`revert`/`investigate`,
decided from the Colab result). If Part B could not be completed this
session (e.g. no Colab access at hand-off time but the Cerebras sweep is
done), say so explicitly, report the Cerebras sweep's own findings as a
provisional signal, and hand off cleanly — do not fabricate placeholder
Colab numbers or treat the Cerebras result as if it were the Colab one.

## Merge preconditions

Full local suite green (notebook-generator tests). Merges last per
`INTEGRATION.md`'s order, after Track 1 and Track 2 — confirm your
notebook's backend wiring actually resolves through Track 1's
`select_backend` before treating your real run as final.

## Rollback approach

The notebook-generator script change is isolated and additive; if the
real Colab run's conclusion is `revert` (i.e. `baseline-120` as measured
doesn't clear even a modest bar), that is a valid, recorded outcome, not
something to walk back — `STRATEGY.md` §7.1 explicitly treats `revert`
and `investigate` as legitimate conclusions. Only revert the *code* (the
notebook-generator change itself) via `git revert` if a defect is found
in the generator script, not in response to the experiment's own result.
