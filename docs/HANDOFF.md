# Project handoff

Copy this template's structure for each handoff entry (newest first, or one
file per handoff under `docs/handoffs/` if the history grows long — not
needed yet). See `docs/TEAM_WORKFLOW.md` for the 5-day schedule this feeds
into.

- Updated at: 2026-08-06
- Current owner: (local session, Claude Code — `baseline-120-followups` +
  `policy-prompt-legal-budget` + `colab-ready` integration)
- Next owner: whoever gets the push-to-`origin/master` go-ahead, or picks up
  an item from "Exact next action" below — not auto-started
- Branch: `master`, locally **33 commits ahead of `origin/master`** after
  this round's integration. All three source branches
  (`feat/baseline-120-followups`, `feat/policy-prompt-legal-budget`,
  `integration/baseline-120-colab-ready`) are merged in, sequentially, via
  `integration/baseline-120-followups`, per
  `docs/superpowers/plans/2026-08-06-baseline-120-followups-integration.md`.
  The earlier `baseline-120-reki-core` 4-track integration (below) was
  already on `origin/master` before this round started.
- Commit: `24aefc6` (merge of `integration/baseline-120-followups` into
  `master`), local only as of this update — **not yet pushed to
  `origin/master`**, pending explicit human-owner confirmation (see "Exact
  next action")
- Experiment ID: `baseline-120` remains flagged `investigate` (see
  `docs/superpowers/experiments/baseline-120.md`) — this round lands
  infrastructure (visualizer, trace export) and a real fix (the
  `build_prompt()` legal-actions gap), not a new experiment result;
  `baseline-100` remains `investigate` too, unchanged
- Config ID/hash: n/a — no new model-in-loop sweep was run this round; see
  "Exact next action" for the recommended next one
- Sprint day (1–5): still Day 3 by commit-date timeline. Both of the human
  owner's stated prerequisites for the authoritative Colab run (a
  visualizer, and the `build_prompt()` legal-actions fix) are now on
  `master` — see "Exact next action" for what's actually next

## Objective

Day 1 (local-skeleton plan, 15 tasks), Day 2 (Colab Gemma-4-31B load,
`baseline-100`), and Day 3's 4 parallel tracks (`baseline-115`,
`baseline-130`, `exp-140`, `exp-150-duck-tools-ab`) are all complete and
merged into `master`. See "Parallel work split" below for per-track detail
and the integration record. `baseline-120-reki-core`'s own 4-track
validation is also now merged (infrastructure complete, dev-lane
investigated, authoritative Colab run explicitly postponed) — see
"baseline-120-reki-core — integration summary" below.

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

**`baseline-120` integration, 2026-08-06** (this update), following
`docs/superpowers/plans/parallel-baseline-120/INTEGRATION.md`'s merge
order (smallest/least-invasive first): each of the 4 branches was first
verified individually green, then merged one at a time into
`integration/baseline-120`, branched from `master` at `220b58e`:

- Track 3 (`feat/baseline-120-local-regression`) alone: 288 passed.
- + Track 1 (`feat/baseline-120-backend-wiring`): 297 passed (272 passed,
  25 deselected with `-m "not slow_local_engine"` for fast iteration).
- + Track 2 (`feat/baseline-120-eval-harness`): 300 passed (275 passed,
  25 deselected, fast filter); `tests/test_run_ablation.py`'s real-engine
  test explicitly re-run and confirmed to now exercise the fixed backend
  path.
- + Track 4 (`feat/baseline-120-colab-validation`, which had already
  merged Track 1 into itself mid-round to build against `select_backend`
  — its identical commits collapsed cleanly, no duplication): **308
  passed, 0 failed** — the full, final count (261 pre-`baseline-120` +
  27 Track 3 + 9 Track 1 + 3 Track 2 + 8 Track 4 net-new [Track 4's own
  branch added 9 tests to `tests/test_build_colab_notebook.py`/-1 for one
  superseded single-game test, +1 new Cloudflare-WAF regression test]).
- Merged into `master` locally: **308 passed, 0 failed**, confirmed again
  — not yet pushed (see top-of-file "Commit" note).

One notable, expected behavior change in this round: the 25-game
crash-safety sweep (`tests/test_real_game_regression.py`) ran in
**~23 minutes** on Track 3's own branch (dominated by repeated
connection-refused retries against the then-still-hardcoded
`GemmaModelBackend` pointed at an unreachable `localhost:8000`) but in
**~20 seconds** once merged alongside Track 1's fix — `Config.backend`
defaults to `"fake"`, so `select_backend` now constructs
`FakeModelBackend(responses=[])`, which raises immediately in-process
with no network I/O. Same assertions (no unhandled exception, reaches a
terminal state or the step cap), same pass count, just via the fast
fallback path instead of the slow one — a genuine side effect of Track
1's fix landing, not a weakened test.

Conflicts across all 4 `baseline-120` merges were exactly what
`INTEGRATION.md` predicted: mechanical, confined to `docs/HANDOFF.md`
(every track appending its own status paragraph in the same place),
resolved by keeping every track's content, never picking one side.
`zerx/model_backend.py`, `zerx/config.py`, `agent/my_agent.py`, and
`tests/test_backend_selection.py`/`tests/test_config.py` (all Track
1-owned this round) merged with **zero conflicts**, confirming the
ownership matrix held — no other track edited a file it didn't own.
`grep -n "GemmaModelBackend(self._config.model_revision)"
agent/my_agent.py` returns nothing post-Track-1-merge, confirming the old
hardcoded construction is actually gone. `zerx/config.py`'s
`gemma_base_url` field appears exactly once, defaulting to the original
hardcoded URL.

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

## Colab/Kaggle quantization decision (2026-08-06, branch `feat/baseline-120-8bit-quant-parity`)

The "Colab state" above (bf16, A100-80GB) predates this decision and its
`--quantization`/`--load-format` flags are now stale for future runs. Human
owner decision, 2026-08-06: Colab must load the model at the same precision
Kaggle will actually use, not whatever precision the attached Colab card
happens to have headroom for — otherwise a Colab validation result doesn't
reflect what Kaggle will actually deploy (`AGENTS.md`/`docs/TEAM_WORKFLOW.md`:
Kaggle is the deployment source of truth, Colab results are provisional).

- Kaggle's RTX Pro 6000 has 48GB VRAM — bf16 (~61.4GB weights) does not fit;
  it needs quantization regardless of what Colab needs.
- `scripts/build_colab_notebook.py` switched from its previous default
  (4-bit bitsandbytes, sized for a since-superseded 40GB assumption) to
  **8-bit dynamic FP8** (`--quantization fp8`), not bf16, so Colab and
  Kaggle run comparable precision.
- Verified against vLLM's own docs (fetched 2026-08-06, Context7 MCP was
  unavailable this session so this went through direct doc fetches
  instead): vLLM's bitsandbytes in-flight quantization only supports 4-bit
  (nf4) from an unquantized checkpoint — there is no in-flight 8-bit
  bitsandbytes mode (`docs.vllm.ai/en/stable/features/quantization/bnb/`).
  The real 8-bit path is vLLM's dynamic FP8 quantization: weights to
  FP8_E4M3 (~1 byte/param, ~31GB total for this model), no calibration
  data, no bitsandbytes package needed
  (`docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/`).
- A100 (Ampere, compute capability 8.0) runs FP8 as weight-only W8A16 via
  the FP8 Marlin kernel (below the >=8.9 threshold for full W8A8 activation
  quantization) — correct weights and memory footprint, but the docs note
  limited latency gains on this card. Kaggle's RTX Pro 6000 is a newer
  architecture and may see real W8A8 speedup instead; that throughput
  difference across the two cards is expected and does not affect the
  precision parity this change is actually for.
- The saved result JSON now records `"quantization"` explicitly (previously
  only `"dtype"` was recorded, so a saved result couldn't distinguish a
  4-bit run from a bf16 run after the fact).
- **Not done yet:** Kaggle's own vLLM launch config doesn't exist anywhere
  in this repo (Kaggle deployment work hasn't started — see "Kaggle state"
  below). When it's built, it must pass the same `--quantization fp8` flag
  — the parity this change establishes only holds once both sides actually
  use it.
- `notebooks/colab_gemma_smoke.ipynb` must be regenerated
  (`python scripts/build_colab_notebook.py`) after this fix is committed,
  per known-issue 6 above, so its embedded `COMMIT_SHA` points at a commit
  that actually contains this change.
- Tests: `tests/test_build_colab_notebook.py` updated (one test replaced,
  two added), 25/25 passing. Full fast suite (`pytest tests/ -q -m "not
  slow_local_engine"`): 270 passed; 2 failed + 3 collection errors, all
  five pre-existing `ModuleNotFoundError` for `arc_agi`/`arcengine` (this
  checkout has no `.venv` — `make setup` was never run this session) —
  unrelated to this change, not investigated further here.

This branch has no file overlap with the two other open
`baseline-120`-followup branches (`feat/baseline-120-followups`:
README/visualizer/trace-export; `feat/policy-prompt-legal-budget`:
`build_prompt()` legal-actions fix) — all three are independent and should
be merged and tested together before the next authoritative Colab run, per
the human owner's instruction.

## Cerebras development state

No `CEREBRAS_API_KEY` exists in **this Claude Code session's own tool
environment** — every `zerx/backends/cerebras_dev.py` test still injects a
fake `http_post` or a literal string, and this remains true after
`baseline-120`. However, a live Cerebras call **has now been made**, by
the human owner directly, in their own terminal (separate process, own
credential — never shared with this session, per `AGENTS.md`'s
credential-isolation rule): Track 4's dev-lane sweep (2026-08-06), 640+
real `gemma-4-31b` calls across 8 games, no rate-limit failures observed.
That run also found and fixed a real Cloudflare-WAF bug (default
`urllib` User-Agent triggering an HTTP 403/error-1010 block, unrelated to
credentials) — see "Known failures or risks" below, item 2's updated
wording, and `docs/superpowers/experiments/baseline-120.md` for full
detail.

## Kaggle state

Still no submission, no notebook push, no Kaggle CLI call — Day 1's exit
condition remains open, and every push needs explicit owner approval per
`AGENTS.md`'s Kaggle gate.

**Updated 2026-08-06, branch `feat/kaggle-p0-model-attach` (Phase A of
`docs/superpowers/specs/2026-08-06-kaggle-p0-model-attach-design.md`).**
Four of the five blockers that stood between this repo and a scored
submission are now closed on that branch; the fifth (actually serving a
model) is deliberately deferred to Phase B, pending a measurement.

Closed by Phase A:

- `scripts/build_notebook.py`'s `ACCELERATOR` is `"rtx6000"`, not the
  starter's default `"t4"` (2x16GB, too small for this model at any
  precision we would run).
- `notebooks/kernel-metadata.json` carries a real kernel id
  (`enzeceb/arc-prize-2026-arc-agi-3-starter`) instead of the
  `REPLACE_WITH_YOUR_USERNAME` placeholder that `make submit` refuses to
  push, and a non-empty `model_sources`. `main()` now syncs both
  `enable_gpu` and `model_sources` from constants, so the metadata file
  cannot drift from what the run cell expects.
- The run cell exports `ZERX_BACKEND=gemma_kaggle`, `ZERX_PLATFORM=kaggle`,
  and `ZERX_GEMMA_BASE_URL` (command-line prefix, not the `.env` file —
  `main.py` runs in a separate process and this does not depend on when
  the framework calls `load_dotenv()`), and runs a **readiness gate**
  before gameplay: it resolves `Config.from_env()` / `select_backend()`
  in-process and raises `SystemExit` if the result is a
  `FakeModelBackend` or if `KAGGLE_MODEL_DIR` is unset/missing. In-process
  on purpose — IPython swallows a shell command's non-zero exit, so only a
  real Python exception stops the notebook.
- Local environment on the Windows owner's machine: `.venv` created,
  `arc-agi`/`kaggle`/`pytest`/`numpy` installed, framework cloned and
  slimmed. `make` is not installed there and the only interpreter is
  Python 3.14, so the commands from `baseline-000.md`'s "Windows-native
  environment deviations" were run directly; the `Makefile` is untouched.
  Note `pygame` has no Python 3.14 wheel and fails to build from source on
  that machine — it is not needed for the Kaggle path, but
  `scripts/visualize_play.py` cannot run there.

Not closed, by design — see "Why Phase B is deferred" below.

## Known failures or risks (carried over, still real)

1. ~~`zerx/backends/cerebras_dev.py`'s `platform` kwarg defaults to
   `"local"` and is never wired to the real `Config.platform`~~ **Fixed**
   on `feat/baseline-120-backend-wiring` — `zerx/model_backend.py`'s new
   `select_backend(config)` factory constructs the backend named by
   `config.backend` and forwards `config.platform` to `CerebrasDevBackend`
   explicitly; `agent/my_agent.py`'s `MyAgent.__init__` now calls it
   instead of hardcoding `GemmaModelBackend`. See
   `docs/superpowers/plans/2026-08-05-baseline-120-backend-wiring.md`.
2. No true rate-limit backoff in `CerebrasDevBackend.generate()`'s retry
   loop. **Updated 2026-08-06:** a live Cerebras test now exists (Track
   4's `baseline-120` dev-lane sweep — 640+ real calls across 8 games,
   2026-08-06) and hit no rate-limit failures, so this is no longer
   purely theoretical exposure — but the backoff gap itself remains
   unaddressed; this item stays open until backoff is actually
   implemented and tested.
3. ~~`parse_action(None, ...)` raises `AttributeError`, inert because
   `decide()` wraps the only real call site in `try/except Exception`.~~
   **SOLVED** on `newest-update` (2026-08-06) — `parse_action` and
   `_extract_json_object` both reject a non-`str` input up front and return
   `None`, matching the function's documented "never raises" contract. It
   was only ever inert for `decide()`'s call site; `zerx/candidates.py`'s
   `generate_candidates` and any future caller got the crash. Test:
   `tests/test_policy_parse.py::test_parse_action_returns_none_for_non_string_input`.
4. `history` is computed in `agent/my_agent.py` and passed to
   `decide()`/`perceive()`, but `perceive()` ignores it — deliberate
   interface stability for future movement-delta perception.
   **Updated 2026-08-06 (repo audit, ARC-AUDIT-013):** re-confirmed still
   true on `master`, `feat/baseline-120-followups`, and
   `integration/baseline-120-colab-ready` — `perceive()`'s signature still
   documents `history` as accepted-but-unused. Two consequences worth
   recording beyond "deliberate": (a) it costs four `_to_game_frame`
   conversions per action that are then discarded, and (b) it means the
   agent has **no state-delta perception at all** — it cannot see what its
   own last action changed, which is the core evidence loop ARC-AGI-3
   rewards. `zerx/scene.py`'s `compare_frames()`/`classify_transition()`
   already implement exactly this and are unwired (see ARC-HANDOFF-003).
   Still correctly classified as planned future work (`exp-150`), not a
   regression — listed here so the cost is visible, not to reopen it.
5. `baseline-100`'s per-game outcome wasn't captured (see Colab state
   above) — small, independent follow-up.
6. **`zerx/policy.py`'s `build_prompt()` never lists the actual legal
   action names in the prompt text** — only a literal `<ACTION_NAME>`
   placeholder — confirmed root cause of `baseline-120`'s dev-lane sweep
   flat `0.0` result (Track 4, 2026-08-06): real `gemma-4-31b` calls
   succeeded (HTTP 200, real reasoning attempted), but the model invented
   invalid action names (`"ACTION0"`, `"WAIT"`) with no way to know the
   real vocabulary, so every decision fell through to the
   deterministic/heuristic fallback chain — indistinguishable from the
   pre-existing missing-backend "before" reference by symptom alone
   (dominant single action per game), even though the underlying cause
   this time is a prompt-design gap, not a broken connection. Fix
   candidate: add a `legal_actions` parameter to `build_prompt()`'s
   signature, render them in the prompt text, re-run the same 8-game
   sweep. See `docs/superpowers/experiments/baseline-120.md` for full
   detail. **Independently reproduced live, 2026-08-06,** using
   `feat/baseline-120-followups`' new visualizer against a real
   `cerebras_dev` run on `ls20` with a valid API key: the model returned
   a perfectly well-formed `{"action": "ACTION6", "data": {"x": 36, "y":
   15}}`, but `ls20` never has `ACTION6` legal, so `parse_action()`
   correctly rejected it and every step fell through to
   `fallback_deterministic` — same symptom, now confirmed via a second,
   independent live run rather than only Track 4's original sweep. Fix
   already exists, not yet merged: `feat/policy-prompt-legal-budget`'s
   `e402a0d fix(policy): surface legal actions and budget signal in
   build_prompt` — see
   `docs/superpowers/plans/2026-08-06-baseline-120-followups-integration.md`.

   **Updated 2026-08-06 — PARTIALLY RESOLVED on
   `feat/policy-prompt-legal-budget` (commit `e402a0d`), not yet on
   `master`.** The code half of the fix candidate is done exactly as
   described: `build_prompt()` now takes `legal_actions` (and `budget`),
   renders `Legal actions this turn: ...`, and closes by binding the model
   to that list; both `decide()` call sites pass them. Verified on that
   branch against every other branch in the repo — no other branch
   contains this fix.

   *Evidence so far (key-free mechanism A/B, real local engine, real
   `ls20` + `vc33` frames, 25 steps/game, 50 decisions/arm).* A simulated
   model constrained to name only actions it can actually read in the
   prompt — the precise capability this fix adds — was run against a
   master-shaped prompt and the fixed prompt:

   | Arm | `source="model"` | fallback |
   |---|---|---|
   | master prompt (no legal-actions line) | 0 | **50 (100%)** |
   | fixed prompt (`legal_actions` rendered) | **50 (100%)** | 0 |

   The master arm reproduces Track 4's recorded symptom exactly (every
   decision falls through to the fallback chain). This confirms the
   information channel was closed and is now open. **It does not measure
   whether real `gemma-4-31b` scores better** — by construction the
   simulated model always picks a legal name, which a real model may not.

   *Still outstanding — do not close this item until done:* re-run the
   original 8-game `cerebras_dev` sweep (`ls20, vc33, su15, tn36, ka59,
   lf52, tr87, sc25`) and compare against `baseline-120`'s flat `0.0`
   reference. All 8 games are available to the local engine; the only
   blocker is that `CEREBRAS_API_KEY` was not present in the environment
   when this was attempted. Note also that `MAX_ACTIONS` silently caps at
   80 steps/game (open item 7 below), so the re-run must record its actual
   step count rather than the requested one.

   **Updated 2026-08-06 — FULLY RESOLVED, now on `master`.** `e402a0d`
   merged into `master` via the "Integration" section above (merge commit
   `f0c68f7`); `zerx/policy.py`'s `build_prompt()` on `master` now takes
   and renders `legal_actions`/`budget`, verified directly post-merge. The
   re-run item above is still genuinely outstanding — it's now the
   leading candidate in "Exact next action" below.
7. ~~**`scripts/play_local.py`'s `MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS,
   args.max_steps)` can only ever *lower* the step cap**, never raise it
   above `MyAgentCls`'s existing default (80, inherited from the vendored
   base `agents.agent.Agent` class)~~ — confirmed by Track 4's dev-lane
   sweep (2026-08-06): `--max-steps 100` silently capped the run at 80
   steps/game, not 100. Recorded as the actual, accurate step count in
   `docs/superpowers/experiments/baseline-120.md` rather than the
   originally intended one.
   **SOLVED** on `newest-update` (2026-08-06), together with the much
   larger problem underneath it (ARC-HANDOFF-007 below): the cap is now a
   `Config` field, and every caller sets `ZERX_MAX_ACTIONS` instead of
   poking the class attribute. Fixed in `scripts/play_local.py`,
   `eval/run_ablation.py`'s `run_games`, `scripts/build_colab_notebook.py`'s
   `smoke_game_cell`, and `scripts/visualize_play.py` (that one was already
   an instance-level assignment and worked, but its `min()` was reduced to
   a plain assignment for the same reason). Verified live:
   `scripts/play_local.py --game ls20 --max-steps 120` now runs 121
   actions, not 81.
8. ~~**`scripts/play_local.py:114` crashes with `UnicodeEncodeError` on
   Windows non-UTF8 consoles**~~ **SOLVED** on `newest-update`
   (2026-08-06) — the per-game summary line now prints ASCII `->` instead
   of `→`, so the multi-game loop no longer aborts after the first game on
   a `cp1254` console. Original description follows.

   Crashes when printing multi-game summaries — the
   final per-game summary line hardcodes a `→` character that the
   Windows `cp1254` console codepage cannot encode, so the script's loop
   over games terminates via an uncaught exception right after the first
   game. Confirmed by Track 3 (2026-08-05): this is why that track's own
   reproduction of the original `ls20`+`vc33` finding only ever completed
   `vc33` — `ls20` never actually ran in that reproduction. Track 3's own
   sweep drives `MyAgent` directly rather than through this script, so it
   was not exposed to the crash; not fixed (out of scope for all
   `baseline-120` tracks).
9. **`build_prompt()` never asks the model for a reasoning/rationale
   field, only the bare action JSON.** The prompt's own final
   instruction is *"Respond with exactly one JSON object: {"action":
   ..., "data": ...}"* — no field for the model to explain *why* it
   chose that action. **This is a deliberate design choice, not a bug:**
   confirmed live 2026-08-06 while debugging a real `cerebras_dev` run
   with `feat/baseline-120-followups`' new visualizer — every response
   the model actually returned was bare JSON and nothing else, exactly
   matching what the prompt asked for; the model is following
   instructions correctly. Keeping responses short avoids wasted
   tokens/latency on every single call, which matters given `decide()`
   allows only one bounded model call per step.

   Worth revisiting as a **switchable** option (a new `Config` flag,
   default off to preserve today's behavior) once prompt work resumes on
   `feat/policy-prompt-legal-budget` or its successor: a real-language
   rationale field would make the visualizer's reasoning panel
   meaningfully more useful for debugging *why* the model picked an
   action, not just *what* it picked — at the cost of extra output
   tokens and latency per call, and a slightly larger JSON schema to
   parse/validate. Not started; no fix candidate written yet beyond this
   note.
10. ~~**`budget_soft_cap`'s default (50) silently turns the back half of a
    100-step game into heuristic-only play, with no model call at all.**~~
    **SOLVED** on `newest-update` (2026-08-06) — the default is no longer
    an unrelated constant: `Config.budget_soft_cap` now defaults to `400`,
    matching the new `Config.max_actions` default, so
    `should_favor_execution` flips at action 320 of 400 (the last 20% of
    the real horizon, which is the semantic the signal was designed for)
    instead of action 40. A deliberately lower soft cap is still a legal
    ablation, but `Config.__post_init__` now emits a `logger.warning`
    naming both values and the number of actions the model will not be
    consulted for, so it can never be silent again. The Colab notebook's
    per-run `ZERX_BUDGET_SOFT_CAP=1000` diagnostic override is kept (it is
    now redundant rather than load-bearing) and its comment corrected.
    Note this changes `Config.config_hash()`, so records from before this
    commit are not hash-comparable with records after it. Tests:
    `tests/test_config.py::test_default_budget_soft_cap_does_not_silence_the_model_early`
    and `::test_low_budget_soft_cap_is_allowed_but_warns`. Original
    description follows.
    Found 2026-08-06 while root-causing this session's real Colab run of
    `integration/baseline-120-colab-ready` @ `4a1fda1` (8 games, 8-bit
    fp8, legal-actions fix included): `aggregate_score: 0.0`, all 8 games
    `NOT_FINISHED`, 0 levels completed, and — the actual tell — every
    single game stopped at exactly 81 actions despite 8 different games.
    `zerx/policy.py`'s `decide()`:
    `budget_favors_execution = budget.should_favor_execution and top.score
    > 0.0`, gated on nothing but `config.heuristic_first` being
    irrelevant to it (`should_favor_execution` alone is enough) — once
    `actions_taken / budget_soft_cap >= 0.8` (`zerx/budget.py`), any turn
    with a scored click candidate skips the model call outright. At the
    default `budget_soft_cap=50` that's action 40 — so roughly the last
    half of every `MAX_STEPS_PER_GAME=100` game plays heuristic-only,
    model untested for that portion. This is a distinct mechanism from
    item 6 (the model was never even asked, not "asked and refused") and
    was not caught by Track 4's original 8-game dev-lane sweep (which
    used the pre-fix prompt, so item 6 dominated the symptom) or by
    `feat/policy-prompt-legal-budget`'s own A/B evidence (which used a
    25-step/game harness, short enough that `should_favor_execution`
    never crossed threshold, so this mechanism never fired in that test).
    Not itself proven to cause the `0.0` for the model-driven front half
    of each game — no per-decision `source` was captured in that run, so
    whether the front ~40 actions were genuinely `"model"` and simply
    unsuccessful, or fell back for an unrelated reason, is undetermined
    from that result alone.

    **Fix applied, `fix/baseline-120-colab-diagnostics`:**
    `scripts/build_colab_notebook.py`'s `smoke_game_cell` now sets
    `ZERX_TRACE_EXPORT_PATH` (per-decision JSONL trace via
    `zerx/trace.py`'s `JsonlTraceWriter`, already built and merged by
    `feat/baseline-120-followups` but never wired into the Colab
    notebook) and raises `ZERX_BUDGET_SOFT_CAP` to `1000` for this
    diagnostic run only (a per-run env override, not a
    `zerx/config.py` default change — the default stays `50`, this is
    not claimed to be the right production value, just the right value
    for isolating model behavior in one measurement) so the next Colab
    run can actually attribute its result to the model instead of a mix
    diluted by an unrelated budget policy. Also fixed the same
    `MAX_ACTIONS` min-only cap bug as item 7 in this same cell (was
    silently capping at 81 instead of the requested 100 — confirmed by
    that same real run). `save_results_cell` now copies the trace
    directory to Drive and records `budget_soft_cap`/`trace_dir` in the
    saved result JSON.

    **Re-run, 2026-08-06 (commit `72d0426`), 8 games, real gemma-4-31b-it
    on Colab A100-80GB, fp8, `MAX_ACTIONS=101` (off-by-one from the
    is_done check, not the min-only bug — actually honored the requested
    100 this time), `ZERX_BUDGET_SOFT_CAP=1000`.** Still `aggregate_score:
    0.0`, 0 levels completed on all 8 games — but this time the trace made
    the actual cause visible instead of leaving it a mystery. `ls20`'s
    trace (101 steps): `source="model"` 25, `source="fallback_deterministic"`
    76 (75%); only `ACTION1` (94x) and `RESET` (7x) were ever executed —
    `ACTION2`/`ACTION3`/`ACTION4` never once, despite being legal.

    **Real root cause (distinct from item 6, which is genuinely fixed):**
    `build_prompt()` rendered its "Ranked click candidates" section
    *unconditionally*, even on turns where `ACTION6` wasn't legal —
    directly contradicting the "Legal actions this turn" line four lines
    below it in the same prompt. The model's own raw output confirms this
    caused real damage, not just risked it: `'call:{"action": "ACTION6",
    "data": {"x": 36, "y": 45}}'` on `ls20` turns where `ACTION6` was
    never legal (`ls20` never has it legal at all, per item 6's original
    finding) — `parse_action()` correctly rejected it every time, falling
    through to `_deterministic_fallback`'s fixed preference order, which
    resolves to a static `ACTION1` for `ls20` specifically. One captured
    reasoning even shows the model noticing the contradiction itself
    mid-response: *"it appears there is a mismatch between the candidate
    list and the legal action list."*

    **Fixed, same commit round, TDD (`superpowers:test-driven-development`
    — failing test written and confirmed RED before the code change, per
    this project's standing practice):** `zerx/policy.py`'s `build_prompt()`
    now gates the entire candidates section on `ActionName.ACTION6 in
    legal_actions`, matching `decide()`'s own existing heuristic-path gate
    (`if candidates and ActionName.ACTION6 in legal_actions`) instead of
    contradicting it. New tests:
    `test_build_prompt_omits_candidates_section_when_action6_not_legal`
    (the bug fix) plus fixture updates to
    `test_build_prompt_lists_ranked_click_candidates`/
    `test_build_prompt_without_candidates_says_so_when_action6_legal` (both
    now explicitly pass `ACTION6` as legal, since that's what they were
    actually testing all along). Full fast suite after the fix: 312
    passed (only the pre-existing `arcengine`-import environment gap
    unrelated to this change, on modules requiring a real `.venv`).

    **Not yet re-verified on a live Colab/Cerebras run** — this fix
    addresses a real, evidenced mechanism (not a guess: confirmed by the
    model's own raw output twice now, once via Cerebras in item 6's live
    reproduction and once via this real Gemma-4-31B-it Colab trace), but
    whether it moves `aggregate_score` off `0.0` is still an open,
    falsifiable question, not a claimed result. The other real
    contributor visible in the same trace — the model never once chose
    `ACTION2`/`ACTION3`/`ACTION4` across 101 steps, model-sourced or not
    — may be genuine capability/exploration limitation rather than a
    prompt defect; not investigated further this round, and not assumed
    to be fixed by this change.

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

**Local `master` is 33 commits ahead of `origin/master`, not yet
pushed** — that push, and any `STRATEGY.md` edit, require explicit
human-owner confirmation before this session proceeds (see top-of-file
"Commit" note). Once confirmed and pushed, this is the priority-ordered
list of what's actually next; none of the items below are started, all
are recommendations for the human owner to schedule:

1. **Both of the human owner's stated prerequisites for the
   authoritative Colab run are now done** (see "Colab run — explicitly
   postponed" below for the original reasoning): the visualizer
   (`scripts/visualize_play.py`) and the `build_prompt()` legal-actions
   fix are both on `master` as of this integration. The next step is
   therefore either or both of:
   (a) re-run the 8-game `cerebras_dev` dev-lane sweep
   (`ls20, vc33, su15, tn36, ka59, lf52, tr87, sc25`) with the fixed
   prompt — it should no longer fall back to invented action names or
   illegal `ACTION6` guesses — and/or
   (b) finally attempt the real, authoritative Colab Gemma-4-31B-it run.
   Only after a real number from (b) exists should the
   `keep`/`revert`/`investigate` verdict be written into `STRATEGY.md`
   §7 — still the one `STRATEGY.md` edit this whole `baseline-120`
   effort authorizes, and still not done by this session.
2. ~~A general `README.md` documenting project usage~~ **Done** —
   shipped by `feat/baseline-120-followups`, now on `master`.
3. ~~A personal `ARC_API_KEY` so local runs attribute to the human
   owner's account~~ **Done (documentation-only)** — `README.md`
   confirms `arc_agi`'s `Arcade` client already resolves `ARC_API_KEY`
   from the environment with zero code changes needed.
4. ~~A JSON-like export of played games for offline inspection~~
   **Done** — `zerx/trace.py`'s `JsonlTraceWriter`/`TraceStep`/`TraceMeta`,
   wired into `agent/my_agent.py` behind `ZERX_TRACE_EXPORT_PATH`
   (off by default), now on `master`.
5. Kaggle Day 1 smoke submission is still open — get explicit approval
   before running it, independent of everything above.
6. `baseline-100`'s results-capture gap is now closed going forward by
   Track 4's Part A notebook rewrite (see "Tests executed and results"
   above) — no further action needed for that specific gap; a fresh Colab
   run would exercise the fix.
7. **Kaggle submission still has no model attached at all** — see
   ARC-HANDOFF-001 below (P0, unresolved by this or any branch). This is
   very likely the single biggest remaining blocker to a real score and
   is independent of the Colab-run item above.
8. The 3 branches merged this round (`feat/baseline-120-followups`,
   `feat/policy-prompt-legal-budget`, `integration/baseline-120-colab-ready`)
   are fully merged into `master` (locally) — safe to delete once the
   human owner confirms the push, not deleted automatically. Same
   standing offer for the earlier `feat/baseline-120-*` and Day-3
   branches, already merged.
9. **Resume/fork from a recorded step — documented, not built.** See
   `docs/superpowers/specs/2026-08-06-baseline-120-followups-design.md`'s
   "Future work: resume/fork from a recorded step" section for the full
   mechanism (deterministically replay a saved trace's recorded actions
   for steps `[0, N)` through the real engine's public step API, verify
   the reached frame matches what was recorded, then hand off live
   control to different code/config/backend/platform from step N
   onward — for fine-tuning/debugging, same environment-split category as
   `scripts/play_local.py`, never used during a scored Kaggle run). The
   trace format (`zerx/trace.py`'s `TraceStep`/`TraceMeta`, now built and
   merged on `feat/baseline-120-followups`) was deliberately designed to
   support this without a breaking change — `TraceStep` already records
   the ordered `action_name`/`action_x`/`action_y` for every step, and
   `TraceMeta` already records `game_id`/`seed`. Recommended as a future
   session's task (e.g. a new `scripts/resume_play.py`, or a
   `--resume-from trace.jsonl:N` flag on `scripts/visualize_play.py`),
   not started.

## baseline-120-reki-core — integration summary (consolidated 2026-08-06)

**Status: infrastructure complete, dev-lane investigated, Colab run
explicitly postponed.** This is not a finished `baseline-120` result —
per `STRATEGY.md` §7.1, only a real Colab Gemma-4-31B-it run can support
a `keep`/`revert`/`investigate` conclusion for the ladder entry itself,
and that run has not happened yet (see "Colab run — explicitly
postponed" below). What *is* true and complete: all 4 tracks per
`docs/superpowers/plans/parallel-baseline-120/README.md` merged into
`master` (locally; not yet pushed), full suite green (308 passed, 0
failed — see "Tests executed and results" above), and a real, working
dev-lane proxy signal exists for the first time.

**What each track delivered, now merged:**

- **Track 1 (backend selection wiring):** `select_backend(config:
  Config) -> ModelBackend` added to `zerx/model_backend.py`, matching the
  frozen interface exactly. `agent/my_agent.py`'s `MyAgent.__init__` now
  calls it instead of hardcoding `GemmaModelBackend` — the exact bug this
  plan's README originally measured (`0.0` aggregate score, 0 levels
  completed, all-`ACTION6`, `ls20`+`vc33`) is fixed. New `Config.gemma_base_url`
  field, defaulting to the original hardcoded URL (no behavior change for
  `gemma_local`). See `docs/superpowers/plans/2026-08-05-baseline-120-backend-wiring.md`.
- **Track 2 (real-game eval harness):** `run_games(config, game_ids,
  max_steps=200) -> List[ExperimentRecord]` added to
  `eval/run_ablation.py`, matching the frozen interface exactly; its
  real-engine integration test now exercises the actual fixed backend
  path (confirmed by explicit re-run after Track 1 merged). See
  `docs/superpowers/plans/2026-08-05-baseline-120-eval-harness.md`.
- **Track 3 (local regression & fallback-loop investigation):** full
  25-game crash-safety sweep (previously only `ls20`+`vc33` had ever been
  checked) — 23 passed, 2 transient skips (both passed on immediate
  retry), 0 failed, no unhandled exception on any public game. Root-caused
  the original stuck-action-loop report to **two distinct, pre-existing
  mechanisms** (missing backend wiring for `ACTION6`-less games; a
  HUD-vs-gameplay-change blind spot in `zerx/transitions.py`'s whole-grid
  diff for `ACTION6`-legal games, already documented as `exp-150-duck-tools`
  Variant A scope) — neither required a fix in this track's owned files.
  Also found (not fixed, out of scope) the `scripts/play_local.py:114`
  Windows Unicode crash — see "Known failures or risks" item 8. See
  `docs/superpowers/plans/2026-08-05-baseline-120-local-regression.md`.
- **Track 4 (Colab validation), Part A — complete:** `scripts/build_colab_notebook.py`'s
  `smoke_game_cell`/`save_results_cell` rewritten for a real in-process
  multi-game loop with real per-game RHAE capture (closing
  `baseline-100`'s results-capture gap for future runs). Found and fixed
  a real Cloudflare-WAF bug blocking every Cerebras call (default
  `urllib` User-Agent triggering HTTP 403/error-1010 — commit `ebfdaf1`,
  outside this track's originally-scoped files, done with the human
  owner's explicit approval mid-session since it blocked this track's own
  deliverable; confirmed no other track touched
  `zerx/backends/cerebras_dev.py` or `tests/test_cerebras_dev.py` this
  round). **Track 4, Part B — real, but deliberately partial (a
  sequencing decision, not an incomplete result):** ran a real
  `cerebras_dev` sweep through the actual harness — 8 games (`ls20, vc33,
  su15, tn36, ka59, lf52, tr87, sc25`), 80 real steps/game (see "Known
  failures or risks" item 7 for why 80 not the requested 100), 640 real
  model-in-loop decisions total, against the actual `agent/my_agent.py`/
  `decide()` loop with a genuinely reachable model. Result: `0.0`
  aggregate score, 0 levels completed everywhere — numerically identical
  to the pre-existing missing-backend "before" reference, but this time
  root-caused to a genuinely different, fixable cause: `build_prompt()`
  never lists the legal action vocabulary, so the model invents invalid
  action names and every decision falls through to fallback (see "Known
  failures or risks" item 6). Conclusion in
  `docs/superpowers/experiments/baseline-120.md`: **`investigate`** — not
  `keep` (behavior is indistinguishable from the fallback-only reference)
  and not `revert` (the loop's real reasoning path was never actually
  exercised, given the prompt gap — nothing substantive to revert). The
  **authoritative Colab Gemma-4-31B-it run was not performed** this
  round — see immediately below.

### Colab run — explicitly postponed (human owner's decision, recorded here per `INTEGRATION.md`'s precondition)

The human owner explicitly decided that the authoritative Colab
Gemma-4-31B-it run for `baseline-120` should happen **after** two things
land on `master`, not now, for a specific reason: right now the only way
to tell whether a Cerebras (or, later, Gemma) call actually reasoned or
silently fell back to the deterministic/heuristic chain is indirect —
inferring it from repetitive action-count patterns in a text log, which
is exactly what Track 4's dev-lane sweep above had to do, since no
per-step `Decision.source` logging or visual trace exists yet. That
inference method is slow and fragile. The two prerequisites, in the human
owner's own stated order:

1. **A visualizer should land first** — reference:
   [github.com/Darkosxl/Agent_Harness_Example](https://github.com/Darkosxl/Agent_Harness_Example),
   a pygame-based live replay viewer: render the grid plus a scrollable
   reasoning-text panel per step, with pause/step-back-forward through a
   capped history buffer, so a human can directly watch or replay each
   decision instead of reconstructing it from aggregate logs after the
   fact.
2. **The `build_prompt()` legal-actions gap should be fixed** (see
   "Known failures or risks" item 6) — this is what's actually needed
   for a non-fallback result in the first place. Re-running the Colab
   sweep before fixing it would very likely just reproduce the same
   fallback-dominated `0.0` result the dev-lane sweep already produced.

Neither of these is implemented as part of this integration — both are
recorded as recommendations under "Exact next action" above, not started.
Per `docs/superpowers/plans/parallel-baseline-120/INTEGRATION.md`'s own
precondition ("if the human owner explicitly decides to land the
infrastructure now and run the experiment in a follow-up session, that
decision must be stated explicitly in `docs/HANDOFF.md`, not implied"):
this section is that explicit statement. `STRATEGY.md` is **not** edited
this round — no `keep`/`revert`/`investigate` verdict is written into its
§7 ladder table until the real Colab number exists; the Cerebras dev-lane
result stays a labeled proxy only, per `AGENTS.md`/`STRATEGY.md`'s hard
backend-mismatch rule.

## Integration — `feat/baseline-120-followups` + `feat/policy-prompt-legal-budget` + `integration/baseline-120-colab-ready` → `master` (2026-08-06)

Executed per
`docs/superpowers/plans/2026-08-06-baseline-120-followups-integration.md`,
in a fresh session with no memory of any of the three branches' build
history — every SHA, merge-base, and branch-state claim in that plan was
re-verified against `origin` before merging. `feat/baseline-120-followups`
had drifted one docs-only commit past the plan's recorded tip (`a4088b4`,
adding this file's own "live reproduction" entry above and the
integration plan doc itself — no code changed); the other two branches
matched the plan's recorded SHAs exactly.

**Merge order and results** (smallest/least-invasive first, per the
plan's "Why this order"), full unfiltered suite after each step:

1. `feat/baseline-120-followups` — clean fast-forward from `master` @
   `b405d3b`, no conflicts. **353 passed, 0 failed.**
2. `feat/policy-prompt-legal-budget` — conflicts in `zerx/policy.py` and
   this file, exactly as the plan (and ARC-HANDOFF-004 below) predicted.
   `zerx/policy.py` resolved by union: the `raw_response`/`model_error`
   capture from `feat/baseline-120-followups` plus the 5-argument
   `build_prompt(perception, new_memory, candidates, legal_actions,
   budget)` call from `feat/policy-prompt-legal-budget`, at both
   `decide()` call sites — not a "pick one side" resolution. This file's
   conflict was two branches independently appending real content to the
   same "known failures item 6" paragraph — kept both, in sequence.
   **372 passed, 0 failed** (+19 over step 1, matching that branch's own
   solo-checkout count exactly).
3. `integration/baseline-120-colab-ready` — auto-merged with **no
   conflict markers at all**, since most of its content was already
   present via step 2. This file's diff from this step was 58 purely
   additive lines (the "Colab/Kaggle quantization decision" section
   above) — no content lost or overwritten, confirmed by reading the
   diff, not just trusting the clean auto-merge. **375 passed, 0
   failed** (+3 over step 2, matching that branch's own unique diff).

**Test count sanity check:** 308 (pre-integration `master`) + 45
(`feat/baseline-120-followups`'s own net-new) + 19
(`feat/policy-prompt-legal-budget`'s own net-new) + 3
(`integration/baseline-120-colab-ready`'s own unique net-new — most of
its tests were already counted via step 2's shared history) = 375.
Matches exactly.

**What this round closes:**

- **Known failures item 6 — the `build_prompt()` legal-actions gap — is
  now fixed on `master`.** `zerx/policy.py`'s `build_prompt()` takes
  `legal_actions`/`budget` and renders them in the prompt text. Verified
  directly post-merge:
  ```python
  prompt = build_prompt(
      PerceptionResult(ascii_grid="0", objects=()), MemoryState(),
      legal_actions=frozenset({ActionName.ACTION1, ActionName.RESET}),
  )
  assert "ACTION1" in prompt and "ACTION2" not in prompt  # passes
  ```
  This is the actual fix for both Track 4's original dev-lane `0.0`
  sweep and this session's own live `ls20` reproduction recorded above.
- **ARC-HANDOFF-004** (the predicted merge hazard, below) **is
  resolved** — the union resolution landed exactly as that entry
  recommended, at merge commit `f0c68f7`.
- **The visualizer + trace tooling `feat/baseline-120-followups` built
  is what let a human directly watch and diagnose the dev-lane Cerebras
  run** that reproduced item 6 live, closing the loop the "Colab run —
  explicitly postponed" note (below) described as the reason the
  visualizer needed to exist before the authoritative Colab run.
- Post-merge verification gates (per the plan): `select_backend()`'s
  lazy `CerebrasDevBackend` import is the only occurrence in
  `zerx/model_backend.py`, still inside the `cerebras_dev` branch, not
  module scope. `STRATEGY.md` is byte-identical to `origin/master`'s
  copy — untouched by this integration, per this project's standing
  rule that only the integration owner edits it, and only after a real
  Colab number exists.

**Noted, not fixed (explicitly out of this integration's scope per the
plan):** `agent/my_agent.py`'s private `_shapes_match` (guarding the
exact-state-memory outcome-recording block) and `zerx/transitions.py`'s
own `_shapes_match` (now added inside `_diff`, via
`feat/policy-prompt-legal-budget`'s ARC-AUDIT-004 fix) are now duplicate
implementations of the same shape check. Not dead code — both call
sites are real and semantically distinct (one guards `_diff`'s shape
assumption inside `TransitionLedger.finalize()`; the other gates
whether `ExactStateMemory.record_outcome()` runs at all) — but worth
deduplicating (import from `zerx.transitions` instead of reimplementing)
in a future small cleanup.

**Not pushed to `origin/master` yet** — pending explicit human-owner
confirmation, per this project's standing rule (every prior integration
here followed the same gate). See "Exact next action" below.

### `feat/baseline-120-followups` — build detail (merged above)

Implemented via a 7-task plan
(`docs/superpowers/plans/2026-08-06-baseline-120-followups.md`), each
task individually code-reviewed.

**What was built** — the 4 items recorded as recommendations in this
file's "Exact next action" section after the `baseline-120` integration:

1. **Trace export / JSONL format.** New `zerx/trace.py` module:
   `TraceMeta`/`TraceStep` frozen dataclasses, the `TraceRecorder`
   protocol, `JsonlTraceWriter` (appends one JSON line per step to a
   `traces/<game_id>-<timestamp>.jsonl` file — `traces/` gitignored, same
   treatment as `notebooks/*.ipynb`), and `CompositeTraceRecorder` (fans
   out to multiple recorders, e.g. live render + file write
   simultaneously). `zerx/policy.py`'s `Decision` gained a new optional
   `raw_response: Optional[str] = None` field, populated by `decide()`
   whenever a model call happens — including on a failed parse, which is
   what makes a captured trace useful for diagnosing the pre-existing
   `build_prompt()` legal-actions gap (see "Known failures or risks" item
   6 above). `zerx/config.py` gained `trace_export_path: Optional[str] =
   None` (env var `ZERX_TRACE_EXPORT_PATH`), off by default.
   `agent/my_agent.py`'s `MyAgent` gained a public `self.trace_recorder`
   attribute (a `JsonlTraceWriter` if `trace_export_path` is set, else
   `None`), reassignable from an external script before `agent.main()`
   runs — the same "reach into agent internals" seam
   `scripts/play_local.py` already uses for `MAX_ACTIONS`, done here
   intentionally. `choose_action()` calls `self.trace_recorder.record(...)`
   once per step only when a recorder is attached — zero overhead and no
   behavior change when it isn't (off by default, this project's usual
   convention for every new flag).
2. **Live + replay pygame visualizer** (`scripts/visualize_play.py`,
   new): `--live --game <id> [--max-steps N] [--save path]
   [--history-cap N]` attaches a `LivePygameRecorder` (optionally
   composed with a `JsonlTraceWriter` via `--save`) to a real `MyAgent`
   run, rendering the grid plus a reasoning-text side panel per step,
   with SPACE pause (blocks inside `record()` on the game loop's own
   thread, so it genuinely halts execution) and ←/→ history navigation
   through a capped in-memory buffer. `--replay <trace.jsonl>` loads a
   saved trace file with no game engine involved, sharing the same
   render/navigate path, always paused. A follow-up fix round (recorded
   in the local, gitignored `.superpowers/sdd/` session ledger for this
   plan — not source of truth, see "Uncommitted or external artifacts"
   below) resolved 2 Important review findings: the
   reasoning panel was ~55% clipped off-screen at the project's real
   64x64 grid (fixed via a wider window plus a new `_wrap_reasoning` pure
   helper that computes wrap width from real font metrics instead of a
   hardcoded guess), and the live window used to vanish the instant a run
   ended (fixed by keeping it open, paused, pumping events, after
   `agent.main()` returns).
3. **`README.md`** (new, repo root): project overview, setup, running
   locally, `ZERX_*` env var overview, running tests (including the
   `-m "not slow_local_engine"` fast-iteration filter), and the new
   visualizer's `--live`/`--replay` usage.
4. **`ARC_API_KEY` documentation.** The design investigation's
   no-code-change finding — that `arc_agi`'s `Arcade` client already
   resolves `ARC_API_KEY` via `constructor arg > ARC_API_KEY env var >
   anonymous-key fallback` internally, and every call site in this repo
   constructs `Arcade(...)` with no override, so setting the env var in
   your own shell already works with zero code changes — was confirmed
   accurate; this item is documentation-only, folded into `README.md`.

**Manual-verification result (Task 5's own smoke test, both before and
after the fix round):** a live `--game ls20 --max-steps 5 --save
<path>.jsonl` run exited cleanly (exit code 0, no traceback) and produced
a well-formed trace file (valid meta line + step lines, correct
`game_id`, parses back via `_load_trace` into the exact dataclasses) —
confirmed both pre-fix and post-fix. **Visual/UI correctness was
explicitly not verified by any implementer:** grid cell colors matching
the palette, reasoning-panel text legibility/wrapping, SPACE actually
pausing/resuming a live run, and ←/→ arrow-key history navigation
changing what's rendered on screen were all out of reach — no screenshot
or display-inspection tool was available in the implementing sessions.
This remains an open item: a human needs eyes on the actual running
window before the visualizer is treated as fully proven, not just "code
complete."

**Final test count (this task, full unfiltered suite, no `-m` filter):**
`.venv/Scripts/pytest.exe tests/ -q` → **332 passed, 0 failed** (1
pre-existing, unrelated `PytestUnknownMarkWarning` for
`pytest.mark.slow_local_engine`, same warning noted in the
`baseline-120-reki-core` integration above).

**Deferred:** 14 Minor findings surfaced across this branch's per-task
code reviews, intentionally left unfixed, per each task's explicit
instruction to defer Minor findings to a final whole-branch review
rather than fix them piecemeal. Breakdown by task (full per-finding
descriptions live in this branch's SDD ledger,
`.superpowers/sdd/2026-08-06-baseline-120-followups/progress.md` —
gitignored, not source of truth, see "Uncommitted or external artifacts"
below — and in the individual task review reports):
- Task 2 (`zerx/trace.py`): 2 — `JsonlTraceWriter` untested without a
  `write_meta` call; nested-directory `mkdir` untested.
- Task 4 (config/agent wiring): 2 — no test for trace fidelity under the
  exact-state-suppression override; the new trace-recording block lacks
  a banner-comment pair unlike its neighboring config-gated blocks.
- Task 5 (`scripts/visualize_play.py`): 10 — 8 from the original review
  round (composite-recorder ordering on quit, `--history-cap 0`
  crashing, argparse gaps, `_run_replay` touching recorder privates, the
  replay window mis-captioned "live", meta discarded in replay, a
  zero-step trace opening a blank window, a missing `VENDOR.exists()`
  guard) plus 2 more surfaced by the fix-round re-review (SPACE-after-
  run-end can re-close the live window as an edge case;
  `_wrap_reasoning`'s "... (N more lines)" indicator isn't
  width-bounded).

They are not itemized beyond this per-task breakdown — triaging and
deciding which to fix is the final reviewer's job, not a checklist to be
resolved in this status entry.

**Final whole-branch review — done.** Triaged all 14 deferred Minor
findings above as fine-to-defer (none were must-fix-before-merge), and
found 3 new Important findings no single task-scoped review could see —
all 3 traced to code or a scope decision this session's own plan
authored, not implementer mistakes, so each was confirmed with the human
owner before fixing:

1. `JsonlTraceWriter` opened in append mode with no auto-naming, so
   re-running the README's own documented example silently merged two
   runs into one corrupt trace file. Fixed: it now refuses to construct
   against an already-existing exact file, and supports a directory-mode
   that auto-names `<dir>/<game_id>-<timestamp>.jsonl`. Config-driven
   traces (`ZERX_TRACE_EXPORT_PATH`) previously had no meta line and
   could never be replayed — `MyAgent.__init__` now always calls
   `write_meta(...)` on its writer.
2. A `trace_recorder.record()` failure (disk full, permissions) could
   propagate into `_choose_action_inner` and skip the transition-ledger
   update for that step, desyncing agent state for the rest of the run —
   on exactly the Colab runs this project's promotion gate depends on.
   Fixed: wrapped in `try/except Exception` with a log warning; a
   dev-only observability sink can no longer alter real agent behavior.
3. The approved design spec's ↑/↓ reasoning-panel scroll was silently
   dropped during planning in favor of a truncation indicator. Fixed:
   real scrolling implemented (`_clamp_scroll`, `LivePygameRecorder`'s
   `_reasoning_scroll`/`_current_step` state, UP/DOWN handlers).

Fix wave commits `1305572`, `644f83d`; scoped re-review confirmed all 3
addressed with no new breakage. **Final test count (full unfiltered
suite, superseding the `332 passed` figure above):**
`.venv/Scripts/pytest.exe tests/ -q` → **341 passed, 0 failed** (332 + 9
new tests covering the 3 fixes). The visual/UI-correctness gap noted
above is unchanged by this fix wave — still not verified by any
implementer, still needs a human with eyes on the running window.

## Uncommitted or external artifacts

None tracked or required. `.venv/`, `vendor/ARC-AGI-3-Agents/`,
`environment_files/`, `notebooks/*.ipynb`, and `.superpowers/` (SDD
session ledger/briefs/reports) all exist locally and are gitignored — not
source of truth. No credentials of any kind exist in **this Claude Code
session's own tool environment** (no `CEREBRAS_API_KEY`, no Kaggle token)
— every test that would need one injects a fake instead; see "Cerebras
development state" above for the one nuance (a live key now exists in the
human owner's own terminal, never shared with this session).

## Unresolved work from the 2026-08-06 full-repository audit

Source: `docs/audits/2026-08-06-full-repository-audit.md` (20 findings).
Each finding was then checked against **every** branch in the repo
(branch-wide resolution check, 2026-08-06). Only findings that **no
branch solves** appear below.

**Already solved, and now merged onto `master`** via the "Integration"
section above (2026-08-06, local `master` only — not yet pushed, see
"Exact next action") — do NOT re-implement:

| Audit ID | Problem | Solved on |
|---|---|---|
| ARC-AUDIT-001 | Kaggle bundle unimportable (`ModuleNotFoundError: zerx.backends`) | `feat/policy-prompt-legal-budget` @ `c964ea0`, now on `master` |
| ARC-AUDIT-002 | Notebook writes `/tmp/zerx/*.py` without creating `/tmp/zerx` | same |
| ARC-AUDIT-004 | `transitions._diff` IndexError / false "no change" on grid-shape change | same |
| ARC-AUDIT-005 | `levels_completed` discarded (`score=0` hardcoded) | same — note `feat/baseline-120-followups` still has `score=0`; its `levels_completed` reference is trace metadata only, not this fix |
| ARC-AUDIT-006 | Unbounded prompt object table (~49k tokens on a legal frame) | same |
| ARC-AUDIT-012 | Suspected cross-game state leak | **Not a defect** — `Swarm.main()` builds a fresh agent per game. Verified, closed, no work needed. |

`feat/baseline-120-followups` independently solved the **observability**
half of ARC-AUDIT-003 (`Decision.model_error` / `raw_response` +
per-step `logger.warning`, commits `35c0577` / `75eba2f`) and improved
Cerebras HTTP error surfacing (`35a1f2d`) — also now on `master`. Do not
re-implement those either. ARC-AUDIT-003's **other** half (no model
actually attached to the Kaggle notebook) remains unresolved — see
ARC-HANDOFF-001 below, still open.

---

### [P0] ARC-HANDOFF-001 — The Kaggle submission notebook contains no model

**Status:** **PARTIALLY RESOLVED, 2026-08-06** on
`feat/kaggle-p0-model-attach` (not on `master`) · **Source:**
ARC-AUDIT-003, ARC-AUDIT-015 · **Category:** Kaggle / strategy

#### Why Phase B is deferred (read before "fixing" this)

This entry's own "Recommended Fix" below says to "add a cell that installs
vLLM offline and launches the server". **That step rests on an assumption
nobody has verified**, and acting on it directly would be building on a
planned-but-unconfirmed path, which `AGENTS.md` explicitly forbids ("Do
not assume that a design-document path or command exists merely because it
was planned").

Evidence found 2026-08-06 while scoping the fix: vLLM is not part of the
Kaggle Python image; vLLM's own tracker carries an open installation issue
titled "vLLM will NOT run in a Kaggle Notebook" for versions above 0.10;
and the community workaround is a multi-gigabyte prebuilt-wheels dataset.
Independently, the Colab bring-up of this same model already cost four
separately-diagnosed install failures (see "Completed changes (Day 1 +
Day 2)" items 1–4 above). The competition's own offline wheels directory
ships `arc-agi`, not vLLM.

So Phase A instead built `scripts/build_probe_notebook.py` →
`notebooks/probe/probe.ipynb`: a non-submission kernel that runs in an
environment configured identically to the real submission (same
`rtx6000` accelerator, internet disabled, same competition and same Gemma
model attached) and reports GPU/VRAM/compute-capability, `torch` and
`transformers` versions, whether `vllm`/`bitsandbytes`/`accelerate` are
importable, which quantization configs `transformers` exposes, the real
`/kaggle/input` mount path for the weights, and the competition wheels
listing — every section independently error-trapped so one failure does
not cost the other answers, with results written to
`/kaggle/working/probe.json`. It consumes no submission slot.

Phase B (serve the model) is written from that result: vLLM server, an
in-process `transformers` backend, or a prequantized checkpoint — chosen
from measurements rather than from this entry's original guess.

**Until Phase B lands, the built notebook deliberately refuses to run on
Kaggle**: `KAGGLE_MODEL_DIR` is `None`, and the run cell raises
`SystemExit` rather than playing heuristics-only and reporting it as a
scored run.

Two acceptance criteria below are now met on that branch
(`model_sources` non-empty; server-unreachable/model-missing causes a
loud, early failure rather than silent fallback), plus the existing tests
still pass — full unfiltered suite **395 passed, 0 failed**, up from 378,
with 17 new tests in `tests/test_build_probe_notebook.py` and
`tests/test_kaggle_bundle_importable.py`. The remaining criteria need
Phase B.

#### Problem
A submission built from any current branch runs the agent with **no
language model at all**. It plays heuristics-only and fails silently —
no exception, no crash, just a low score.

#### Evidence
- `notebooks/kernel-metadata.json`: `model_sources: []`,
  `dataset_sources: []` — identical on `master`,
  `feat/baseline-120-followups`, and
  `integration/baseline-120-colab-ready`.
- `scripts/build_notebook.py`: zero occurrences of `vllm`, zero
  occurrences of `ZERX_BACKEND` on all three branches.
- The `.env` the run cell writes contains only ARC gateway variables.
- Therefore `Config.backend` falls to its default `"fake"`, and
  `select_backend` returns `FakeModelBackend()` with an empty response
  list, whose every `generate()` raises `RuntimeError`.
- Confirmed by importing the real built bundle:
  `select_backend(Config())` → `FakeModelBackend`.

#### Root Cause
The Gemma serving path was only ever built for Colab
(`scripts/build_colab_notebook.py`). It was never extended to the Kaggle
submission notebook. `ACCELERATOR = "t4"` (2×16 GB) also contradicts
AGENTS.md's RTX Pro 6000 (48 GB) target.

#### ARC-AGI-3 Impact
Total. Exploration, modeling, goal inference and planning are all
absent — the entire Gemma thesis is missing from the scored artifact.
Every decision falls through the fallback chain.

#### Reproduction
```bash
.venv/bin/python scripts/build_notebook.py
# extract the %%writefile /tmp/zerx/* cells into a dir, then:
PYTHONPATH=<dir> python -c "from zerx.config import Config; \
from zerx.model_backend import select_backend; print(type(select_backend(Config())))"
# -> <class 'zerx.model_backend.FakeModelBackend'>
```

#### Expected Behavior
The notebook attaches Gemma-4-31B as a Kaggle model source, serves it
(vLLM, offline, from `/kaggle/input`), and sets `ZERX_BACKEND=gemma_kaggle`
+ `ZERX_PLATFORM=kaggle` so `select_backend` returns `GemmaModelBackend`.

#### Current Behavior
No weights attached, no server started, backend silently `fake`.

#### Existing Branch Search
Branches checked: all 15 (`master`, `day1-local-skeleton`,
`day2-colab-gemma-baseline-100`, `feat/baseline-115-exact-state-memory`,
`feat/baseline-120-backend-wiring`, `feat/baseline-120-colab-validation`,
`feat/baseline-120-eval-harness`, `feat/baseline-120-followups`,
`feat/baseline-120-local-regression`, `feat/baseline-130-hypothesis-memory`,
`feat/exp-140-vlm-refinement`, `feat/exp-150-duck-tools-ab`,
`feat/policy-prompt-legal-budget`, `integration/baseline-120-colab-ready`).

Closest existing implementation:
- `integration/baseline-120-colab-ready` @ `96063b9` settled the
  **quantization decision** that this task depends on: Kaggle's 48 GB card
  cannot hold bf16 (~61.4 GB), so Colab was switched to 8-bit fp8 to
  mirror what Kaggle must run. `4a1fda1` also added framework
  clone+slim to the Colab notebook.
- `feat/baseline-120-followups` @ `75eba2f` makes the *failure* visible
  (`model_error` + `logger.warning`) but attaches no model.

Why neither fully solves this: both change Colab or observability. **No
branch touches `notebooks/kernel-metadata.json`'s `model_sources`, adds a
vLLM cell to `scripts/build_notebook.py`, or sets `ZERX_BACKEND` for the
Kaggle run.** Verified by direct `git show` on every branch.

#### Recommended Fix
1. `notebooks/kernel-metadata.json`: add the Gemma model handle to
   `model_sources` (and any wheel dataset to `dataset_sources`).
2. `scripts/build_notebook.py`: set `ACCELERATOR = "rtx6000"`; add a cell
   that installs vLLM offline and launches the server against the
   `/kaggle/input` model path, using the **8-bit fp8** configuration
   already validated on `integration/baseline-120-colab-ready` (reuse
   that branch's flags — do not re-derive them).
3. Export `ZERX_BACKEND=gemma_kaggle`, `ZERX_PLATFORM=kaggle`, and
   `ZERX_GEMMA_BASE_URL` before `main.py --agent myagent` runs.
4. Add a readiness gate: the run cell should fail fast if the server is
   not answering, rather than proceeding into a heuristics-only run.

Invariants to preserve: internet stays disabled; nothing is downloaded at
eval time; `zerx/backends/` is still never bundled; the existing
build-time secret scan still gates the build.

#### Acceptance Criteria
- [ ] Built bundle's `select_backend(Config.from_env())` returns
      `GemmaModelBackend`, not `FakeModelBackend`
- [ ] `model_sources` is non-empty and resolves under `/kaggle/input`
- [ ] No network access is attempted at evaluation time
- [ ] Server-unreachable causes a loud, early failure, not silent fallback
- [ ] `zerx/backends/` still absent from the bundle; secret scan still passes
- [ ] Existing tests continue passing

#### Tests Required
Extend `tests/test_kaggle_bundle_importable.py`: assert the built
notebook sets `ZERX_BACKEND` to a real backend and that
`kernel-metadata.json` has a non-empty `model_sources`; a unit test that
`select_backend` returns `GemmaModelBackend` for the Kaggle env vars; an
offline test asserting no `pip install` without `--no-index`.

#### Risks / Side Effects
Model load can dominate the ~9 h budget; `rtx6000` burns GPU quota
faster. A wrong `ZERX_PLATFORM` re-enables the `cerebras_dev` guard path.

#### Dependencies
Depends on: `integration/baseline-120-colab-ready` @ `96063b9` (the 8-bit
fp8 parity decision and validated vLLM flags).

#### Definition of Done
Kaggle notebook builds, imports, starts the model, plays with
`source="model"` decisions in the trace, and the run is recorded in an
experiment file with GPU/precision/model revision.

---

### [P1] ARC-HANDOFF-002 — Concurrent game threads share mutable `GameAction` singletons

**Status:** UNRESOLVED · **Source:** ARC-AUDIT-007 · **Category:** Correctness / concurrency

#### Problem
On Kaggle all games run **concurrently in threads**, and every thread
mutates the same process-wide `GameAction` enum members. One game can
submit another game's click coordinates.

#### Evidence
- `main.py` (vendored): with no `--game`, "an agent swarm will play all
  available games". The Kaggle run cell calls
  `python main.py --agent myagent` — no `--game`.
- `agents/swarm.py:76-95`: builds one agent per game, then
  `Thread(target=a.main, daemon=True)` for each, all started together.
- `arcengine.enums.GameAction.set_data` is
  `self.action_data = self.action_type(**data)` — mutating a shared
  singleton.
- `agent/my_agent.py:_to_game_action` mutates
  `GameAction.ACTION6.action_data` and `.reasoning`, returns the shared
  member; the framework reads `action.action_data` **later**, in
  `do_action_request`.

#### Root Cause
Upstream architecture: actions are singletons carrying mutable payloads,
and there is a read-after-write window we do not control. Our agent
inherits it and additionally writes `.reasoning`.

#### ARC-AGI-3 Impact
Corrupts ACTION6 — the most information-rich action — across all games at
once. Also poisons `TransitionLedger` and `ExactStateMemory`, which record
the action we *intended*, not the one actually sent, so the evidence loop
learns from fiction. Invisible locally: `make verify-local` passes
`--game`, so single-game runs never reproduce it.

#### Reproduction
Run two games concurrently through `Swarm` with an agent that logs both
the coordinates it set and the coordinates present at
`do_action_request` time; they diverge under load.

#### Expected Behavior
The coordinates a game submits are always the coordinates that game's
policy chose.

#### Current Behavior
Another thread can overwrite them in between.

#### Existing Branch Search
Branches checked: all 15. Grepped every branch's `agent/my_agent.py` for
`Lock`, `threading`, `deepcopy`, `copy(` — **zero matches on every
branch**. No branch modifies `_to_game_action` at all.

Why nothing solves it: `feat/baseline-120-followups` touches
`my_agent.py` (+52 lines) but only for trace recording; it does not
address action identity or thread safety.

#### Recommended Fix
This is **not** fixable from inside `my_agent.py` alone — we do not
control the window between `choose_action` returning and `take_action`
reading. Two credible options, both owner decisions:
- **(a) One game per process** — change the Kaggle run cell to iterate
  games sequentially or in separate processes. Simplest, no framework
  patch; costs wall-clock, which matters against the ~9 h budget.
- **(b) Patch the bundled framework copy** so `do_action_request`
  receives an immutable per-call action payload rather than reading the
  shared enum.
Record whichever is chosen and why; do not do both.

#### Acceptance Criteria
- [ ] A multi-game concurrent run shows zero divergence between intended
      and submitted action data
- [ ] The chosen approach is documented with its runtime cost
- [ ] `TransitionLedger` records match actually-submitted actions

#### Tests Required
A threading regression test driving two agents concurrently and asserting
each submitted action matches that agent's own decision.

#### Risks / Side Effects
(a) increases wall-clock and may not finish all games in budget.
(b) means shipping a modified framework — re-check competition rules.

#### Definition of Done
Concurrency hazard eliminated or explicitly accepted in writing with
measured impact.

---

### [P1] ARC-HANDOFF-003 — Four ablation flags cannot change behaviour

**Status:** UNRESOLVED · **Source:** ARC-AUDIT-008/009/010/011 ·
**Category:** Experiment integrity

#### Problem
Four configuration flags reach `eval/run_ablation.py`'s matrix while
controlling nothing. Any A/B using them is guaranteed to report "no
effect", which risks discarding good ideas for the wrong reason —
directly undermining AGENTS.md's promotion-gate methodology.

#### Evidence
- **`memory_on` (default `True`)** — `zerx/policy.py` calls
  `maybe_refresh(..., summarizer=lambda prev, ctx: prev, ...)`. The
  summary can never become non-empty; the prompt permanently reads
  `What you've learned so far: (nothing yet)`.
- **`structured_memory_on`** — `memory.render_for_prompt()` has **zero**
  production callers and `build_prompt` has no structured-memory
  parameter. When enabled it runs a full `perceive()` flood-fill **every
  action** to feed a no-op whose output is never read.
- **`arbiter_on`** — `candidates.select_candidate` gates on
  `config.arbiter_on and arbiter is not None`, but `decide()` calls
  `select_candidate(model_candidates, config)` with **no arbiter
  argument**. Unsatisfiable.
- **`duck_objects_on`** — appears only in `zerx/config.py` and
  `eval/run_ablation.py:39`. Nothing in `zerx/` or `agent/` reads it.
  (`zerx/scene.py` being unwired is intentional per its docstring; the
  **ablation entry** is the defect.)

#### Root Cause
Each Day-3 track landed its data structures and its config flag, but the
final wiring into the live `decide()`/prompt path was left for later.
The flags shipped anyway.

#### ARC-AGI-3 Impact
No direct gameplay harm — but it makes the experiment ladder untrustworthy,
and `structured_memory_on` wastes CPU per action for nothing.

#### Existing Branch Search
Branches checked: all 15. On `feat/baseline-120-followups` and
`integration/baseline-120-colab-ready`, verified directly:
`summarizer=lambda prev, ctx: prev` still present;
`select_candidate(model_candidates, config)` still arbiter-less;
`duck_objects_on` still only in `config.py`; `render_for_prompt` still
uncalled. **No branch wires any of the four.**

#### Recommended Fix
Per flag, choose *wire* or *remove* — do not leave a third state:
- `memory_on`: supply a real summarizer (a bounded model call at
  `memory_refresh_interval`), measuring its latency separately per
  AGENTS.md; or default it to `False` until then.
- `structured_memory_on`: add a parameter to `build_prompt` and render
  `render_for_prompt(state)`; until then, skip the `perceive()` call so
  the flag is free rather than merely useless.
- `arbiter_on`: pass an arbiter backend from `decide()`, or delete the
  flag and the `_select_with_arbiter` path.
- `duck_objects_on`: remove from `eval/run_ablation.py`'s
  `_CONFIG_ENV_MAP` until `zerx/scene.py` is actually wired.

#### Acceptance Criteria
- [ ] Every flag in `_CONFIG_ENV_MAP` demonstrably changes at least one
      observable behaviour, proven by a test
- [ ] No flag defaults to `True` while being a no-op
- [ ] `structured_memory_on=True` no longer costs a `perceive()` per
      action unless its output is used

#### Tests Required
For each retained flag, a test asserting on/off produce different
`Decision` output or different prompt text. A guard test asserting every
key in `_CONFIG_ENV_MAP` corresponds to a flag read somewhere in
`zerx/` or `agent/`.

#### Risks / Side Effects
Wiring `memory_on` adds model calls (latency, not action budget — measure
separately). Removing flags changes `Config.config_hash()`, so prior
experiment records become non-comparable — note it in the experiment log.

#### Definition of Done
`_CONFIG_ENV_MAP` contains only flags with proven behavioural effect.

---

### [P0-integration] ARC-HANDOFF-004 — Merge hazard: two branches edit the same `decide()` block, one would revert the legal-actions fix

**Status:** **RESOLVED, 2026-08-06** (merge commit `f0c68f7`, part of the
"Integration" section above) · **Source:** branch-wide resolution check ·
**Category:** Integration

**Resolution:** the union resolution below landed exactly as recommended
— confirmed post-merge: `grep -n "model_error" zerx/policy.py` and
`grep -n "legal_actions, budget" zerx/policy.py` (twice) both match on
`master`. One deviation from this entry's own "Dependencies" note below:
the actual integration plan
(`docs/superpowers/plans/2026-08-06-baseline-120-followups-integration.md`)
merged in a different order —
`feat/baseline-120-followups` → `feat/policy-prompt-legal-budget` →
`integration/baseline-120-colab-ready` — with its own documented
rationale (`colab-ready` already contains most of
`policy-prompt-legal-budget`'s content, so merging the latter first lets
git apply only `colab-ready`'s unique diff). Worked cleanly: step 2 hit
exactly the predicted conflict and resolved it via this entry's exact
union; step 3 auto-merged with no conflicts at all. Final suite: 375
passed, 0 failed — exceeds this entry's acceptance bar.

#### Problem
`feat/policy-prompt-legal-budget` and `feat/baseline-120-followups` both
modify the **same lines** of `decide()`'s model-call block in
`zerx/policy.py`. A careless conflict resolution that takes "theirs"
would **silently revert the legal-actions fix** — the confirmed root cause
of `baseline-120`'s flat `0.0` sweep (known issue 6).

#### Evidence
- Mine (`8a6bc05`):
  `build_prompt(perception, new_memory, candidates, legal_actions, budget)`
  at both call sites.
- Theirs (`35a1f2d`): `raw_response = backend.generate(build_prompt(perception, new_memory, candidates))`
  and `except Exception as exc: model_error = ...` — keeps the **old
  3-argument** `build_prompt` call.
- Verified: `feat/baseline-120-followups` has **0** occurrences of
  `Legal actions this turn`.

#### Root Cause
Parallel branches off the same `master` touching one hot function, with
neither aware of the other. Not a bug in either branch.

#### Why this is not "duplicate work"
The two changes are **complementary, not competing**:
- mine adds *inputs* to the prompt (legal actions, budget)
- theirs adds *outputs* from the call (`raw_response`, `model_error`)
The correct resolution is a **union**, not a choice.

#### Recommended Fix (at merge time)
Resolve to a block that keeps both:
```python
raw_response = backend.generate(
    build_prompt(perception, new_memory, candidates, legal_actions, budget)
)
parsed = parse_action(raw_response, legal_actions)
except Exception as exc:
    parsed = None
    model_error = f"{type(exc).__name__}: {exc}"
```
and the same 5-argument `build_prompt(...)` inside the
`candidate_count > 1` branch.

#### Acceptance Criteria
- [x] Post-merge `zerx/policy.py` contains **both** `Legal actions this
      turn` and `model_error` — confirmed by grep, 2026-08-06
- [x] `tests/test_policy_decide.py`'s legal-action tests pass — 46/46
      passed (`test_policy_decide.py` + `test_trace.py`), 2026-08-06
- [x] `feat/baseline-120-followups`' trace tests pass — same run above
- [x] Combined suite ≥ 332 + this branch's new tests, 0 failures — 375
      passed, 0 failed, 2026-08-06

#### Dependencies
Also note `integration/baseline-120-colab-ready` has **already merged**
all three `feat/policy-prompt-legal-budget` commits
(`e402a0d`, `c964ea0`, `8a6bc05`) — merging that branch and
`feat/policy-prompt-legal-budget` separately will produce redundant (but
harmless, already-identical) history. Merge order matters:
`integration/baseline-120-colab-ready` first, then
`feat/baseline-120-followups`, resolving the union above.

---

### [P3] ARC-HANDOFF-005 — Test tooling: root `pytest` is broken and marks are unregistered

**Status:** UNRESOLVED · **Source:** ARC-AUDIT-019/020 · **Category:** Testing

#### Problem
A bare `pytest` from the repo root **fails collection** — it tries to
collect the *vendored* framework's suite:
`ERROR vendor/ARC-AGI-3-Agents/tests - ModuleNotFoundError: No module named 'tests.conftest'`.
AGENTS.md's documented command (`uv run pytest -q`) therefore does not
work as written; everyone must know to type `pytest tests`.

Separately, `cerebras_live` and `slow_local_engine` are **unregistered**
marks (`PytestUnknownMarkWarning`), so `-m` filtering silently depends on
convention — a typo'd mark name selects nothing and reports success.

#### Evidence
There is no `pytest.ini`, `pyproject.toml`, `setup.cfg`, or `tox.ini`
anywhere in `git ls-files` — confirmed absent on **all** branches.
`feat/baseline-120-followups`' own HANDOFF notes the same warning as
"pre-existing, unrelated" — acknowledged there, not fixed.

#### Recommended Fix
Add a root `pytest.ini` (or `[tool.pytest.ini_options]`) with:
```ini
[pytest]
testpaths = tests
markers =
    cerebras_live: hits the real Cerebras API; requires CEREBRAS_API_KEY
    slow_local_engine: drives the real local game engine; slow
```

#### Acceptance Criteria
- [ ] bare `pytest` from the repo root collects only `tests/` and passes
- [ ] no `PytestUnknownMarkWarning`
- [ ] `-m "not slow_local_engine"` still filters correctly
- [ ] AGENTS.md's documented command works as written

#### Risks / Side Effects
`testpaths` changes what CI collects by default — confirm no intended
test lives outside `tests/`.

---

### [P3] ARC-HANDOFF-006 — Config/scanner hardening (three small, independent items)

**Status:** UNRESOLVED · **Source:** ARC-AUDIT-016/017/018 ·
**Category:** Configuration / security

Verified unresolved on **all** branches (`_env_int` has no `try/except`;
`Config.__post_init__` still has exactly one `platform == "kaggle"`
check; `zerx/secret_scan.py` still has exactly 2 `re.compile` patterns).

1. **`Config.from_env` crashes on a malformed value.** `_env_int` /
   `_env_float` call bare `int()`/`float()`, so `ZERX_BUDGET_SOFT_CAP=abc`
   raises `ValueError` inside `MyAgent.__init__` — **outside**
   `choose_action`'s catch-all, so the whole game aborts on a typo.
   *Fix:* catch and re-raise with the offending variable name, or fall
   back to the default with a loud warning. Decide which — a silent
   fallback can hide a misconfigured experiment.
2. **Cerebras lockout implements 1 of 3 required conditions.** AGENTS.md
   requires rejection "whenever `platform=kaggle`, competition mode is
   active, or internet is disabled"; only the first exists. Residual risk
   is low (defence in depth: `CerebrasDevBackend`'s own guard, plus the
   bundle never ships `zerx/backends/`), but the stated contract is
   unmet. *Fix:* add the two missing conditions to
   `Config.__post_init__`.
3. **Secret scanner is literal-pattern-only.** `zerx/secret_scan.py`
   matches only `api.cerebras.ai` and `CEREBRAS_API_KEY`; a leaked key
   *value* without its variable name passes. *Fix:* add generic
   high-entropy / `sk-`-prefixed / bearer-token patterns. Keep the
   existing `secret_scan.py` self-exemption narrow.

#### Acceptance Criteria
- [ ] Malformed `ZERX_*` value produces a clear, actionable error naming
      the variable (or a warned default) — never a bare `ValueError`
- [ ] `Config` rejects `cerebras_dev` under all three documented
      conditions, with a test per condition
- [ ] Secret scan catches a planted bare key value, not just the name
- [ ] Existing scan tests still pass

---

## Branch `newest-update` — 2026-08-06

Scope was explicitly **everything except** the already-triaged Kaggle
blockers: ARC-HANDOFF-001 (no model attached to the submission notebook),
`ACCELERATOR = "t4"`, the `kernel-metadata.json` username placeholder, the
missing local `.venv`/`.kaggle` credentials, the not-yet-attempted Day-1
smoke submission, ARC-HANDOFF-002 (concurrent `GameAction` singletons) and
ARC-HANDOFF-003 (four no-op ablation flags) are all owned elsewhere and were
deliberately **not** touched on this branch. ARC-HANDOFF-005 and -006 (the
P3 tooling/config-hardening items) were likewise left alone for the same
reason.

Environment note: this checkout had no `.venv` and no `vendor/`, so both
were created first (`python3.12 -m venv`, `pip install arc-agi pytest numpy
pygame python-dotenv`, `git clone` + `scripts/slim_framework.py`). Test
counts below are from that environment.

- Suite before any change: **353 passed, 25 deselected**
  (`-m "not slow_local_engine"`); **378** unfiltered.
- Suite after: **367 passed, 25 deselected**; **392 passed, 0 failed**
  unfiltered (+14 net-new tests).

Three previously-unrecorded defects were found, plus four already-recorded
ones closed out (items 3, 7, 8 and 10 in "Known failures or risks" above,
each marked SOLVED in place).

---

### [P0] ARC-HANDOFF-007 — Every Kaggle game was hard-capped at 81 actions

**Status:** **SOLVED** on `newest-update` · **Category:** Scoring

#### Problem
`MyAgent` never overrode `MAX_ACTIONS`, so it inherited
`agents.agent.Agent.MAX_ACTIONS = 80` — a generic "don't loop forever"
guard the upstream framework sets for its own examples. `Agent.main()`'s
loop condition is `self.action_counter <= self.MAX_ACTIONS`, so every game
stopped after 81 actions. Several upstream templates override it
(`reasoning_agent.py` uses 400, `multimodal.py` 40); ours did not.

#### Why it mattered
On Kaggle this is a hard ceiling on the score, and an invisible one: no
exception, no warning, just `Exiting: agent reached MAX_ACTIONS of 80` in a
log nobody reads, and a low number on the leaderboard. Reaching a level
completion in ARC-AGI-3 generally takes more than 80 actions, so the agent
was being stopped before its own thesis could be tested. This is
independent of, and additive to, ARC-HANDOFF-001 — fixing the missing model
alone would not have lifted this ceiling.

#### Fix
New `Config.max_actions` field (default `400`, env `ZERX_MAX_ACTIONS`).
`MyAgent.__init__` sets `self.MAX_ACTIONS = self._config.max_actions`, so
the cap is a recorded, ablatable, serialized decision rather than an
upstream default. 400 is not a tuned value — it matches the upstream
framework's own `reasoning_agent.py` choice and is deliberately
configurable.

Because the cap is now applied to the *instance* at construction, anything
that sets the *class* attribute beforehand is silently overwritten. Every
such call site was migrated to `ZERX_MAX_ACTIONS`:
`scripts/play_local.py`, `eval/run_ablation.py`'s `run_games`, and
`scripts/build_colab_notebook.py`'s `smoke_game_cell`.
`scripts/visualize_play.py` already assigned to the instance after
construction and still works; its `min()` was reduced to a plain assignment
for the same reason as item 7 above.

#### Verification
`scripts/play_local.py --game ls20 --max-steps 120` → `actions=121`
(previously 81 for any requested value above 80).
`tests/test_my_agent.py::test_agent_action_cap_comes_from_config_not_the_upstream_default`
asserts both the inherited value we must not use (80) and the configured
one.

#### Risk / follow-up owned by whoever runs the submission
Raising the cap raises wall-clock exposure against Kaggle's ~9 h notebook
limit — roughly 5× the model calls per game versus the old 81. That risk is
what ARC-HANDOFF-008 below addresses, but the arithmetic still needs a real
measurement: **nobody has yet timed one Gemma-backed action on the target
card.** Do that before the final submission and set `ZERX_MAX_ACTIONS`
accordingly rather than trusting the 400 default blind.

---

### [P1] ARC-HANDOFF-008 — No wall-clock guard on a game

**Status:** **SOLVED** on `newest-update` · **Category:** Run safety

#### Problem
Nothing bounded how long a single game could run. With the 81-action
ceiling this was academic; with ARC-HANDOFF-007's fix it is not. A slow or
hung model turns the whole notebook into a 9-hour timeout kill with no
scorecard at all, rather than costing one game its tail.

#### Fix
New `Config.max_wall_seconds` (default `7200`, `0` disables, env
`ZERX_MAX_WALL_SECONDS`). `MyAgent.is_done()` returns `True` — with a
`logger.warning` naming the game, elapsed time and action count — once
`time.time() - self.timer` crosses it. `Agent.main()` sets `self.timer`
before the first action and calls `is_done()` before every subsequent one,
so this bounds one game without touching the framework.

Guard on `getattr(self, "timer", 0)`: before `main()` runs, `timer` is the
class-level `0`, which would otherwise read as "infinitely overdue" and
refuse to play at all. Covered by
`tests/test_my_agent.py::test_wall_clock_guard_inert_before_main_sets_the_timer`,
plus tests for the over-budget, under-budget and disabled cases.

#### Note on concurrency
On Kaggle all games run concurrently in threads, so per-game elapsed time
is approximately overall elapsed time — a 2 h per-game guard bounds the
whole run near 2 h, comfortably inside the ~9 h limit. This interacts with
ARC-HANDOFF-002 (owned elsewhere): if that is resolved by running games
sequentially instead, the per-game guard must be re-derived from the
9 h budget divided by the game count, not left at 7200.

---

### [P1] ARC-HANDOFF-009 — The deterministic fallback emitted one identical action forever

**Status:** **SOLVED** on `newest-update` · **Category:** Scoring

#### Problem
`zerx/policy._deterministic_fallback` returned the first legal entry of
`_FALLBACK_PREFERENCE` and nothing else. For any game where the model is
unreachable and there are no click candidates — `ls20`, for instance, never
has `ACTION6` legal — that is the same action on every single step for the
entire game.

#### Evidence
A real local run (`scripts/play_local.py --game ls20 --max-steps 120`,
`backend=fake`) produced **121 consecutive `ACTION1`s**. This is strictly
worse than the upstream random baseline: it gathers no evidence, cannot
leave the start state, and starves `TransitionLedger` /
`DeadSignatureTracker` / `ExactStateMemory` of any signal to learn from.

#### Fix
The preference list is filtered to the legal set and indexed by
`actions_taken % len(ordered)`. The function stays pure and deterministic —
same inputs, same output, no new state — while actually exploring the legal
set. `decide()` passes the `actions_taken` it already receives.

#### Verification
Same live `ls20` run after the fix (60 steps): `ACTION1` 16, `ACTION2` 15,
`ACTION3` 15, `ACTION4` 15. Tests:
`tests/test_policy_decide.py::test_deterministic_fallback_rotates_instead_of_repeating_one_action`
(asserts diversification *and* that it is deterministic, not random), plus
single-legal-action and RESET-only edge cases.

#### Consequence for `exact_state_suppression_on`
Three tests in `tests/test_my_agent_exact_state.py` were asserting the old
repeat-forever behavior as the baseline the suppression feature escapes
from. Rotation now subsumes suppression on the deterministic-fallback path,
so those tests were rewritten to drive the feature through the path where
it still genuinely applies — the heuristic path, where an unchanging frame
makes the policy re-propose the identical `ACTION6:x,y` every step. The
swap and the cascade-past-a-second-suppressed-alternative case are both
still covered, and one test now asserts the rotation itself with
suppression off.

---

### Checked and deliberately not changed

- **`vc33`'s all-`ACTION6` fallback loop is not a stuck loop.** A traced
  40-step run (`ZERX_TRACE_EXPORT_PATH`) shows the coordinates do move —
  `(29,0)`, `(27,0)`, `(25,0)`, `(24,0)`, … — so the candidate ranking is
  responding to the frame. They are all on row `y=0` though, which is the
  HUD-versus-gameplay blind spot already recorded as `exp-150-duck-tools`
  Variant A scope. Left there; not reopened here.
- **ARC-HANDOFF-005 / -006** (root `pytest` collection, unregistered marks,
  `Config.from_env` crashing on a malformed value, the 1-of-3 Cerebras
  lockout, the literal-only secret scanner) — all still open, all
  deliberately out of this branch's scope per the owner's split.


---

## Branch `newest-update`, round 2 — 2026-08-06

Scope was widened by the human owner from round 1's "everything except the
Kaggle blockers" to **all remaining open items**, explicitly to maximize
scored performance before submitting.

- Suite: **428 passed, 0 failed** (round 1 ended at 392; +36 net-new tests).
  A bare `pytest` from the repo root now works, so that is the command.

### [P0] ARC-HANDOFF-010 — The agent was never told what its own actions did

**Status:** **SOLVED** · **Category:** Scoring / core capability

#### Problem
Every prompt showed the current board, the object table, the ranked click
candidates and the legal actions — and nothing about the *outcome* of
anything the agent had already tried. `TransitionLedger` had been recording
`changed_pixels`, `change_bbox`, `score_delta`, `terminal` and
`repeated_state` for every step since Day 1; **nothing ever read them.**

This is the single largest capability gap found so far, and it compounds
with a deliberate design constraint: `build_prompt` intentionally does not
describe what `ACTION1`–`ACTION5` do, because AGENTS.md forbids hard-coding
semantics that vary per game. Observed outcomes were therefore the *only*
channel through which the model could ever learn what an action means — and
that channel was closed. The model could not distinguish a no-op from a
useful move and had no reason to stop re-proposing a dead action.

#### Fix
`zerx/transitions.py` gains `render_transition_history(records, limit=8)`,
rendered into `build_prompt` under "What your recent actions actually did",
followed by an explicit instruction not to repeat an action the history
shows changed nothing. `MyAgent` keeps the last 20 records in a deque and
passes them through `decide(recent_transitions=...)`. Bounded to 8 lines in
the prompt so it stays small next to the 64x64 grid.

#### Verification
Live `vc33` run, real engine, reconstructed from the exported trace:
```
- ACTION6(x=47, y=46) -> changed 40 cells in region (x 1-3, y 2-4)
- ACTION6(x=31, y=0)  -> changed NOTHING on the board
- ACTION6(x=29, y=0)  -> changed 40 cells in region (x 1-3, y 2-4)
```
14 tests in the new `tests/test_evidence_loop.py`.

---

### [P1] ARC-HANDOFF-003 — Four ablation flags cannot change behaviour

**Status:** **SOLVED** (was UNRESOLVED) · **Source:** ARC-AUDIT-008/009/010/011

All four are now wired rather than removed, because each had a real use
once the evidence loop above existed to feed it:

- **`memory_on`** (defaulted `True` while controlling nothing) — the
  `summarizer=lambda prev, ctx: prev` no-op is replaced by
  `zerx/transitions.summarize_transitions`, which aggregates the ledger into
  per-action verdicts ("changed the board" vs "did nothing every time
  tried"). Deliberately **model-free**: AGENTS.md warns against turning
  reflection into a second unbounded reasoning loop, so this costs no extra
  call and no latency. It refuses to write off an action that worked even
  once — a single no-op is weak evidence.
- **`structured_memory_on`** — `render_for_prompt` had zero production
  callers, so the flag ran a full `perceive()` flood-fill every action to
  feed a no-op whose output was discarded. It now both feeds a real
  summarizer (confirmed rules / notable failures derived from the same
  ledger) and reaches the prompt via `decide(structured_memory=...)`.
- **`arbiter_on`** — was unsatisfiable: `select_candidate` gated on
  `config.arbiter_on and arbiter is not None`, and `decide()` passed no
  arbiter. It now passes the backend as the arbiter when the flag is on.
  Still off by default, and `select_candidate` still falls back to the
  deterministic pick on any arbiter failure.
- **`duck_objects_on`** — appeared only in `config.py` and the ablation
  matrix. It now selects whether each transition gets a semantic label from
  `zerx/scene.py`'s `classify_transition` (`HUD_ONLY`, `OBJECT_MOVE`,
  `RECOLOR_OR_TRANSFORM`, …) inside the evidence block. Off by default: the
  classifier segments and boundary-traces both frames, which is real
  per-action CPU. Injected as a callable so `zerx/transitions.py` never
  imports `zerx/scene.py`, and wrapped so a classifier failure degrades to
  "no label" instead of breaking the evidence loop it annotates.

A guard test (`tests/test_run_ablation.py`) now asserts every key in
`_CONFIG_ENV_MAP` is read somewhere in `zerx/` or `agent/`, with a short
allowlist for the three genuinely non-behavioural fields (`experiment_id`
is metadata; `competition_mode`/`internet_enabled` are validation-only) —
and it separately proves those two really do enforce the lockout rather
than merely being excused.

---

### [P1] ARC-HANDOFF-002 — Concurrent game threads share mutable `GameAction` singletons

**Status:** **SOLVED** (was UNRESOLVED) · **Source:** ARC-AUDIT-007

The handoff offered two options: run one game per process (costs wall-clock
against the ~9 h budget) or patch the vendored framework (means shipping a
modified framework). **Neither was taken** — a third option closes the
window with no framework patch and no loss of concurrency.

The hazard is the gap between `choose_action` calling `.set_data(...)` on
the shared enum member and `do_action_request` reading `action.action_data`
later. `MyAgent` now stores its own intended action in `_pending_submit` and
overrides `take_action` to re-apply it **and** call through to the read,
both inside one process-wide `_ACTION_SUBMIT_LOCK`. Only the submit itself
is serialized; perception, the model call and the rest of `choose_action`
stay fully concurrent.

The exception path clears `_pending_submit`/`_last_reasoning`, so a
crash-recovery action can never resubmit the previous step's payload.

`tests/test_concurrency_safety.py`: four agents on four threads, each
corrupting the shared singleton before its own submit, asserting every
submitted coordinate belongs to the agent that chose it; plus a test that
the lock is genuinely *held* at the moment the environment is stepped —
because the re-apply alone would pass single-threaded, and the lock is what
makes it correct under `Swarm`.

---

### [P0] ARC-HANDOFF-001 — The Kaggle submission notebook contains no model

**Status:** **SOLVED IN CODE — NOT YET VERIFIED ON KAGGLE** · **Source:** ARC-AUDIT-003/015

All four recommended-fix steps are implemented:

1. `ACCELERATOR` is `"rtx6000"`, not `"t4"`.
2. `notebooks/kernel-metadata.json`'s `model_sources` is populated, and is
   now *derived* from a single constant in `scripts/build_notebook.py`
   rather than hand-edited, so it cannot silently drift back to `[]`.
3. New offline vLLM install cell (`pip install --no-index --find-links`
   against an attached wheel dataset) and a serve cell that resolves the
   model directory under `/kaggle/input` by globbing for `config.json` —
   rather than hardcoding a version number that changes on re-publish — and
   launches vLLM with the **Colab-validated 8-bit fp8 flags**, not
   re-derived ones.
4. `ZERX_BACKEND=gemma_kaggle`, `ZERX_PLATFORM=kaggle`,
   `ZERX_MODEL_REVISION` and `ZERX_GEMMA_BASE_URL` are exported on the
   `main.py --agent myagent` line.
5. A readiness gate raises `SystemExit` if the server never answers. This is
   the point of the whole item: the old failure mode was silent
   heuristics-only play, and a loud early failure is strictly better than a
   meaningless score.

Tests: the built bundle's `select_backend(Config.from_env(kaggle_env))`
returns `GemmaModelBackend`; the run cell exports the env vars; every `pip
install` in the notebook carries `--no-index`; `model_sources` is non-empty;
`enable_internet` is `False`.

**Two things a human must do before pushing — this is not finished
otherwise:**
- `KAGGLE_WHEEL_DATASET` is empty. `scripts/build_notebook.py` prints a
  warning at build time and the notebook hard-fails at run time rather than
  degrading, but a vLLM wheel dataset must actually be built, attached, and
  named in that constant.
- `KAGGLE_MODEL_SOURCE` is set to the documented handle
  `google/gemma-4/transformers/gemma-4-31b-it`. **Confirm the exact string
  in Kaggle's own "Add Input" panel** — it could not be verified from this
  environment, and neither could any of the serving behaviour.
- `kernel-metadata.json`'s `id` still says `REPLACE_WITH_YOUR_USERNAME`.

---

### [P3] ARC-HANDOFF-005 — Root `pytest` broken, marks unregistered

**Status:** **SOLVED.** New root `pytest.ini` with `testpaths = tests` and
both marks registered. A bare `pytest` from the repo root now collects only
`tests/` and passes (428), with no `PytestUnknownMarkWarning`, so AGENTS.md's
documented command works as written and a typo'd `-m` name is an error
instead of silently selecting nothing.

---

### [P3] ARC-HANDOFF-006 — Config/scanner hardening

**Status:** **SOLVED**, all three items.

1. `_env_int`/`_env_float` re-raise with the offending variable *and* value
   named (`ZERX_BUDGET_SOFT_CAP='abc' is not a valid integer`). Raising, not
   silently defaulting — a silent fallback would hide a misconfigured
   experiment and make its results quietly wrong.
2. The cerebras_dev lockout now implements all three documented conditions
   (`platform=kaggle`, competition mode active, internet disabled) via two
   new `Config` fields, with a test per condition.
3. `zerx/secret_scan.py` gains value-based patterns (`csk-`, `sk-`, bearer
   headers, credentials assigned to api-key/token variables) on top of the
   two name-based ones — a leaked key literal pasted without its variable
   name previously shipped clean. A test asserts the scanner does not flag
   its own pattern definitions, since the build gate scans every bundled
   zerx module.

---

### Config changes worth knowing

New fields: `max_actions`, `max_wall_seconds`, `competition_mode`,
`internet_enabled`. Changed default: `budget_soft_cap` 50 → 400.
**`Config.config_hash()` therefore differs from every pre-`newest-update`
record** — experiment records across that boundary are not hash-comparable.

### Still not done

- **No Gemma-backed run has been made.** Everything above is verified
  against the real local engine with `backend=fake`, unit tests, and the
  built notebook's contents. The precision, throughput and per-action
  latency of the Kaggle serving path remain unmeasured.
- **Time one Gemma action on the target card before submitting**, then set
  `ZERX_MAX_ACTIONS` from that measurement. ~25 games × 400 actions ≈ 10,000
  calls; at 1.5–3 s each that is 4–8 h against a ~9 h limit. The 400 default
  is a starting point, not a calibrated value.
- `vc33`'s all-`ACTION6` heuristic play still concentrates on row `y=0`
  (`exp-150-duck-tools` Variant A scope). `duck_objects_on` now at least
  labels those transitions `HUD_ONLY` in the evidence block, which is the
  information needed to act on it, but nothing yet acts on it.
