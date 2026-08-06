# Project handoff

Copy this template's structure for each handoff entry (newest first, or one
file per handoff under `docs/handoffs/` if the history grows long — not
needed yet). See `docs/TEAM_WORKFLOW.md` for the 5-day schedule this feeds
into.

- Updated at: 2026-08-06
- Current owner: (local session, Claude Code — `baseline-120` integration)
- Next owner: whoever picks up the two `baseline-120` prerequisites (visualizer,
  `build_prompt()` legal-actions fix) or the Day 1 Kaggle smoke submission —
  see "Exact next action" below — not auto-started
- Branch: `master` (all 4 `baseline-120` tracks merged in, sequentially, via
  `integration/baseline-120`, per `docs/superpowers/plans/parallel-baseline-120/INTEGRATION.md`)
- Commit: merge of `integration/baseline-120` into `master`, local only as of
  this update — **not yet pushed to `origin/master`**, pending explicit
  human-owner confirmation (see "Exact next action")
- Experiment ID: `baseline-120` (recorded, flagged `investigate` — see
  `docs/superpowers/experiments/baseline-120.md`; `baseline-100` remains
  `investigate` too, unchanged from before, see below)
- Config ID/hash: n/a for `baseline-120`'s dev-lane sweep — see
  `docs/superpowers/experiments/baseline-120.md` for the actual config used
  (`ZERX_BACKEND=cerebras_dev`, `ZERX_MODEL_REVISION=gemma-4-31b`)
- Sprint day (1–5): still Day 3 by commit-date timeline — `baseline-120`'s
  infrastructure is complete and its dev-lane signal is in, but the
  authoritative Colab run is explicitly postponed (see below), and Day 1's
  Kaggle smoke submission is still open — see "Exact next action"

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

Not started. No `make submit`, no Kaggle CLI call, no notebook push.
**Still open** from Day 1's exit condition — needs explicit owner approval
before it happens, per `AGENTS.md`'s Kaggle gate. Not blocking the 4
parallel tracks below (none of them touch Kaggle).

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
3. `parse_action(None, ...)` raises `AttributeError`, inert because
   `decide()` wraps the only real call site in `try/except Exception`.
4. `history` is computed in `agent/my_agent.py` and passed to
   `decide()`/`perceive()`, but `perceive()` ignores it — deliberate
   interface stability for future movement-delta perception.
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
   detail.
7. **`scripts/play_local.py`'s `MyAgentCls.MAX_ACTIONS = min(MyAgentCls.MAX_ACTIONS,
   args.max_steps)` can only ever *lower* the step cap**, never raise it
   above `MyAgentCls`'s existing default (80, inherited from the vendored
   base `agents.agent.Agent` class) — confirmed by Track 4's dev-lane
   sweep (2026-08-06): `--max-steps 100` silently capped the run at 80
   steps/game, not 100. Recorded as the actual, accurate step count in
   `docs/superpowers/experiments/baseline-120.md` rather than the
   originally intended one; not fixed (`scripts/play_local.py` unowned
   this round).
8. **`scripts/play_local.py:114` crashes with `UnicodeEncodeError` on
   Windows non-UTF8 consoles** when printing multi-game summaries — the
   final per-game summary line hardcodes a `→` character that the
   Windows `cp1254` console codepage cannot encode, so the script's loop
   over games terminates via an uncaught exception right after the first
   game. Confirmed by Track 3 (2026-08-05): this is why that track's own
   reproduction of the original `ls20`+`vc33` finding only ever completed
   `vc33` — `ls20` never actually ran in that reproduction. Track 3's own
   sweep drives `MyAgent` directly rather than through this script, so it
   was not exposed to the crash; not fixed (out of scope for all
   `baseline-120` tracks).

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

**`baseline-120` is not yet pushed to `origin/master`** — that push, and
any `STRATEGY.md` edit, require explicit human-owner confirmation before
this session proceeds (see top-of-file "Commit" note). Once confirmed and
pushed, this is the priority-ordered list of what's actually next; none
of the items below are started, all are recommendations for the human
owner to schedule:

1. **`baseline-120`'s own next steps, in the human owner's stated order**
   (see "Colab run — explicitly postponed" below for the full reasoning):
   (a) build the visualizer (reference:
   [github.com/Darkosxl/Agent_Harness_Example](https://github.com/Darkosxl/Agent_Harness_Example)
   — pygame grid + scrollable reasoning panel, pause/step-back-forward
   through a capped history buffer); (b) fix `zerx/policy.py`'s
   `build_prompt()` to include `legal_actions` in the prompt text (see
   "Known failures or risks" item 6); (c) only then re-run the dev-lane
   sweep and/or the authoritative Colab Gemma-4-31B-it run and write the
   real `keep`/`revert`/`investigate` verdict into `STRATEGY.md` §7 — the
   one `STRATEGY.md` edit this whole `baseline-120` effort authorizes,
   and only after that real number exists.
2. **A general `README.md`** documenting project usage — none exists yet;
   recommended by the human owner as a near-term, low-risk documentation
   gap, independent of the items above.
3. **A personal `ARC_API_KEY`** (separate from `CEREBRAS_API_KEY`) so
   local runs attribute to the human owner's account on
   `three.arcprize.org`'s web dashboard instead of an anonymous one —
   complements the visualizer (item 1a), doesn't change how anything
   runs locally. Recommended by the human owner, not started.
4. **A JSON-like export of played games**, for later offline inspection —
   per-game (or per-step) structured data including each decision's
   reasoning/raw model output, not just the final parsed action (today's
   `Decision` dataclass in `zerx/policy.py` and `ExperimentRecord` in
   `eval/run_ablation.py` both discard the raw model response once
   parsed). Recommended by the human owner as the natural data source for
   the visualizer's replay buffer too (same underlying trace, consumed
   either live or from a saved file) — worth designing together with item
   1a rather than as two unrelated features. Not started.
5. Kaggle Day 1 smoke submission is still open — get explicit approval
   before running it, independent of everything above.
6. `baseline-100`'s results-capture gap is now closed going forward by
   Track 4's Part A notebook rewrite (see "Tests executed and results"
   above) — no further action needed for that specific gap; a fresh Colab
   run would exercise the fix.
7. The 4 `feat/baseline-120-*` branches are fully merged into
   `integration/baseline-120` → `master` (locally) — safe to delete once
   the human owner confirms, not deleted automatically. Same standing
   offer for the 4 `feat/...` branches used for Day 3, already merged.
8. **Resume/fork from a recorded step — documented, not built.** See
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

## `feat/baseline-120-followups` — status (2026-08-06)

**Branch:** `feat/baseline-120-followups`, off `master` at `b405d3b` (the
`baseline-120-reki-core` integration commit summarized above). Not yet
merged to `master` — pushed to `origin` as a feature branch only, pending
a whole-branch code review and the human owner's explicit merge
go-ahead, per this project's standing branch-promotion rule. Implemented
via a 7-task plan
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
