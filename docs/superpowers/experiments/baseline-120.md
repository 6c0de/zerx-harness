# baseline-120 — real-game validation (Reki-core: reflection + click proposals + soft failure memory)

- Date: 2026-08-05
- Base commit: `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb` (branch `feat/baseline-120-colab-validation`)
- Game sample: `ls20, vc33, su15, tn36, ka59, lf52, tr87, sc25` — 8 of the
  25 documented public games. Keeps the existing `ls20`+`vc33` precedent
  `docs/superpowers/plans/parallel-baseline-120/README.md`'s own
  "concrete, empirical finding" measured this rung's 0.0/0-levels/all-
  `ACTION6` fallback-only "before" reference against (see that file), plus
  6 more games spread across the documented 25-game list
  (`su15, bp35, wa30, tn36, cd82, g50t, ka59, cn04, dc22, vc33, re86,
  sp80, lf52, ft09, sb26, tr87, m0r0, r11l, sc25, ls20, tu93, lp85, s5i5,
  ar25, sk48`) for per-game regression coverage, per `AGENTS.md`'s
  "repeated seeds/configurations" and "per-game regressions" language.
  `max_steps_per_game = 100`, deliberately below `scripts/play_local.py`'s
  200-step default: no prior per-decision-latency measurement exists in
  this repo for the 31B model (`baseline-100.md`'s own record only
  validated environment/packaging, not a timed play-through), so 8 games
  × 200 steps risked exceeding a single Colab session at an unknown
  latency. 100 was chosen as a documented, conservative trade-off — a
  human running the real Colab session should watch the first 1–2 games'
  wall-clock time and abort/reduce `GAME_SAMPLE` or
  `MAX_STEPS_PER_GAME` if the full sample is projecting past a reasonable
  single session (a rough a-priori bound: at an assumed 5–30s per model
  decision — unmeasured, hence the wide range — 8 × 100 = 800 decisions
  is roughly 1.1–6.7 hours; the true figure should be recorded here once
  a real run happens).
- Seeds: the local `arc_agi` public games are deterministic per game, not
  randomized per run — "repeated" here means repeated full playthroughs
  of the same game set across the dev-lane proxy and the Colab run, not
  RNG seed variation (matching the parallel-baseline-120 README's own
  note on this).

## Part A — notebook + tooling (this session, no GPU/Cerebras dependency)

`scripts/build_colab_notebook.py`'s `smoke_game_cell` rewritten from a
one-game subprocess call (`!python3.12 scripts/play_local.py --game ls20
--max-steps 50`) to an in-process loop over the `GAME_SAMPLE` module
constant, driving `MyAgent` directly (the same construction pattern
`scripts/play_local.py` itself uses: `arc_agi.Arcade(OperationMode.NORMAL)`,
`arc.make(game_id)`, `MyAgentCls(...)`, `agent.main()`), so that
`save_results_cell` — running in the **same** Colab kernel — can call
`arc.get_scorecard()` afterward and read real per-game results. A child
process's in-memory `Arcade`/scorecard state would have been unreachable
from a later notebook cell, which is exactly why `baseline-100.md`'s own
record could only validate environment/setup, never the actual per-game
outcome.

`save_results_cell` now computes, per game: `state`, `levels_completed`,
`actions`, `wall_time_seconds`, and (via
`EnvironmentScorecard.find_environment(game_id)`, matching
`docs/superpowers/plans/parallel-baseline-120/README.md`'s frozen
interface) the real `rhae` score and any `rhae_message` (e.g. "Human
baseline actions are not available for this environment" when
`EnvironmentInfo.baseline_actions` is empty for a given game — surfaced
explicitly rather than silently reported as a genuine `0.0`). A single
game raising an exception is caught per-game and recorded
(`"exception": repr(exc)`), so one bad game does not lose results already
collected for earlier games in the sample. The full per-game breakdown is
written to Drive under `"per_game"`, alongside the existing aggregate
score — not just an aggregate as before.

- Full local suite: **268 passed, 0 failed**
  (`.venv/Scripts/pytest.exe tests/ -q`) — grown from the confirmed
  261-passing baseline (261 − 1 superseded `play_local.py`-subprocess test
  + 8 new tests in `tests/test_build_colab_notebook.py`covering the game
  sample, the in-process multi-game loop, per-game exception isolation,
  and real RHAE capture).
- Notebook regenerated cleanly
  (`.venv/Scripts/python.exe scripts/build_colab_notebook.py`), JSON
  well-formed, 7 cells (unchanged count — only cell *content* changed).

## Cerebras prompt/parse sanity check (Part A step 3)

**Skipped.** No `CEREBRAS_API_KEY` was present in this session's shell
environment (checked twice: once before writing the plan, once
immediately before this step — both `absent`, without printing the
variable's value at any point). Per
`docs/superpowers/plans/parallel-baseline-120/person-4-colab-validation.md`'s
own explicit instruction ("if you genuinely cannot get a key, say so
explicitly in your status update rather than silently skipping it") this
is recorded here rather than silently omitted. Each teammate uses their
own Cerebras credential per `AGENTS.md`'s team contract — whoever next
works this branch with a key available should construct
`CerebrasDevBackend` directly, call `zerx.policy.build_prompt`/`parse_action`
against a synthetic `PerceptionResult`/`MemoryState` (no game loop, no
heuristics, no memory refresh — see this plan's Task 4 for the exact,
ready-to-adapt script), and record the result here before Part B's full
sweep.

## Part B — dev-lane `cerebras_dev` sweep via the real harness

**Blocked, doubly.** Two independent preconditions were both unmet this
session:

1. Track 1's backend-selection factory
   (`zerx.model_backend.select_backend`, branch
   `feat/baseline-120-backend-wiring`) was not found on `origin` or
   locally (`git ls-remote --heads origin feat/baseline-120-backend-wiring`
   returned nothing; only `origin/feat/baseline-120-eval-harness` — Track
   2 — exists as of this session). Without it,
   `agent/my_agent.py:156`'s `__init__` still unconditionally constructs
   `GemmaModelBackend` regardless of `Config.backend`, so setting
   `ZERX_BACKEND=cerebras_dev` and running the real harness would silently
   fall through to `fallback_heuristic`/`fallback_deterministic` on every
   step — the exact `0.0`/all-`ACTION6` failure mode this plan's own
   README already measured as the pre-existing "before" reference, not a
   meaningful `cerebras_dev` signal.
2. No `CEREBRAS_API_KEY` was available in this environment (see above),
   so even a standalone, harness-bypassing call would have nothing to
   sanity-check against.

No sweep was run. Re-attempt once Track 1 is merged (or its branch is
directly pullable — `git merge origin/feat/baseline-120-backend-wiring`)
**and** a Cerebras credential is available in the executing session's own
environment.

## Part B — authoritative Colab Gemma-4-31B-it run

**Not performed this session.** Running the generated notebook
(`notebooks/colab_gemma_smoke.ipynb`) on a real Colab A100/L4 GPU runtime
requires a human to upload it at colab.research.google.com, attach a GPU
runtime, and run all cells — no tool available to this Claude Code
session can drive a live Colab browser session (`AGENTS.md`'s environment
split: Colab is a human-operated, model-loading environment, distinct
from this local, model-free development environment). The harness-level
`cerebras_dev` sweep above also did not run (see "doubly blocked" above),
so no real per-game model-in-loop signal exists yet for this rung from
either lane.

## Conclusion

**`investigate`**, per `STRATEGY.md` §7.1 — the closest fit to "a
measurement/logging defect exists," following `baseline-100.md`'s own
precedent for recording an honest, partially-complete result rather than
forcing a `keep`/`revert` verdict. The notebook/tooling gap
`baseline-100.md` flagged ("per-game play outcome not captured") is now
closed (Part A: real per-game state/levels/actions/RHAE capture,
exception-isolated across an 8-game sample, full suite green at 268
tests). But no real per-game model-in-loop measurement exists yet for
`baseline-120` itself — neither the `cerebras_dev` dev-lane proxy (Part B,
blocked on Track 1 + a Cerebras credential) nor the authoritative Colab
Gemma run (Part B, requires human execution on Colab). This is explicitly
**not** `keep` (no evidence supports it — this record contains zero
real-model per-game results) and explicitly **not** `revert` (nothing was
measured to revert). Next step for whoever picks this up: (1) merge or
pull Track 1's branch once it exists, (2) obtain a personal
`CEREBRAS_API_KEY` and run Part A step 3 + Part B step 4 through the real
harness, (3) have a human run the now-ready notebook on a real Colab
A100/L4 GPU runtime with the same 8-game sample, then (4) re-decide this
conclusion from the Colab result specifically, per
`AGENTS.md`/`STRATEGY.md`'s hard rule that a Cerebras result never
substitutes for it.
