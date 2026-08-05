# `baseline-120-reki-core` real-game validation — read this first, all 4 people

Four people, four machines, four Claude Code sessions (Sonnet 5), same
GitHub repo (`https://github.com/6c0de/zerx-harness`), working
**simultaneously** on four tracks that together validate
`baseline-120-reki-core` against real games — the next unstarted rung on
`STRATEGY.md` §7's experiment ladder. This mirrors
`docs/superpowers/plans/parallel-day3/`'s structure and etiquette; read
that directory's `README.md` too if you haven't seen the pattern before,
but this file is self-contained.

## Verified starting point (do not re-derive, re-verify if in doubt)

- **Base master commit:** `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb` (`origin/master` matches exactly — confirmed via `git fetch` + `git diff master origin/master` producing no output at plan-writing time, 2026-08-05).
- **Local test suite:** `.venv/bin/pytest tests/ -q` → **261 passed, 0 failed** on this exact commit (matches `docs/HANDOFF.md`'s recorded count).
- **All 4 Day 3 tracks** (`baseline-115-exact-state-memory`, `baseline-130-hypothesis`, `exp-140-vlm-refinement`, `exp-150-duck-tools-ab`) are confirmed merged into `master` — verified by `git merge-base --is-ancestor` for all 4 remote branch tips, not by trusting `docs/HANDOFF.md`'s prose alone. Every new `Config` flag they added (`exact_state_suppression_on`, `duck_objects_on`, `candidate_count`, `structured_memory_on`) still defaults to its inert value on `master` (verified by reading `zerx/config.py` directly).
- **Kaggle:** no verifiable submission evidence exists anywhere in the repository (`.kaggle/access_token` absent, `notebooks/kernel-metadata.json` still has the placeholder `id`, no `submission*` artifact, no receipt/score in any doc). Kaggle work is untouched by this stage — see "Kaggle / external gate" below.

### A concrete, empirical finding this plan is built on

This session ran `.venv/bin/python scripts/play_local.py --game ls20,vc33 --max-steps 50` against the real local game engine on the verified base commit (no code changes). Result: **both games finished at `state=GameState.NOT_FINISHED`, `levels_completed=0`, and every single action across both 50-step runs was `ACTION6`** (no variation), with **`Aggregate scorecard score: 0.0`**. Root cause, confirmed by reading `agent/my_agent.py:156`: `MyAgent.__init__` unconditionally constructs `GemmaModelBackend(self._config.model_revision)` — `Config.backend` (`"fake"|"cerebras_dev"|"gemma_local"|"gemma_kaggle"`) is **read from environment but never consulted when choosing which backend class to build**. With no vLLM server listening on `localhost:8000`, every `backend.generate()` call fails and `zerx/policy.py`'s `decide()` silently falls through to `fallback_heuristic`/`fallback_deterministic` every single step — the harness never crashes (that part of `baseline-110`'s "no regressions" promotion criterion holds), but no real reasoning ever ran. This is exactly the gap `docs/HANDOFF.md`'s "Known failures" #1 already flagged ("Whichever track adds a backend-selection factory must forward `platform=config.platform` explicitly") — this plan's Track 1 closes it. Treat `0.0` aggregate score / `0` levels completed / all-`ACTION6` as this stage's **measured "before" reference**, not a guess.

## Why real-game validation, and why this exact scope

`STRATEGY.md` §7's ladder:

| ID | Status on `master` today |
|---|---|
| `baseline-100-minimal` | Built (Tasks 1–15) |
| `baseline-110-evidence` | Built (transition ledger) |
| `baseline-115-exact-state-memory` | Code merged, flag off by default, **never exercised against a real game** |
| `baseline-120-reki-core` | **Not started** — this plan |
| `baseline-125-phase-control` | Blocked on `baseline-130` (done) but not started — out of scope, see below |
| `baseline-130-hypothesis` | Code merged, flag off by default, never exercised against a real game |
| `exp-140-vlm-refinement` | Infra merged, off by default, no A/B run yet |
| `exp-150-duck-tools` A/B | Code merged, flag off by default, Variants C/D not started |

`docs/HANDOFF.md`'s own "Exact next action" #1, written by the session that
did the Day 3 integration, already names `baseline-120-reki-core` as the
natural next step, with the explicit reasoning that **everything built in
the 4 merged tracks is currently local, model-free, and off-by-default —
no track has yet been exercised against an actual game.** `baseline-120`'s
promotion criterion ("better completion/action efficiency on held-out
seeds/games") requires a working, measured reference run of the
**pre-existing** Reki-core loop (reflection memory, ranked click
candidates in-prompt, graded soft-failure evidence via
`DeadSignatureTracker` — all built since Day 1/2, all currently unwired
against a real backend) — this is the reference every later ladder rung
(`baseline-115`'s "fewer repeated no-ops," `baseline-130`'s "fewer
repeated probes," `exp-140`/`exp-150`'s ablations) needs to be compared
against. Skipping straight to those without this reference would violate
`STRATEGY.md` §7.1's rule against promoting on anything but a real,
reproducible comparison.

**This plan's scope is `baseline-120-reki-core` only.** It does not touch
`baseline-125`, does not touch `baseline-130`'s structured memory flag,
does not run `exp-140`/`exp-150`'s ablations, and does not touch Kaggle.
Per this project's own rule (`AGENTS.md`, `STRATEGY.md` §7.1): don't
promote a later rung before its predecessor has a real, evidenced
measurement.

## The 4 tracks

| # | Track | Branch | Owns |
|---|---|---|---|
| 1 | Backend selection wiring | `feat/baseline-120-backend-wiring` | `zerx/model_backend.py` (new factory), `agent/my_agent.py`'s backend-construction line, one new `Config` field |
| 2 | Real-game eval harness | `feat/baseline-120-eval-harness` | `eval/run_ablation.py`'s missing runner, its tests |
| 3 | Local regression & fallback-loop investigation | `feat/baseline-120-local-regression` | new test file(s) exercising all 25 public games locally, no GPU |
| 4 | Colab execution, experiment record, docs | `feat/baseline-120-colab-validation` | `scripts/build_colab_notebook.py` extension, `docs/superpowers/experiments/baseline-120.md`, status doc updates |

Fill in the "Owner" column (human name) in your own copy before starting;
this file stays generic.

## Why these 4 and how they relate

Unlike Day 3's four ladder rungs (which were deliberately chosen because
**none** depended on another), `baseline-120` is one rung being split by
**architectural layer**, not by independent feature. Track 1 (backend
selection) is a real prerequisite for Track 4's actual Colab run, and
Track 2's harness is exercised for real only once Track 1's factory
exists. This plan makes that honest rather than forcing artificial
parallelism — see "Track dependency graph" below for exactly what can run
concurrently vs. what's gated, and note that even the gated work has
substantial, real, independently-committable prep that does not wait.

## The one rule that keeps 4 simultaneous branches mergeable (same as Day 3)

**Every change ships additive-only in shared files, and each shared file
has exactly one owner this round** (see the ownership matrix below — this
round has fewer forced shared-file conflicts than Day 3, since only Track
1 touches `zerx/config.py` and only Track 1 touches `agent/my_agent.py`).

### `zerx/config.py`

Only Track 1 touches this file this round. If your own track discovers a
genuine need for a new `Config` field, stop and re-read "Ownership
matrix" — flag it in your status update rather than editing a file you
don't own.

### `agent/my_agent.py`

Only Track 1 touches this file this round (a single line in `__init__`:
replacing `self._backend = GemmaModelBackend(self._config.model_revision)`
with a call through Track 1's new factory). No other track adds a
delimited comment-banner block here this round — if your track's logic
needs agent-level wiring, it belongs in your own new module or script
instead (see your `person-N-*.md` file).

### Any new module, script extension, or test file

No etiquette needed beyond "don't edit a file another track owns" — see
the ownership matrix.

## Definition of done (every track)

- A short plan exists at `docs/superpowers/plans/2026-08-05-baseline-120-<your-track>.md`,
  written via the `superpowers:writing-plans` skill — your scope is
  already fully specified in your `person-N-*.md` file and in this
  `README.md`, so go straight to `writing-plans` without a separate
  brainstorming session, exactly like Day 3's tracks did.
- TDD throughout (`superpowers:test-driven-development`).
- New tests pass; the **full existing suite still passes** (261 tests as
  of this plan — confirm the count hasn't silently dropped).
- No behavior change for any existing test with no env vars set — your
  work either adds new, independently-tested surface, or fixes a
  confirmed, documented bug (Track 1's factory fix, Track 3's
  investigation) with its own new regression test proving the fix.
- No changes to `scripts/build_notebook.py` (the **Kaggle** submission
  notebook builder — different from `scripts/build_colab_notebook.py`),
  `.kaggle/`, `make submit`, or anything touching `CEREBRAS_API_KEY`
  handling beyond what Track 1's factory forwards — out of scope for all
  4 tracks.
- Commit messages follow the existing style in `git log` (imperative,
  explains why not just what).
- Push to **your own branch only** — do not merge to `master` yourself.
  One person (see `INTEGRATION.md`) merges all 4 in sequence.
- Update `docs/HANDOFF.md`'s status area with a one-line update for your
  track when done — not a rewrite. Do not edit `STRATEGY.md`; that's the
  integration owner's job, after the real experiment numbers exist (Track
  4's deliverable).

## Track dependency graph

```text
Track 3 (local regression, no GPU)  ────────────────────────────► independent, mergeable any time
Track 1 (backend factory) ──────┬──► Track 2 (eval harness, real integration test)
                                 │
                                 └──► Track 4 (Colab run) ──► docs/superpowers/experiments/baseline-120.md
```

- **Track 3 has no dependency on any other track.** It uses the harness
  as it exists today (including deliberately constructing
  `FakeModelBackend`/heuristic-only runs directly in its own test code,
  not through Track 1's factory) and can merge first.
- **Track 2 can write and unit-test its runner against a `FakeModelBackend`
  from day one** — it does not need Track 1's actual code to exist to make
  progress, only the frozen interface below. Its one real-integration test
  (proving the runner works end-to-end against the real local engine) is
  the only piece that benefits from Track 1 having already landed; write
  it, but expect to confirm it during integration if Track 1 isn't merged
  yet on your machine.
- **Track 4's notebook/game-list/results-schema work, and a standalone
  prompt/JSON-parse sanity check against `CerebrasDevBackend` (constructed
  directly, bypassing `agent/my_agent.py` entirely), do not depend on
  Track 1 or 2** and can proceed in parallel using `os.environ["ZERX_BACKEND"]
  = "gemma_local"` exactly as the existing `smoke_game_cell` already does
  (this happens to route through `GemmaModelBackend` correctly today even
  without Track 1's fix, because `gemma_local` is the one backend name
  that already matches the hardcoded class — see "A concrete, empirical
  finding" above). **Track 4's full `cerebras_dev` sweep through the real
  harness (the actual first fast-iteration signal for this stage — see
  its own file, Part B step 4) needs `select_backend` to exist**, which it
  can get either from Track 1 merging first, or by pulling Track 1's
  feature branch directly (`git merge origin/feat/baseline-120-backend-wiring`)
  into their own branch ahead of the official master-merge order — the
  4 people are on 4 different machines working concurrently in wall-clock
  time; `INTEGRATION.md`'s merge *order* is about the sequence branches
  land on `master`, not about when each person is allowed to start using
  another track's already-pushed code. **The real Colab run and the final
  written experiment record remain genuinely gated** on Track 1 actually
  existing somewhere Track 4 can build against, and benefit from Track 2's
  `run_games` for a consistent record format — say so plainly in status
  updates rather than forcing a premature "done."

## Interface / data contracts (frozen now, so 2/3/4 can code against them immediately)

**Track 1 ships this function signature in `zerx/model_backend.py`** —
treat it as frozen from the start of this plan, not just after Track 1
merges:

```python
def select_backend(config: Config) -> ModelBackend:
    """Construct the ModelBackend named by config.backend
    ('fake' | 'cerebras_dev' | 'gemma_local' | 'gemma_kaggle'),
    forwarding config.platform to CerebrasDevBackend so its existing
    platform=='kaggle' lockout applies. Raises ValueError for any other
    backend string. 'fake' returns FakeModelBackend() with an empty
    responses list (deliberate: every call raises, exercising the
    fallback chain) -- not a general-purpose scripted-response
    constructor; callers who need scripted responses still construct
    FakeModelBackend(responses=[...]) directly.
    """
```

**Track 2 ships this function signature in `eval/run_ablation.py`**:

```python
def run_games(
    config: Config,
    game_ids: Sequence[str],
    max_steps: int = 200,
) -> List[ExperimentRecord]:
    """Play each game_id locally via the real arc_agi Arcade + MyAgent
    (same NORMAL-mode pattern as scripts/play_local.py), driving MyAgent's
    backend/platform choice by setting the matching ZERX_* environment
    variables from `config` around construction (mirrors
    scripts/build_colab_notebook.py's smoke_game_cell pattern -- MyAgent
    itself still calls Config.from_env() internally; this function does
    not change that). Restores prior env state afterward. Returns one
    ExperimentRecord per game_id, with `rhae` populated from
    arc.get_scorecard()'s EnvironmentScorecard for that game when
    available, else None with the reason noted in your track's plan file.
    """
```

Both signatures are additive to existing files — neither replaces
anything.

## Ownership matrix

| File / area | Owner | Other tracks may... | Change rule |
|---|---|---|---|
| `zerx/model_backend.py` | Track 1 | import `select_backend` once merged | new code only, no edits to existing classes |
| `agent/my_agent.py` (`__init__`'s backend line) | Track 1 | read/exercise via tests, never edit | single-line change, no other edits this round |
| `zerx/config.py` (new `gemma_base_url` field) | Track 1 | read the field once merged | append-only, end of field list, matches Day 3's etiquette |
| `eval/run_ablation.py` | Track 2 | import `run_games`/`ExperimentRecord` once merged | new function only, existing `ExperimentRecord`/`write_records`/`sweep_configs` untouched |
| `tests/test_run_ablation.py` | Track 2 | — | additive tests only |
| new `tests/test_backend_selection.py` | Track 1 | — | new file |
| new `tests/test_real_game_regression.py` | Track 3 | — | new file |
| `scripts/build_colab_notebook.py` | Track 4 | — | extend `smoke_game_cell`/`save_results_cell`; do not touch `scripts/build_notebook.py` (Kaggle) |
| `docs/superpowers/experiments/baseline-120.md` | Track 4 | Track 2/3 may hand Track 4 their findings to cite | new file, single author for the final write-up |
| `docs/HANDOFF.md` | all 4 (one-line status each) + integration owner (final reconciliation) | — | append/edit only your own status line during your track; integration owner does the final rewrite |
| `STRATEGY.md` | integration owner only, after Track 4's real numbers exist | all tracks may read | not edited by any individual track |
| `zerx/heuristics.py` (only if Track 3's investigation confirms a genuine bug) | Track 3, conditionally | — | minimal, test-covered fix only if root-caused; if no bug is confirmed, this file is untouched |

If Track 3's investigation finds the root cause lies in `agent/my_agent.py`
instead of `zerx/heuristics.py` (plausible — see Track 3's own file), stop
and coordinate with Track 1 rather than editing a file Track 1 owns this
round; note the finding in your status update and let the integration
owner sequence the fix.

## Test strategy

- **Fast, no-GPU, every track:** `.venv/bin/pytest tests/ -q` must stay green throughout, and grow by each track's new tests.
- **Track 1:** unit tests over `select_backend` for all 4 backend strings, including the `platform=kaggle` + `cerebras_dev` rejection forwarding — no network, no real server.
- **Track 2:** unit tests over `run_games` using a scripted `FakeModelBackend` (env-injected) against the real local `arc_agi` engine for a small step count on one cheap game (`ls20` or `vc33`, matching existing precedent) — this is a real integration test (it does hit the real game engine, no network beyond what `arc_agi`'s `NORMAL` mode already does today, verified safe and read-only in this session).
- **Track 3:** the full 25-game crash-safety sweep (bounded step count, e.g. 30 — cheap enough to run in CI-like time; document your actual chosen cap and its wall-clock cost in your plan file) plus the deeper stuck-loop investigation on `ls20`+`vc33`.
- **Track 4:** no new local test suite requirement beyond a light test on the notebook-generation script extension (`tests/test_build_colab_notebook.py`, already exists, extend it) confirming the generated notebook contains the new multi-game loop and results-capture cells. The actual Colab run is not a `pytest`-gated activity — it's the Colab gate from `AGENTS.md`.

## Benchmark / real-game strategy

Three tiers, cheapest first — each one gates spending time on the next:

1. **Fast, no-GPU (everyone):** `Track 3`'s 25-game crash sweep — must
   pass (no unhandled exception, every game reaches a terminal or
   step-capped state) before anyone spends Cerebras or Colab time.
2. **Fast, cheap, real-model (`Track 4`, gated on Track 1 existing on
   some branch they can build against — not necessarily merged to
   `master` yet, see "Track dependency graph"):** the full `cerebras_dev`
   sweep through the real harness, across the documented game sample.
   Token budget for this is confirmed ample this stage, and the Cerebras
   "preview" lifecycle risk is an accepted, unrelated risk (see Track 4's
   own file) — so this tier is now the primary way `baseline-120` gets
   its first real signal, not an optional nicety. Its result is a
   dev-lane proxy, never the `baseline-120` score itself
   (`AGENTS.md`/`STRATEGY.md`'s hard backend-mismatch rule, unmoved by
   quota).
3. **Slow, authoritative (`Track 4` only, gated on Track 1+2 merging to
   `master`):** the actual `baseline-120` comparison run on Colab A100/L4
   with the real `google/gemma-4-31B-it` backend, reproducing the exact
   same game sample tier 2 used. This is the only tier whose result can
   be written into `docs/superpowers/experiments/baseline-120.md`'s
   `keep`/`revert`/`investigate` conclusion.

Game sample size is your call (the existing precedent is `ls20`+`vc33`;
STRATEGY.md's "repeated seeds/configurations" and "per-game regressions"
rules mean more than 2 games is stronger evidence; document your actual
sample and why in `docs/superpowers/experiments/baseline-120.md`).
Compare tier 3's result against this
plan's measured "before" reference (0.0 aggregate score, 0 levels
completed, fallback-only) — this is a real prior measurement on this
exact commit, not an invented control.

## Reproducibility rules

Same as `AGENTS.md`'s "Configuration and reproducibility" section,
unchanged by this plan: record experiment ID, base commit, model
revision/precision/backend, config hash, games/seeds used, and a
keep/revert/investigate conclusion per `STRATEGY.md` §7.1. `Track 4`'s
`docs/superpowers/experiments/baseline-120.md` is where this lives, using
the exact same fields `docs/superpowers/experiments/baseline-100.md`
already established the format for. Seeds: the local `arc_agi` public
games are not literally seeded (deterministic per game, not randomized
per run) — "repeated" here means repeated full playthroughs of the same
game set, not RNG seed variation; note this explicitly in the experiment
record rather than inventing a seed concept the harness doesn't have.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Today is Day 3 by commit-date timeline (`git log` shows the Day 3 split committed 2026-08-05, matching this session's system date) and `AGENTS.md`'s schedule says the first full Kaggle compatibility run should start "as early in [Day 3] as possible" — but `docs/HANDOFF.md` confirms Kaggle work has **not started at all**, not even Day 1's smoke submission | Out of scope for this plan (see "Kaggle / external gate" below) but flagged here as a real, human-owner-level schedule risk this plan does not resolve — surfaced in the final report, not silently absorbed into this stage's scope |
| Track 4's Colab run is gated on Track 1+2 | Track 4 has real, independently-committable prep work (notebook structure, results schema, a standalone Cerebras prompt/parse sanity check) that does not wait, plus a full `cerebras_dev` sweep through the real harness the moment Track 1's branch is pullable (not necessarily merged to `master` — see "Track dependency graph") — this is now the primary way this stage gets an early, cheap, real-model signal before Colab, not just optional prep — see Track 4's own file |
| Model-identity mismatch between Cerebras's `cerebras_dev` proxy and the exact deployed Gemma-4-31B-it | Independent of Cerebras's token quota or "preview" lifecycle status — Cerebras uses its own weight-only quantization; `AGENTS.md`'s hard rule stands regardless of budget: a `cerebras_dev` result is never recorded as the `baseline-120` score, only as a labeled dev-lane proxy alongside the real Colab Gemma result |
| Track 3's fallback-loop investigation might not find a fixable bug (e.g. it may be an inherent property of a heuristic-only run against a game requiring genuine reasoning) | Explicitly scoped as "investigate and document" first (`superpowers:systematic-debugging`); a confirmed non-bug ("this is expected heuristic-only behavior, not a defect") is an acceptable, valuable outcome — document it, don't force a fix |
| `arc_agi`'s local `NORMAL` mode fetches environment metadata from `https://three.arcprize.org` on first use (observed this session: "Got anonymous API key") | This is the same public, credential-free, read-only call `make list-games`/`make play-local` already make today — not a new dependency introduced by this plan, but worth knowing if a track's machine has no internet access when testing |
| RHAE (`arc_agi.scorecard.EnvironmentScoreCalculator`'s `((baseline_actions / actions_taken) ** 2) * 100`, capped at 115) requires `EnvironmentInfo.baseline_actions` to be populated for a given local game; if it's empty for some games, `EnvironmentScore.score` is `0.0` with an explicit `message` field, not a crash | Track 2's `run_games` must surface that `message` field in the `ExperimentRecord` (or a log line) rather than silently reporting a `0.0` RHAE indistinguishable from a genuine zero-completion result — document which of your chosen games actually have baseline data before treating any `0.0` as a real measurement |

## Measurable acceptance criteria

`STRATEGY.md` §7's own promotion text for `baseline-120` is
**"Better completion/action efficiency on held-out seeds/games"** — no
numeric threshold is written anywhere in this repository for this rung
(unlike, say, `AGENTS.md`'s ~1% aggregate-leaderboard target, which is a
different, later-stage number). That absence is a real gap in the
project's documented decisions, not something this plan can fill on the
repository's behalf.

**PROPOSED THRESHOLD (this plan's own proposal, not a repository
decision — confirm or replace it with the human owner before treating it
as binding):**

- At least one of the sampled public games reaches `GameState.WIN` (any
  positive RHAE) using the real Gemma backend, on at least one of the
  documented playthroughs — i.e., genuine measured progress over this
  plan's confirmed 0.0/0-completions fallback-only reference, not
  "the harness ran without crashing" (that bar is already cleared today).
- No regression in the Track 3 crash-safety sweep (still 0 unhandled
  exceptions across all 25 games) after Track 1's backend-factory change
  lands.
- The full local suite (261 + new tests) stays green after every merge.

If Track 4's real run does not clear the first bullet, that is itself a
valid, recordable `STRATEGY.md` §7.1 outcome (`investigate` — e.g. "the
model reasons but the prompt/perception format doesn't yet produce
winning play on this game sample") — do not force a `keep` conclusion to
satisfy this proposed threshold artificially.

## Merge order

Smallest / most independent first, most dependent last — same logic as
`docs/superpowers/plans/parallel-day3/INTEGRATION.md`:

1. `feat/baseline-120-local-regression` (Track 3) — new test file only, no shared-file touches beyond a `docs/HANDOFF.md` one-liner.
2. `feat/baseline-120-backend-wiring` (Track 1) — small, foundational, unblocks 2 and 4's real integration.
3. `feat/baseline-120-eval-harness` (Track 2) — depends on 1 for its real-integration test to mean anything end-to-end.
4. `feat/baseline-120-colab-validation` (Track 4) — biggest, most dependent, carries the actual experiment record and status-doc updates; merges last, same pattern Day 3 used for `baseline-130`.

Full procedure, conflict expectations, and post-merge test gates: see
`INTEGRATION.md` in this directory.

## Final go/no-go gate

`baseline-120` is promotable (per this plan's proposed threshold above)
only when: all 4 branches are merged, the full suite is green, Track 3's
25-game crash sweep is clean, and Track 4's
`docs/superpowers/experiments/baseline-120.md` records a real Colab Gemma
run with an explicit `keep`/`revert`/`investigate` conclusion. The
integration owner then updates `docs/HANDOFF.md`'s "Exact next action" and
is the only one who touches `STRATEGY.md` (recording the actual outcome
against the ladder entry) — not automatic, a decision for the human
owner, exactly as `docs/HANDOFF.md`'s own precedent already states for
`baseline-120`'s current "recommendation, not automatic" phrasing.

## Kaggle / external gate — explicitly separate

Nothing in this plan submits to Kaggle, touches `.kaggle/`, runs
`make submit`, or edits `scripts/build_notebook.py`. `docs/HANDOFF.md`'s
Day 1 Kaggle-smoke-submission item and `AGENTS.md`'s Day 3
first-full-Kaggle-run item remain open, separate, human-owner-approved
actions — this plan does not schedule, start, or imply either. If the
human owner wants to run them, that is a distinct session with explicit
approval per `AGENTS.md`'s Kaggle gate, independent of whether this
plan's 4 tracks are done.
