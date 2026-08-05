# Developer 2 — Real-game eval harness

**Read `README.md` in this directory first** — shared context, the frozen
interface contracts (including Track 1's `select_backend`, which you
depend on conceptually but not physically at merge time), and the
ownership matrix this file assumes.

- **Track:** Real-game eval harness for `baseline-120-reki-core`
- **Base master SHA:** `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb`
- **Branch:** `feat/baseline-120-eval-harness` (create from base SHA above)

## Purpose and expected outcome

`eval/run_ablation.py` already defines `ExperimentRecord`, `write_records`,
and `sweep_configs` — but its own module docstring says outright: **"The
actual 'play N local games with this config' loop is wired in once
agent/my_agent.py's harness adapter is exercised against real games...
this module owns the record format independent of that wiring."** That
wiring has never been written. `scripts/play_local.py` proves the pattern
works (real `arc_agi.Arcade` in `NORMAL` mode, real `MyAgent`, a real
per-game summary), but it's a standalone script, not a reusable function
that produces `ExperimentRecord`s other code (or a future ablation sweep)
can consume.

Your job: implement that missing loop as a real, tested, reusable function,
producing genuine `ExperimentRecord`s including RHAE pulled from the
`arc_agi` package's own scoring (`arc_agi.scorecard.EnvironmentScorecard`,
returned by `Arcade.get_scorecard()`) — not an invented metric. This is
what Track 4 uses for the actual Colab comparison run, and what future
ablation work (`exp-140`'s candidate-count sweep, `exp-150`'s Duck-tools
variants) will build on.

## Commands to run before starting

```bash
git fetch origin
git checkout -b feat/baseline-120-eval-harness 8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb
.venv/bin/pytest tests/ -q   # confirm 261 passed, 0 failed before you touch anything
.venv/bin/python scripts/play_local.py --list   # confirms your machine can reach arc_agi's NORMAL-mode game list (makes one read-only, credential-free network call to three.arcprize.org — same one `make list-games` already makes)
```

## Files you own this round

- `eval/run_ablation.py` — add the new function (below); do not restructure
  `ExperimentRecord`, `write_records`, or `sweep_configs`.
- `tests/test_run_ablation.py` — extend with new tests (existing file, but
  no other track touches it this round).

## Do not touch

`zerx/model_backend.py`, `agent/my_agent.py`, `zerx/config.py`,
`scripts/play_local.py` (read it for the pattern, don't edit it — it's
not owned by any track this round and should stay exactly as-is so it
remains a stable reference), `scripts/build_colab_notebook.py`.

## What you're building

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

Concrete implementation notes, grounded in what this session verified
directly by reading `scripts/play_local.py` and
`.venv/lib/python3.12/site-packages/arc_agi/scorecard.py`:

- `arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL)`, then
  `arc.make(game_id, ...)` to construct each environment, exactly as
  `scripts/play_local.py` does. You need `agent/my_agent.py`'s
  `MyAgent` class (import it the same way `scripts/play_local.py`'s
  `load_my_agent_class()` does via `importlib`, or add a small local
  helper — your call, document it).
- `MyAgent.__init__` calls `Config.from_env()` itself; it does not accept
  an injected `Config`. Rather than changing `MyAgent`'s constructor
  signature (which would require coordinating with Track 1, who owns
  `agent/my_agent.py` this round), set the matching `ZERX_*` environment
  variables from your `config` argument before constructing `MyAgent`,
  and restore the previous environment afterward (a context manager or
  try/finally — either is fine, but it must not leak env state into other
  tests or later calls in the same process). `zerx/config.py`'s
  `from_env` already lists every `ZERX_*` name; map each `Config` field
  to its matching env var name exactly as that file does.
- After playing all `game_ids`, call `arc.get_scorecard()` — verified this
  session (reading `arc_agi/base.py:515`) to return an
  `EnvironmentScorecard` (already computed via
  `EnvironmentScorecard.from_scorecard`), **not** the raw `Scorecard`.
  `EnvironmentScorecard.environments` is a list of `EnvironmentScoreList`,
  one per game actually played, each with a `.score` (the RHAE-style
  number, `((baseline_actions/actions_taken)**2)*100` capped at 115,
  averaged per level) and a `.actions`/`.levels_completed`. Match each
  `EnvironmentScoreList.id` back to your `game_ids` to build each
  `ExperimentRecord`.
- **RHAE availability is not guaranteed.** `arc_agi/scorecard.py`'s
  `EnvironmentScore.message` is set to strings like `"Human baseline
  actions are not available for this environment"` when
  `EnvironmentInfo.baseline_actions` is empty for that game — in that
  case the computed `.score` is `0.0` but it does **not** mean "zero
  performance," it means "unmeasurable here." Your `ExperimentRecord.rhae`
  should be `None` in that case (not `0.0`), and you should surface the
  `message` some way your caller (Track 4) can see — a log line at minimum;
  consider whether `ExperimentRecord` needs a new optional field for this
  or whether logging is sufficient for this stage (your call, document
  it — this is a real, non-trivial design decision, not busywork).
- `wall_time_seconds`, `invalid_outputs`, `repairs`, `fallbacks`, `resets`,
  `exceptions` are already fields on `ExperimentRecord` — populate what
  you can observe from the harness (e.g. `agent.action_counter` for
  actions taken, wall-clock via `time.monotonic()` around the play loop).
  Fields you genuinely cannot observe from outside `MyAgent` (e.g.
  per-decision repair/fallback counts — `Decision.source`/`.repaired` are
  internal to each `choose_action` call and not currently aggregated
  anywhere) should be documented as a known gap in your plan file rather
  than guessed at or silently left as a wrong default — `0` is a
  misleading value if the real count is unknown, not just unmeasured;
  say so explicitly rather than let a reader assume it means zero
  fallbacks occurred.

## Config field usage

You use the existing `Config` object your caller passes in — you do not
add any new `Config` field yourself (that would conflict with Track 1's
ownership of `zerx/config.py` this round). If you find you need one,
stop and flag it rather than editing that file.

## Tests

`tests/test_run_ablation.py` (extend the existing file):

- Existing tests (`ExperimentRecord.to_json_line`, `write_records`,
  `sweep_configs`) must keep passing unchanged.
- New: `run_games` against a real local `arc_agi` environment (`ls20` or
  `vc33`, matching existing precedent), `max_steps` small (e.g. 10, to
  keep the test fast), `config.backend="fake"` — asserts you get back one
  `ExperimentRecord` per game, `actions_taken` matches what the harness
  actually reports, and the function does not raise even though the fake
  backend never produces a real model response (this is intentionally the
  same fallback-heavy path this session observed directly — you are
  testing that your harness *records* that behavior faithfully, not that
  it produces winning play).
- New: environment-variable restoration — call `run_games` with a
  temporary env var already set to something unrelated before the call,
  assert it's unchanged after the call returns (proves you're not leaking
  state).
- New: a game with no available `baseline_actions` (if you can identify
  one among the 25 public games — check via `arc.get_environments()`'s
  `EnvironmentInfo.baseline_actions`) produces `rhae=None`, not `rhae=0.0`,
  in the resulting record. If all 25 public games happen to have baseline
  data (verify, don't assume), document that finding instead and adjust
  this test to use a constructed/mocked `EnvironmentInfo` if needed — your
  call, document the actual situation you found.

Keep the real-engine test(s) to a small step count and a single cheap
game — this suite must stay fast; do not add a 200-step or all-25-games
test here, that is Track 3's job with its own, separately-owned file.

## Verification commands

```bash
.venv/bin/pytest tests/ -q
.venv/bin/pytest tests/test_run_ablation.py -v
```

## Expected outputs

- `eval/run_ablation.py` gains `run_games`, nothing else changes.
- `tests/test_run_ablation.py` grows by a handful of focused tests, at
  least one of which genuinely exercises the real local game engine.
- Full suite green.

## Artifact and log locations

`run_games` itself does not write files (that's `write_records`'s job,
already built) — your tests may write to a temp directory
(`tmp_path` pytest fixture) if needed for `write_records` integration,
never to a path inside the repo.

## Performance / runtime bounds

Your real-engine test(s) should complete in well under 10 seconds each —
if `arc_agi`'s network call to fetch environment metadata is slow on a
given run, that's environmental, not a bug in your code; do not add
retry/timeout logic to work around it, that's out of scope for this
track.

## Edge cases

- `game_ids` containing an id `arc.make()` can't resolve — decide whether
  to skip (matching `scripts/play_local.py`'s own `if env is None: skip`
  behavior) or raise, and document your choice; consistency with the
  existing script's behavior is the safer default.
- Empty `game_ids` list — should return an empty list, not raise.

## Failure-mode behavior

If a single game's play loop raises an exception `MyAgent.choose_action`
itself didn't catch (should not happen, given `AGENTS.md`'s guarantee,
but this is exactly the kind of harness code that should not silently
swallow a real bug) — let it propagate rather than catching broadly and
returning a partial/misleading record. Do not add a blanket
`try/except Exception` around the whole per-game loop; that would hide a
genuine regression from whoever calls `run_games` next.

## Definition of done

- `run_games` implemented and tested per above.
- `docs/HANDOFF.md` one-line status update.
- Your own `docs/superpowers/plans/2026-08-05-baseline-120-eval-harness.md`
  plan file, written before coding.

## PR checklist

- [ ] `run_games` signature matches the frozen interface in `README.md` exactly.
- [ ] At least one test genuinely exercises the real local `arc_agi` engine (not all-mocked).
- [ ] `rhae=None` vs `rhae=0.0` distinction implemented and tested.
- [ ] Env var restoration tested explicitly.
- [ ] Full suite green, count reported in PR description.
- [ ] No edits outside "Files you own this round."

## Handoff format

Update `docs/HANDOFF.md` with branch, commit SHA, test count added, and
one sentence on the `rhae=None` vs `0.0` design decision you made (Track
4 needs to know this to interpret its own real run's records correctly).

## Merge preconditions

Full suite green. If Track 1 (`feat/baseline-120-backend-wiring`) has
already merged to `master` by the time you're ready, rebase and confirm
your real-engine test(s) still pass with `config.backend` actually
routed through `select_backend` (not just the pre-Track-1 hardcoded
path) — this is the point where your work gets its first real end-to-end
confirmation. If Track 1 hasn't merged yet, merge as planned; the
integration owner re-verifies this interaction during the Track 2 merge
step regardless (see `INTEGRATION.md`).

## Rollback approach

`run_games` is new, additive code with no callers yet inside this
repository (Track 4 is the first real caller, and merges after you) — a
`git revert` of your merge commit is clean with no downstream breakage.
