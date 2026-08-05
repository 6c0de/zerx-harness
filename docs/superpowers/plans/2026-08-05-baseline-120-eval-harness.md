# `baseline-120-eval-harness` — plan

Track 2 of `docs/superpowers/plans/parallel-baseline-120/`. Scope: add
`run_games(config, game_ids, max_steps=200) -> List[ExperimentRecord]` to
`eval/run_ablation.py`, tested in `tests/test_run_ablation.py`. Frozen
signature from the plan README, reproduced in
`docs/superpowers/plans/parallel-baseline-120/person-2-eval-harness.md`.

## Approach

1. `run_games` sets `ZERX_*` env vars from `config` (mirrors `Config.from_env`'s
   names), constructs `arc_agi.Arcade(operation_mode=NORMAL)`, loads
   `MyAgent` via `importlib` (same pattern as `scripts/play_local.py`'s
   `load_my_agent_class`), plays each `game_id` (skip unresolvable ids,
   matching `play_local.py`), restores env in a `finally` block.
2. After all games play, call `arc.get_scorecard()` once, match
   `EnvironmentScoreList.id` (split on `"-"`, first segment) back to each
   `game_id`, and build one `ExperimentRecord` per requested game.
   `rhae`: if the environment's last run carries a `message` (no baseline
   data), record `rhae=None` and log the message; otherwise
   `rhae=env_score.score`.
3. Unobservable fields (`invalid_outputs`, `repairs`, `fallbacks`,
   `exceptions`) are not aggregated anywhere outside `MyAgent`'s internal
   `Decision` objects — record them as `0` is misleading, so document this
   as a known gap (comment in code + this plan) and leave them `0` with an
   explicit note, since `ExperimentRecord` is frozen/non-Optional there
   this round (adding an Optional wouldn't help distinguish without
   widening the schema, out of scope for Track 2 — Track 4 to note in the
   experiment record). `resets` comes from `EnvironmentScoreList.resets`.
   `wall_time_seconds` via `time.monotonic()` around each game's `agent.main()`.
   `actions_taken` via `agent.action_counter`. `levels_completed` via
   `agent.frames[-1].levels_completed`.
4. No exception swallowing around the per-game loop (per task spec) —
   only `arc.make()` returning `None` is treated as skip, matching
   `play_local.py`.

## Tests (`tests/test_run_ablation.py`, additive)

- real-engine smoke test: `run_games(Config(backend="fake"), ["ls20"], max_steps=5)`
  returns one record, `actions_taken > 0`, does not raise.
- empty `game_ids` -> `[]`, no engine call needed to fail.
- env var restoration: set an unrelated env var before call, assert
  unchanged after.
- `rhae` type: assert `record.rhae is None or isinstance(record.rhae, float)`
  for the smoke game; document actual observed behavior for `ls20`.

## Verification

`.venv/bin/pytest tests/ -q` stays green (261 + new). No edits outside
`eval/run_ablation.py` / `tests/test_run_ablation.py`.
