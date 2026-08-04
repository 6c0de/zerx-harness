# baseline-000 — unmodified starter

- Date: 2026-08-04
- Upstream starter repo: https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter
- Upstream starter commit: `eeb1535404f321d280a8f9194bbc1d7aca5f05fc`
- Vendored framework repo: https://github.com/arcprize/ARC-AGI-3-Agents.git
- Vendored framework commit: `4743e7d0aaae0ded0d98a89a7e282e63564cd58b`
- Python version: `Python 3.12.13` (via `python3.12` on PATH / `uv python`, not the
  machine's default `python`, which is 3.14.3 and does not satisfy the
  starter's `arc-agi>=0.9.6` requirement)
- `arc-agi` installed version: 0.9.9 (pulls in `arcengine==0.9.3`)

## Agent/GameAction/Frame import paths (verbatim from `agent/my_agent.py`)

```python
from arcengine import FrameData, GameAction, GameState
from agents.agent import Agent
```

Contract (enforced by the vendored `agents.agent.Agent` ABC):
- Subclass `agents.agent.Agent`.
- Class must be named `MyAgent`.
- Implement `is_done(frames: list[FrameData], latest_frame: FrameData) -> bool`.
- Implement `choose_action(frames: list[FrameData], latest_frame: FrameData) -> GameAction`.

## Frame attribute names (verified against `arcengine/enums.py` + a live probe, not assumed)

`FrameData` is a pydantic `BaseModel` with these fields (only the ones our
adapter needs are listed):

- `frame: list[list[list[int]]]` — a list of animation *sub-frames* rendered
  while resolving one action (`arcengine`'s game loop can render more than
  one internal frame per external action). **The current 64×64 grid is
  `frame.frame[-1]`** (a `list[list[int]]`, rows of ints, colors 0–15), not
  `frame.frame` itself. Empty (`[]`) when nothing has been rendered yet
  (e.g. the very first call, state `NOT_PLAYED`).
- `state: GameState` — an **enum**, not a string. Values seen: `NOT_PLAYED`,
  `NOT_FINISHED`, `WIN`, `GAME_OVER`. Compare with `is`/`in`, not `==` against
  a literal string.
- `available_actions: list[int]` — the currently-legal **non-RESET** action
  ids for this game/level (e.g. `[1, 2, 3, 4]`). Verified: `RESET` (id 0) is
  **never** included in this list and is always implicitly legal regardless
  of its contents (`base_game.py`'s default `available_actions=[1,2,3,4,5,6]`
  excludes 0 and 7; `RESET` is handled unconditionally in the engine's
  `handle_reset()` path, not gated by this list). `ACTION7` (id 7) only
  appears here for games that actually expose it — confirms AGENTS.md's rule
  to never hard-code `ACTION7` as legal.
- `levels_completed: int`, `win_levels: int` — **no `.score` field exists** on
  `FrameData`. These are the closest analogs but a different concept
  (level-completion counters, not a continuous score). Per this plan's
  Task 14 guidance: default our internal `GameFrame.score` to `0` rather than
  remapping `levels_completed` onto it — `score_delta` in
  `zerx/transitions.py` then reads as always `0`, which is honest, not
  broken.
- `guid`, `full_reset`, `game_id`, `action_input` exist but are unused by the
  adapter.

`GameAction` is an `Enum` (`RESET=0, ACTION1..5, ACTION6=6 (complex), ACTION7=7`):
- `.is_complex()` / `.is_simple()` — `ACTION6` is the only complex action.
- `.set_data({"x": int, "y": int})` — **mutates the shared enum member's**
  `.action_data` in place and returns it; the real starter agent calls this
  directly on the `GameAction.ACTION6` singleton and returns the same
  (mutated) member as the action — see `agent/my_agent.py`'s original body
  (`git show HEAD~1:agent/my_agent.py` after this commit). We replicate that
  exact pattern in Task 14 rather than fighting it.
- `.from_id(action_id)` / `.from_name(name)` — enum lookup helpers. **Use
  `.from_id()`, not `GameAction(action_id)`** — the latter raises
  `ValueError: 1 is not a valid GameAction` for every non-RESET id.
  Verified directly against the vendored `arcengine` package (Task 14):
  `GameAction`'s custom `__init__` reassigns `self._value_` to the plain
  int action id, but Python's Enum machinery already built
  `_value2member_map_` from the original `(action_id, action_type)` tuple
  argument *before* `__init__` ran, so the map is keyed by the stale tuple
  and value-based lookup (`GameAction(1)`) never matches, even though
  `GameAction.ACTION1.value == 1` reads correctly afterward. `.from_id()`
  sidesteps this because it linear-scans `.value` instead of consulting
  that stale map. Bracket lookup by name (`GameAction["ACTION1"]`) is
  unaffected — it uses `_member_map_`, not the value map.
- Coordinates for `ACTION6` are validated by a pydantic `ComplexAction` model
  to `x, y` in `[0, 63]` inclusive — matches our independent
  `zerx/types.py` `Action` validation.

### Real starter default behavior (both corrected vs. the plan's illustrative placeholders)

- `is_done`: `return latest_frame.state is GameState.WIN` — the framework's
  `Agent.main()` loop stops here. **Does not** stop on `GAME_OVER`; the
  starter's own comment: "Stop once we win. Don't stop on GAME_OVER — we
  want to RESET and retry." Task 14's placeholder (`state == "GAME_OVER"`)
  would have been wrong on two counts: string comparison against an enum,
  and the wrong terminal condition entirely.
- `choose_action`'s terminal guard: `if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER): return GameAction.RESET`.
  Our internal `GameFrame.is_game_over` must be set `True` for **both**
  `NOT_PLAYED` and `GAME_OVER` (not `GAME_OVER` alone) so `zerx.policy.decide()`'s
  existing `if frame.is_game_over: return RESET` short-circuit fires
  correctly on a game's very first call too.

## Local dev network behavior (not a private-data concern)

`scripts/play_local.py` constructs `arc_agi.Arcade(operation_mode=OperationMode.NORMAL)`,
which fetches public environment/game metadata from `three.arcprize.org` on
first use per game (auto-generating an anonymous API key) and caches game
source under `environment_files/` (gitignored). This is the documented
public dev flow, not hidden game/engine internals or private evaluation
data.

## Windows-native environment deviations (recorded per this plan's own
"stop and say so" instruction — resolved with the human owner before
proceeding)

1. **`make` is unusable via native Windows venv paths.** The vendored
   `Makefile` hardcodes Unix venv layout (`.venv/bin/python`, `.venv/bin/pip`).
   Native Windows `python -m venv` produces `.venv/Scripts/` instead — there
   is no `bin/` directory. Resolution (human-approved): run the underlying
   commands directly instead of through `make`, e.g.:
   ```bash
   python3.12 -m venv .venv
   .venv/Scripts/python.exe -m pip install "arc-agi>=0.9.6" "kaggle>=2.2" python-dotenv pandas pyarrow
   git clone --depth 1 https://github.com/arcprize/ARC-AGI-3-Agents.git vendor/ARC-AGI-3-Agents
   .venv/Scripts/python.exe scripts/slim_framework.py
   .venv/Scripts/python.exe scripts/play_local.py --max-steps 200
   ```
   The vendored `Makefile` itself is **not modified** — it stays byte-for-byte
   what a Mac/Linux teammate would use, per `AGENTS.md`'s team contract.
2. **`scripts/slim_framework.py` had a real, Windows-only bug**: its
   `INIT.write_text(SLIM)` call (no `encoding="utf-8"`) writes the file using
   the platform's default codepage. `SLIM`'s docstring contains an em dash
   (U+2014); on this machine's codepage that byte is not valid UTF-8, so the
   next import of the slimmed `agents/__init__.py` raised
   `SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0x97`.
   Fixed locally (in our copy under `scripts/`) by adding
   `encoding="utf-8"` to that one `write_text()` call — a one-line
   correctness fix to a genuine bug, not a behavior change.
3. **`scripts/play_local.py`'s per-game summary `print()` uses a Unicode
   arrow (`→`, U+2192)**, which crashed with `UnicodeEncodeError` under this
   machine's Turkish (`cp1254`) console codepage after the very first game.
   Resolved without touching the script: ran it with
   `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` set in the environment.

None of the above are competition-integrity or scope issues — they are
Windows-console/venv-layout friction in local dev tooling only. Nothing
about the real upstream API, game logic, or Kaggle submission path was
changed.

## `make play-local` result (all 25 games, `--max-steps 200`, unmodified random-baseline `agent/my_agent.py`)

Every game ran to `MAX_ACTIONS` (80, the starter's built-in cap, below the
200-step CLI cap) without a Python exception, ending in `NOT_FINISHED` with
`levels_completed=0` for all 25 games:

```
ls20  tu93  cn04  m0r0  wa30  lp85  vc33  sc25  tn36  tr87
re86  lf52  ft09  r11l  cd82  dc22  g50t  sk48  ka59  sp80
s5i5  ar25  sb26  bp35  su15
```

Aggregate scorecard score: **0.0**.

## Conclusion

Baseline recorded, matches README's documented `0.0` score for the random
agent. No `zerx` code involved yet. Real API facts above (frame shape,
`GameState` enum semantics, `available_actions` id list, absence of a raw
`score` field, `set_data`'s mutate-and-return pattern) are what Task 14's
adapter must use verbatim instead of the plan's illustrative placeholders.
