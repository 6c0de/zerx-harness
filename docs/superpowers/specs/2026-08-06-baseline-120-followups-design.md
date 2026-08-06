# `baseline-120` follow-ons — design

Date: 2026-08-06
Status: approved for planning (see approval note at end)

## Purpose

Four items were recorded as recommendations in `docs/HANDOFF.md`'s "Exact
next action" after the `baseline-120` integration (see commit
`b405d3b`), all explicitly *not started* at the time:

(a) a general `README.md`;
(b) a pygame-based live replay visualizer (reference:
[github.com/Darkosxl/Agent_Harness_Example](https://github.com/Darkosxl/Agent_Harness_Example));
(c) a personal `ARC_API_KEY` so local/Colab runs attribute to the human
owner's account on `three.arcprize.org` instead of an anonymous one;
(d) a JSON-like export of played games (per-step structured data
including each decision's reasoning/raw model output), explicitly noted
as the natural data source for the visualizer's replay buffer — "worth
designing together rather than as two unrelated features."

This spec covers all four, split into two sub-projects:

- **Sub-project 1 (small, bundled):** README.md + documenting
  `ARC_API_KEY`.
- **Sub-project 2 (the real design work):** one shared trace format,
  consumed either live (feeding a pygame window during an actual run) or
  from a saved file (offline replay).

A fifth, related capability was raised during design review — resuming
or forking a previously-recorded game from a specific step, under
different code/config/backend/environment, for fine-tuning/debugging.
Per explicit decision during brainstorming: **design for it now (so the
trace format doesn't need a breaking change later), build it later.** See
"Future work: resume/fork from a recorded step" below.

## Sub-project 2: trace format + visualizer

### Investigation findings that shaped this design

- The vendored `agents.agent.Agent.main()` (base class of `MyAgent`) is a
  simple synchronous loop: `while not is_done(...) and action_counter <=
  MAX_ACTIONS: action = self.choose_action(...)`. `choose_action()` is
  called once per step, in-process, on the same thread as the loop. This
  means live visualization needs no loop-reimplementation and no
  cross-process streaming — a recorder attached inside `choose_action()`
  can render synchronously and even block (for pause) without disturbing
  the real game loop's control flow.
- `zerx/policy.py`'s `decide()` is the only place that sees a model
  backend's raw response before parsing, and per `STRATEGY.md` §6 it must
  stay a pure, side-effect-free function. Resolution (confirmed during
  design review): widen `Decision` with a new optional field
  `raw_response: Optional[str] = None`, populated by `decide()` and
  returned normally — a richer return value, not a side-effecting
  callback. Additive-only; no existing `Decision(...)` construction site
  breaks.
- `arc_agi`'s `Arcade` client already resolves `ARC_API_KEY` via
  `constructor arg > ARC_API_KEY env var > anonymous-key fallback`
  internally (`.venv/Lib/site-packages/arc_agi/base.py`). Every call site
  in this repo (`scripts/play_local.py`, `eval/run_ablation.py`,
  `scripts/build_colab_notebook.py`, `tests/test_real_game_regression.py`)
  constructs `Arcade(operation_mode=OperationMode.NORMAL)` with no
  `arc_api_key` override. **No code change is needed for item (c)** —
  setting the env var in your own shell already works. This is a
  documentation item only, folded into the README.
- `arc_agi`'s `Arcade.make()` accepts a `seed: int = 0` parameter, but no
  snapshot/checkpoint/save-state API exists anywhere in the client. This
  confirms "resume from step N" can only work by deterministically
  replaying the recorded action prefix through the real, public step API
  — there is no shortcut engine-state restore to hook into instead.

### Data model (`zerx/trace.py`, new module)

```python
@dataclass(frozen=True)
class TraceMeta:
    game_id: str
    seed: int
    backend: str
    config_hash: str
    started_at: str  # ISO 8601

@dataclass(frozen=True)
class TraceStep:
    step_index: int
    game_id: str
    grid: tuple[tuple[int, ...], ...]  # the frame the decision was made against
    action_name: str
    action_x: Optional[int]
    action_y: Optional[int]
    source: str
    repaired: bool
    target_object_label: Optional[str]
    reasoning: str  # raw_response when source == "model"; else a
                     # synthesized one-line description of why the
                     # fallback/heuristic path fired
    levels_completed: int
    game_state: str
```

Each `TraceStep` is self-contained (the frame it was decided *from*, not
its outcome) — no deferred "attach outcome later" bookkeeping is needed,
because consecutive steps naturally show before/after: step N+1's `grid`
*is* the observed outcome of step N's action. This mirrors
`zerx/transitions.py`'s existing "attach the previous action's result
once the next frame exists" discipline without duplicating it.

`TraceRecorder` protocol: one method, `record(step: TraceStep) -> None`.
Two implementations ship in `zerx/trace.py` (no pygame dependency, fully
unit-testable):

- `JsonlTraceWriter(path)` — appends one JSON line per step.
- `CompositeTraceRecorder([...])` — fans out to multiple recorders (used
  by live mode to render *and* save simultaneously).

### File format

One file per recorded game, JSONL, gitignored under a new `traces/`
directory (same treatment as `notebooks/*.ipynb` — generated, not source
of truth). Default name `traces/<game_id>-<timestamp>.jsonl`.

- Line 1: `{"type": "meta", ...TraceMeta fields...}`
- Lines 2+: `{"type": "step", ...TraceStep fields...}`

The `"type"` discriminator is a cheap, forward-compatible choice — it
lets a future record type (e.g. a fork/resume lineage marker) be added
without a breaking format change.

### Capture point & wiring

- `zerx/config.py`: new field `trace_export_path: Optional[str] = None`
  (env var `ZERX_TRACE_EXPORT_PATH`), appended at the end of the field
  list, off by default — matches this project's existing ablatable-flag
  convention exactly.
- `agent/my_agent.py`: new **public** attribute `self.trace_recorder:
  Optional[TraceRecorder]`, built from config in `__init__` (a
  `JsonlTraceWriter` if `trace_export_path` is set, else `None`). Public
  (not underscore-prefixed) specifically so an external script can
  reassign it *after* construction, *before* calling `agent.main()` — the
  same "reach into agent internals from an external script" pattern
  `scripts/play_local.py` already uses for `MAX_ACTIONS` (see
  `docs/HANDOFF.md`'s known-failures item 7), done here as an intended
  public seam instead of an incidental one. `choose_action()` calls
  `self.trace_recorder.record(...)` once per step, after computing the
  decision, only when a recorder is attached — zero overhead and zero
  behavior change when it isn't.
- `zerx/policy.py`: `Decision.raw_response` populated by `decide()`
  whenever a model call happens, **including on a failed parse** — this
  is exactly what makes the visualizer useful for diagnosing the
  `build_prompt()` legal-actions gap `docs/HANDOFF.md` already documents.

### Live visualizer (`scripts/visualize_play.py`, new)

Pure dev tooling — not part of `zerx/`, never bundled into the Kaggle
submission notebook (`scripts/build_notebook.py` only bundles
`zerx/*.py`), free to depend on `pygame`.

Two modes sharing one render/navigate code path:

- `--live --game <id> [--max-steps N] [--save path] [--history-cap 500]`:
  constructs the real engine + `MyAgent` exactly like
  `scripts/play_local.py`, then sets `agent.trace_recorder =
  LivePygameRecorder(cap)` (or a `CompositeTraceRecorder` wrapping that
  plus a `JsonlTraceWriter` if `--save` is given) before calling
  `agent.main()`. `LivePygameRecorder` lives in this script (pygame-only
  code stays out of `zerx/`). SPACE pauses by blocking *inside*
  `record()` — which runs on the game loop's own thread — pumping pygame
  events until un-paused, so pause genuinely halts execution rather than
  just freezing the display. ←/→ step back/forward through the capped
  in-memory buffer (default 500 steps, matching the reference repo).
  ↑/↓ scroll the reasoning-text panel.
- `--replay <trace.jsonl>`: loads a saved file, no game engine touched at
  all, same render/navigate code, always in a "paused" state (no running
  loop to pause).

Rendering: the frame's color-index grid drawn with a fixed small ARC
palette (0–9), plus a side panel showing the current step's action,
source, and reasoning text (scrollable).

### Testing strategy

- `zerx/trace.py`: full unit coverage — `TraceStep`/`TraceMeta`
  construction and JSON round-trip, `JsonlTraceWriter` appends valid
  lines to an existing file, `CompositeTraceRecorder` fans out to every
  child recorder. No pygame or display dependency (matches `AGENTS.md`'s
  "local, model-free tests" gate).
- `zerx/policy.py`: extend existing `decide()` tests to assert
  `raw_response` is populated on the model-call path and `None` when no
  model call was attempted.
- `agent/my_agent.py`: extend `tests/test_my_agent.py` — a fake in-memory
  recorder confirms `record()` fires exactly once per `choose_action()`
  call when attached, and confirm it's `None`/unused when
  `trace_export_path` is unset (off-by-default, like every other
  feature in this project).
- `scripts/visualize_play.py`: the pygame render/event loop itself isn't
  meaningfully unit-testable (same category as
  `scripts/build_colab_notebook.py`, which tests generation, not
  execution) — logic is kept in small pure helpers (grid→color mapping,
  `--replay` file parsing, buffer navigation index math) that *are*
  tested directly. Actual on-screen rendering is verified manually
  against a real local game once implemented, reported as manual
  verification, not automated coverage.
- `pygame` is added to `requirements-zerx.txt`. It is never imported
  under `zerx/`, so `scripts/build_notebook.py`'s Kaggle bundle and
  `zerx/secret_scan.py`'s scope are both unaffected.

## Sub-project 1: README.md + `ARC_API_KEY` documentation

`README.md` (new, repo root): project overview (one paragraph, pointing
to `AGENTS.md`/`STRATEGY.md` for the authoritative contract rather than
duplicating it); setup (`uv sync --frozen` or
`pip install -r requirements-zerx.txt`); running locally
(`scripts/play_local.py` usage, `ZERX_*` env var overview pointing at
`zerx/config.py` as source of truth); setting `ARC_API_KEY` for account
attribution (documentation only, no code — see investigation findings
above); running tests (`pytest tests/ -q`, and the
`-m "not slow_local_engine"` fast-iteration filter Track 3's own plan
documented); the new visualizer's `--live`/`--replay` usage.

## Future work: resume/fork from a recorded step (documented now, not built)

Mechanism: given a saved trace and a step index N, deterministically
replay the exact recorded actions for steps `[0, N)` through the real
engine via the normal public step API (`arc_env.step(action)` — nothing
hidden, nothing cloned), verify the reached frame matches what was
recorded (a sanity check — these public games are deterministic per game,
confirmed by this project's own earlier finding, not randomized per
run), then hand off live control to whatever code/config/backend/platform
is specified *this time*, continuing from step N onward.

**Why this stays within `AGENTS.md`'s hard rules:** it never reads hidden
game/engine source, runtime fields, or internal state, and it never
clones or reconstructs the real environment's internal state for search
— it only re-drives the same public action API with actions that were
legitimately taken before. This is categorically different from
`STRATEGY.md`'s excluded ProjectForty2-FORGE-style state cloning, which
inspects/reconstructs *hidden* state. It is dev-time tooling, same
environment-split category as `scripts/play_local.py` or a Colab
notebook — never used during an actual scored Kaggle run, and every
action taken during a resume is a real action against the real
engine (local or Colab), not a counterfactual probe against a cloned
copy.

**Why the trace format already supports this without a breaking change:**
`TraceStep` already records the exact `action_name`/`action_x`/`action_y`
for every step in order, and `TraceMeta` already records `game_id`/`seed`
— everything the replay-prefix mechanism needs. The mechanism itself
(replay-then-diverge orchestration) belongs in its own future script
(e.g. `scripts/resume_play.py`, or a `--resume-from trace.jsonl:N` flag
on `scripts/visualize_play.py`), not inside `zerx/trace.py`, which stays
a pure data-capture library.

Not implemented this round. To be written up as its own
`docs/HANDOFF.md` "Exact next action" item once this spec's
implementation lands.

## Branch

One new branch, `feat/baseline-120-followups`, off `master` at `b405d3b`.

## Approval note

Design discussed and approved interactively with the human owner during
brainstorming (2026-08-06): visualizer v1 scope confirmed as live +
replay together (not replay-only first); `Decision.raw_response`
(widening the dataclass) confirmed over a `decide()` callback parameter;
resume/fork confirmed as same-game/different-environment-code (not a
literal cross-game engine swap), and confirmed as design-now/build-later
rather than in this round's implementation.
