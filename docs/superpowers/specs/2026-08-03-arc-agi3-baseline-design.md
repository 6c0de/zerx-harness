# ARC-AGI-3 Baseline Agent — Design

## Goal

Build a Kaggle submission for the ARC Prize 2026 — ARC-AGI-3 competition that
plays the interactive game environments using a locally-hosted **Gemma-4-31B**
model as the policy. Target: reach a leaderboard score around **~1.0%**
(RHAE-based), which would exceed current frontier closed-model scores
(Gemini 3.1 Pro 0.37%, Claude 0.25%, Grok 0%, as of mid-2026).

Primary success criterion is **score**, not architectural purity. The design
favors reusing ideas already proven by top open community solutions (Reki,
Forge — June 2026 milestone) over inventing from scratch, while keeping the
code ours and fully ablation-instrumented so we know which piece is earning
points.

## Background / Constraints

- **Scoring (RHAE):** per-level score = `(human_median_actions /
  agent_actions)^2`, capped at 1.15. Inefficient solves are penalized
  quadratically — "eventually correct but slow" scores near zero. Action
  efficiency matters as much as correctness.
- **Kaggle sandbox:** no internet access during evaluation. No closed APIs
  (GPT/Claude/Gemini) allowed — the submitted model must be local, open-weight
  (Gemma-4-31B, per this project's target).
- **Official starter kit** (`arcprize/ARC-AGI-3-Kaggle-Starter`): the only file
  a submission needs to modify is `agent/my_agent.py`, defining
  `class MyAgent(Agent)` with two methods:
  - `is_done(frames, latest_frame) -> bool`
  - `choose_action(frames, latest_frame) -> GameAction`
  Local iteration happens via `make play-local` against the real game engine
  (same engine Kaggle runs); `make submit` builds and pushes the notebook.
  5 official submissions/day.
- **Action space:** `RESET`, `ACTION1`-`ACTION4` (directional), `ACTION5`
  (interact/select/rotate), `ACTION6` (click, `data={"x","y"}`, grid is
  0-63), `ACTION7` (undo — not available in competition). On game-over, only
  `RESET` is legal; any other action returns HTTP 400.
- **Hardware:** Kaggle accelerator is chosen per-notebook (`cpu`, `t4`,
  `p100`, `rtx6000`). `rtx6000` (48GB, g4-standard-48) is ARC-AGI-3-exclusive
  and is what a 31B model needs — must not be used for early iteration
  (burns quota). Gemma-4-31B needs 24GB+ VRAM; won't fit on the user's local
  RTX 4060.
- **Dev environment split:** Colab Pro (A100/L4) is the real dev+eval loop
  for anything that loads the model. Local RTX 4060 is only for
  model-free unit tests (perception/heuristics/budget logic).
- **Prior art referenced (ideas only, not copied code):** Reki
  (vision-LLM-as-policy, labeled-frame rendering, JSON single-action output,
  periodic reflection memory, numpy click heuristic with "dead signature"
  filtering to stop re-clicking useless object types) and Forge (same core
  idea wrapped in a generator/arbiter framework — notably, their
  *best-scoring* profile had the arbiter and other extra machinery turned
  off). Both built on the official GPT-OSS-120B template, swapped to
  Gemma-4-31B.

## Non-goals

- Not building the arbiter/multi-candidate-generator layer from Forge as a
  default-on feature — Forge's own ablation showed it hurt their best run.
  It may exist as an off-by-default toggle if time permits later, but it is
  out of scope for the baseline.
- Not targeting the $700K grand prize (100% RHAE) or any milestone prize
  submission logistics (open-sourcing under CC0/MIT-0, etc.) in this design.
  Purely a personal score target.
- Not supporting any model other than Gemma-4-31B. No multi-model
  abstraction beyond what's needed to swap dev (Colab) vs. eval (Kaggle)
  hosting of the same model.
- The Duck's code-writing/REPL approach is not part of this baseline. Noted
  as a possible future direction, not designed here.

## Architecture

Extends the official starter kit rather than replacing it — `agent/my_agent.py`
stays the thin entrypoint the harness calls; all real logic lives in an
importable package so it can be exercised identically from `make play-local`,
a Colab notebook, and the Kaggle submission notebook.

```
ARC-AGI-3-Kaggle-Starter/       (official starter, cloned as-is)
├── agent/
│   └── my_agent.py             # MyAgent(Agent): is_done(), choose_action() — orchestration only
├── zerx/                       # our package
│   ├── perception.py           # frame -> labeled image + ASCII grid representation
│   ├── policy.py                # prompt build, Gemma-4-31B call, JSON schema + self-repair, legal-action guard
│   ├── memory.py                 # reflection memory, refreshed every N steps
│   ├── heuristics.py             # numpy click-candidate scan, dead-signature tracking (no GPU)
│   ├── budget.py                  # RHAE-aware action budget / safe-mode trigger
│   ├── model_backend.py          # loads Gemma-4-31B (Colab vs Kaggle rtx6000), single load point
│   └── config.py                 # env-var flags gating every module above, for ablation
├── eval/
│   └── run_ablation.py         # sweeps flag combinations over `make play-local` games, logs per-game RHAE
└── notebooks/                  # untouched, auto-generated by starter's build step
```

Each module is independently testable and independently disable-able via
`config.py` env vars — this is the ablation discipline Reki used, and it's
required to know which piece is actually contributing to score.

## Data flow (per `choose_action` call)

1. Starter harness calls `choose_action(frames, latest_frame)`.
2. `perception.py` renders `latest_frame` (plus a short trailing history) into
   a labeled image and/or ASCII grid.
3. `heuristics.py` runs first (cheap, no GPU): dead-signature filter removes
   previously-useless object types; a click-candidate scan looks for
   small/rare-colored/button-like shapes. If `HEURISTIC_FIRST=1` and
   confidence is high, this can return an action directly, skipping the LLM
   call entirely (saves both wall-clock and action-count risk).
4. Otherwise `policy.py` builds a prompt from the perception output plus
   `memory.py`'s current reflection summary, calls Gemma-4-31B through
   `model_backend.py`, and parses a single JSON action. Malformed JSON
   triggers one self-repair retry; if that also fails, fall back to the
   heuristic layer's suggestion (never a raw random action if avoidable).
5. `budget.py` checks cumulative actions for the level against the
   RHAE-implied budget (roughly 5x human median, tightened as it's approached)
   and can force a conservative/safe move rather than an exploratory one when
   close to the cap.
6. Every N steps, `memory.py` refreshes its reflection summary (cheap
   rule-based summary, or a short model call — TBD during implementation
   based on what the ablation shows is worth the action-budget/latency cost).
7. `policy.py`'s legal-action guard validates the chosen action against
   current game state (in particular: force `RESET` if the game is in a
   game-over state) before returning it to the harness.

## Error handling

- `choose_action` must never raise — the starter harness has no surrounding
  try/except. Any unexpected failure in perception/policy/model backend
  falls back to the heuristic layer, and if that also fails, a legal random
  action (never an illegal one — avoids the 400 on game-over misfires).
- Game-over state is special-cased explicitly in the legal-action guard, not
  left to the model to infer from the prompt.
- Model backend load failures (e.g., OOM on a misconfigured accelerator) fail
  fast at notebook startup, not mid-game — the whole point of testing on
  Colab first is to catch this before spending a daily submission.

## Testing / ablation

- Unit tests (run on local RTX 4060, no model load): perception frame→grid
  correctness, policy JSON-schema validation and self-repair logic, budget
  cap math, heuristic click-candidate scoring.
- `eval/run_ablation.py` sweeps `config.py` flags (`HEURISTIC_FIRST`,
  `MEMORY_ON`, `ARBITER_ON` off-by-default, etc.) across every local game via
  `make play-local`, logging per-game RHAE per configuration. This is how we
  decide what's worth keeping before spending one of the 5 daily Kaggle
  submissions.
- Full model-in-the-loop runs happen on Colab Pro against the local game set
  before any `make submit`.

## Open questions to resolve during implementation

- Exact quantization for Gemma-4-31B on Kaggle's `rtx6000` (48GB) vs. Colab's
  A100/L4 — pick the highest precision that fits comfortably, confirm
  identical behavior across both before relying on Colab numbers as a
  predictor of Kaggle score.
- Whether reflection-memory refresh should be a cheap heuristic summary or a
  short additional model call — decide via ablation, not upfront.
- How Gemma-4-31B weights get attached to the Kaggle notebook given no
  internet at eval time (Kaggle Dataset attachment) — needs a concrete
  walkthrough during the environment-setup step.
