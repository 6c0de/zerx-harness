# AGENTS.md — ARC-AGI-3 Zerx baseline

Applies to any coding agent working in this repository (Codex, Claude Code,
etc.) — see also `CLAUDE.md` at the repo root for tool-specific notes.

## Mission

Build a Kaggle-valid ARC Prize 2026 ARC-AGI-3 agent using Gemma-4-31B as a local vision-language policy. Optimize measured RHAE score: both level completion and action efficiency matter. Reuse proven Reki ideas without copying implementation code, keep every behavior ablatable, and prefer the simplest profile that wins repeated comparisons.

The initial score target is approximately 1% — roughly the level needed to
exceed current frontier closed-model scores (Gemini 3.1 Pro 0.37%, Claude
0.25%, Grok 0%, as of mid-2026). Treat it as a project goal, not as evidence
that any specific architecture will achieve it.

## Source of truth and repository discovery

Before editing, inspect the actual clone, its current branch, `git status`, README files, build metadata, and tests. Do not assume that a design-document path or command exists merely because it was planned.

The upstream starter is `github.com/arcprize/ARC-AGI-3-Kaggle-Starter` — a
local dev kit where the only file you normally edit is `agent/my_agent.py`,
defining `class MyAgent(Agent)` with `is_done(frames, latest_frame)` and
`choose_action(frames, latest_frame) -> GameAction`. `make play-local` runs
your agent against the real game engine locally; `make submit` builds and
pushes the Kaggle notebook (5 official submissions/day). Verify this against
the actual clone before relying on it — starter kits evolve.

The intended project layout is:

```text
repository/
├── agent/
│   └── my_agent.py          # thin MyAgent(Agent) harness adapter
├── zerx/
│   ├── perception.py        # frame -> labeled image / compact grid view
│   ├── policy.py            # prompt, one-action JSON, repair, legal guard
│   ├── memory.py            # reflection state and refresh
│   ├── heuristics.py        # click candidates and dead signatures
│   ├── budget.py            # action-efficiency state and policy hints
│   ├── model_backend.py      # only Gemma load/inference boundary
│   └── config.py            # typed, serializable ablation configuration
├── eval/
│   └── run_ablation.py
├── tests/
└── notebooks/               # generated artifacts; do not hand-edit by default
```

If the checked-out official starter instead uses `agents/templates/my_agent.py`, `main.py`, or another structure, first identify its documented extension and packaging points. Adapt the proposed `zerx/` package to the real harness; do not create a parallel entrypoint that Kaggle never imports. Record the exact upstream repository and commit used.

## Environment split

- **Local computer (Codex / Claude Code):** source control, code editing, static checks, unit tests, mocked model tests, packaging inspection, and experiment records. Never load Gemma-4-31B on the local RTX 4060.
- **Colab Pro A100 or L4:** model-loading checks and full model-in-the-loop development against legally available local/public games. A100 is preferred. L4 requires a separately verified quantization/configuration.
- **Kaggle RTX Pro 6000 (48GB, `g4-standard-48`, ARC-AGI-3-exclusive):** final environment-compatibility runs, offline packaging, official submissions, and leaderboard evaluation. Do not use Kaggle GPU quota for early unit-level iteration. Other Kaggle accelerator options (`cpu`, `t4`, `p100`) exist but don't fit a 31B model.

Colab results are provisional. Kaggle is the deployment source of truth because CUDA, vLLM, memory, attached inputs, runtime limits, and offline behavior can differ. Record the actual GPU, GPU memory, precision/quantization, model revision, package versions, and prompt/config hash for every model run.

## Scope

Baseline features:

- Gemma-4-31B only;
- labeled-frame and/or compact grid perception;
- exactly one proposed environment action per policy call;
- strict JSON parsing, at most one bounded repair attempt, legal-action validation, and a safe fallback chain;
- reflection memory refreshed at a configurable interval;
- NumPy click-candidate heuristic and dead-signature tracking;
- action-budget telemetry and a configurable shift from probing to execution;
- flags/configuration that allow each feature to be disabled independently.

Out of baseline scope:

- Forge-style arbiter or multiple candidate generators as default behavior;
- multi-model abstraction beyond Colab/Kaggle hosting of the same model;
- code-writing or REPL agents inspired by The Duck;
- multi-action queues;
- training or fine-tuning Gemma;
- milestone-prize publication logistics.

An arbiter may be added later only as an off-by-default experiment with a stated hypothesis and baseline comparison.

## Required control flow

`agent/my_agent.py` must stay thin. It owns harness integration and delegates to importable, testable `zerx` modules.

For each `choose_action(frames, latest_frame)` call:

1. Detect terminal/game-over state first. Return `RESET` when that is the only legal action.
2. Read the current frame's available-action metadata; do not assume all games support the same actions.
3. Produce perception data from the latest stable frame and a bounded trailing history.
4. Generate heuristic candidates and filter dead signatures.
5. If the configured heuristic-first policy meets a calibrated threshold, propose its action; otherwise call Gemma once.
6. Parse one structured action. Permit at most one bounded repair attempt. Validate action name, required data, and `ACTION6` coordinates in `[0, 63]`.
7. Apply the action-budget policy as a strategy signal. Never invent a "safe" move that has not passed the same legality checks.
8. Return one legal action and record the decision path, latency, fallback/repair status, memory version, and configuration ID.

`choose_action` must not leak an unexpected exception through the harness. Catch errors at module boundaries, log concise diagnostics, and use this fallback order:

`validated model action -> validated heuristic action -> validated deterministic legal fallback -> RESET if terminal`

A random fallback is last resort only and must be sampled from the current frame's legal actions. Model initialization failures and out-of-memory conditions must fail before gameplay rather than degrading an entire evaluation silently.

## Design corrections and cautions

- Reasoning/model calls do not directly consume game actions, but they consume wall-clock time. Do not describe a reflection model call as consuming "action budget"; measure latency separately.
- A human-median action count may be unavailable to the agent. `budget.py` must not assume hidden evaluation data. Use observable limits or a configurable proxy and label it accordingly.
- A high heuristic confidence score is not automatically calibrated. Keep `HEURISTIC_FIRST` off until offline/model-in-loop comparisons demonstrate value.
- Self-repair must not become an unbounded second reasoning loop. Prefer deterministic extraction/repair; if model-based repair is tested, count its latency and failure rate separately.
- Do not hard-code `ACTION1`–`ACTION5` semantics. Their meanings vary by game.
- Never return `ACTION7` in competition unless the current competition interface explicitly exposes it as legal.

## Configuration and reproducibility

Use a typed configuration object. Environment variables may populate it at startup, but feature code must not read environment variables directly. Serialize the resolved configuration with every run.

At minimum record:

- experiment ID, date, owner, base commit, and upstream commit;
- model revision, precision/quantization, backend settings, GPU type, and package/runtime versions;
- prompt/perception/config hashes;
- public games and seeds used;
- levels completed, per-level/game action counts, calculated RHAE where valid, wall time, inference latency, invalid outputs, repairs, fallbacks, resets, exceptions, and OOM/timeouts;
- baseline delta and conclusion: keep, revert, or investigate.

Do not call a feature an improvement from one run. Use repeated seeds/configurations and inspect per-game regressions. Never optimize using private evaluation information or modify competition fixtures.

## Testing gates

Local, model-free tests must cover:

- frame/history conversion and coordinate convention;
- JSON extraction, schema validation, deterministic repair, and rejection;
- every legal-action guard, including terminal-only `RESET` and `ACTION6` bounds;
- fallback ordering and the guarantee that `choose_action` does not raise;
- memory refresh/reset/isolation between games;
- dead-signature and click-candidate scoring;
- action-budget/proxy calculations and boundary cases;
- configuration serialization and feature disabling.

Use the repository's locked dependency and documented commands. If it uses `uv`, prefer:

```bash
uv sync --frozen
uv run pytest -q
```

Before a Colab run, the local suite must pass. Before a Kaggle run, a Colab model-load smoke test and at least one model-in-loop test must pass with the candidate model configuration.

## Colab gate

The Colab notebook must install pinned versions, clone or check out an exact commit, print the resolved environment without secrets, load the exact Gemma revision, and save structured results outside ephemeral runtime storage.

Never treat Drive as source control. Commit code locally; use Drive only for non-secret artifacts and experiment results. Do not place Kaggle tokens, model-provider secrets, or private competition material in notebook cells or outputs.

For any candidate promoted from Colab, document known differences from Kaggle, especially GPU memory, dtype/quantization, vLLM flags, model path, and internet availability.

## Kaggle gate

Kaggle execution must work with internet disabled. Model weights and dependency artifacts must be attached as permitted Kaggle inputs, read from `/kaggle/input`, and never downloaded at evaluation time. Write outputs only under `/kaggle/working`.

Before spending remote compute or a submission slot, show the user:

- commit and experiment ID;
- upload/build contents;
- notebook/kernel slug and accelerator;
- attached competition, model, datasets, and dependency sources;
- resolved model path and offline status;
- expected output.

Never push a notebook, start a paid/quota-consuming run, or submit to the competition without explicit user approval. Use the official starter's verified build/submit flow when present; otherwise use the official Kaggle CLI. Do not assume planned commands such as `make play-local` or `make submit` until they are confirmed in the checked-out repository.

Validate `submission.parquet` by byte size, successful readback, schema, rows, nulls, and embedded/source identity. A file's existence alone does not prove success. Submission is a separate explicit approval after validation.

## Definition of done

A change is complete when its hypothesis is stated, focused and full tests pass, the resolved config is recorded, no unrelated edits or secrets are included, and remaining Colab/Kaggle verification is explicit.

A Colab experiment is complete when results and environment metadata are persisted and compared with the unchanged baseline.

A Kaggle experiment is complete only when the kernel reaches a terminal status, logs and outputs are downloaded and checked, the artifact is structurally valid, and the experiment record is updated. A submission is complete when Kaggle reports its terminal evaluation result and that result is recorded.
