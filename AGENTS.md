# AGENTS.md — ARC-AGI-3 Zerx baseline

Applies to any coding agent working in this repository (Codex, Claude Code,
etc.) — see also `CLAUDE.md` at the repo root for tool-specific notes.

## Mission

Build a Kaggle-valid ARC Prize 2026 ARC-AGI-3 agent using Gemma-4-31B as a local vision-language policy. Optimize measured RHAE score: both level completion and action efficiency matter. Reuse proven Reki ideas without copying implementation code, keep every behavior ablatable, and prefer the simplest profile that wins repeated comparisons.

The initial score target is approximately 1% — as of early August 2026 that
would exceed frontier closed-model scores on the public leaderboard (Gemini
3.1 Pro 0.37%, Claude 0.25%, Grok 0%). Leaderboard numbers move; verify the
live board before treating these as a fixed bar. Treat 1% as a project goal,
not as evidence that any specific architecture will achieve it.

**Delivery window: this is a 5-day build, currently Day 1, due Day 5.** A
serious Kaggle run can occupy most of the notebook's ~9-hour allowance plus
unpredictable queue/scoring time — see the Day-by-day schedule in the Kaggle
gate section. There is no slack for a Day-5-morning first attempt; the
intended final submission must start no later than Day 4.

## Cross-agent team contract

This file is the authoritative repository instruction set for Codex, Claude
Code, and any other coding agent working on this project. `CLAUDE.md` and
any Codex-specific file must point here rather than duplicate these rules.

Repository state — not an AI chat transcript — is project memory. Before
working, read this `AGENTS.md`, `STRATEGY.md`, `docs/TEAM_WORKFLOW.md`,
`docs/HANDOFF.md`, the active experiment record, and the actual repository
README/build instructions. Do not rely on an earlier Codex or Claude
conversation; if a chat conclusion matters, record it in Git as a decision,
experiment result, or handoff note.

`STRATEGY.md` records prior-art analysis (ReKi, Murad/Forge VLM,
ProjectForty2 FORGE, Tycho), adoption decisions, trade-offs, and experiment
sequencing. It does not override the operating, safety, ownership,
evaluation-integrity, or deployment rules in this file.

One person owns a branch/experiment at a time. Do not have two agents (or
two people) edit the same working tree concurrently — exchange work through
reviewed commits and branches, never copied chat patches. Before switching
PCs or AI products, the current owner stops at a coherent checkpoint, runs
and records relevant tests, commits and pushes, updates `docs/HANDOFF.md`
(template: `docs/HANDOFF.md`), and states the exact next action. The next
owner pulls the named commit, recreates the environment, and runs the
smoke suite before continuing.

Each teammate uses their own Codex/Claude, Colab, Git, and Kaggle
credentials. Never share login sessions, cookies, tokens, or credentials
through Git, chat, notebooks, Drive, or handoff files.

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
│   ├── model_backend.py      # ModelBackend protocol, fake + Gemma backends
│   ├── secret_scan.py        # scans generated artifacts for leaked credentials
│   ├── backends/
│   │   └── cerebras_dev.py   # development-only Cerebras adapter (never in Kaggle)
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
- **Cerebras API (development only, network-based, no local GPU needed):** high-throughput prototyping of prompts, perception formats, JSON schema/repair, and reflection-memory formats against public/local games. See "Cerebras development boundary" below — it is never part of the Kaggle submission runtime.

Colab results are provisional. Kaggle is the deployment source of truth because CUDA, vLLM, memory, attached inputs, runtime limits, and offline behavior can differ. Record the actual GPU, GPU memory, precision/quantization, model revision, package versions, and prompt/config hash for every model run.

## Cerebras development boundary

The Cerebras Inference Cloud is currently (verified August 2026, may change —
re-check `inference-docs.cerebras.ai/models/overview` before relying on it)
serving `gemma-4-31b` — the same model family and parameter count as our
Kaggle target — as a **preview** model on its shared/public endpoint, at
~1850 tok/s, and it is the first model on the platform with **image input
support** (screenshots, UI states, documents — a good match for ARC-AGI-3
grid frames, not just ASCII/text). This is a materially stronger dev proxy
than a generic "different, text-only model" would be. It is not, however, a
free pass to skip Colab/Kaggle validation:

- **Preview, not production.** Cerebras's own docs say preview models "are
  intended for evaluation purposes only... may be discontinued on short
  notice." Never let a submission-critical path depend on preview
  availability persisting.
- **Model identity is not guaranteed identical across tiers.** The public
  preview model ID is `gemma-4-31b`; Cerebras's separate dedicated-endpoint
  catalog lists `google/gemma-4-31b-it`. Do not assume these are the exact
  same artifact as whatever we load in Colab/Kaggle — query the account's
  live model list and record the exact model ID and API version used,
  rather than hard-coding a name from documentation or from this file.
- **Quantization/serving differences remain.** Cerebras stores weights with
  its own weight-only quantization; our Colab/Kaggle deployment may use a
  different precision. A Cerebras result is a fast signal for **rejecting
  weak ideas and comparing prompt/perception formats**, not a substitute
  for the exact deployed Gemma backend.
- **Image-input support is itself flagged "Preview"** in Cerebras's own
  capability docs. Treat it as usable-but-evolving, not a stable long-term
  contract.

Any prompt, memory, perception, or heuristic change selected using Cerebras
— text or image mode — must still be reproduced with the exact Gemma-4-31B
backend in Colab before promotion toward Kaggle.

The Cerebras API key is a developer credential, not a model artifact. Store
it only as `CEREBRAS_API_KEY` in the user's environment or an ignored local
secret store. Never place it in code, Git, notebooks, Drive, logs,
experiment records, generated Kaggle artifacts, or chat.

Implement it behind the same narrow `ModelBackend` protocol as the fake and
Gemma backends, in its own module (`zerx/backends/cerebras_dev.py`), with
backend id `cerebras_dev`. It must:

- accept either a compact ASCII/grid representation or the same labeled
  image sent to Gemma — configurable, since both are now plausible given
  image-input support; default to whichever perception format is under
  active ablation;
- read `CEREBRAS_API_KEY` only at its own client construction (the one
  deliberate exception to "only `config.py` reads env vars" — a credential
  is not a config value); record `credential_present: true/false` in
  configuration/logs, never the key;
- use an explicitly recorded Cerebras model ID and API version, queried
  from the account rather than assumed;
- request the same one-action JSON schema, then run through the identical
  local parser and legal-action validation as every other backend;
- use bounded retries, a request timeout, and rate-limit handling; record
  latency and token usage, never credentials;
- never receive private evaluation frames, competition secrets, or
  personal data.

Hard safeguards:

- configuration validation rejects `cerebras_dev` whenever `platform=kaggle`,
  competition mode is active, or internet is disabled;
- the Kaggle build excludes the Cerebras SDK/client and any Cerebras-specific
  secret-loading code;
- a packaging test (`zerx/secret_scan.py`) scans the generated artifact for
  `CEREBRAS_API_KEY`, `api.cerebras.ai`, and known secret values, and fails
  the build if found;
- no fallback from the Kaggle Gemma backend to any network backend is ever
  permitted, for any reason, including Kaggle-side failures.

Do not describe Cerebras output as a Gemma-Kaggle baseline — it is a
development-proxy result with its own backend/model identifier.

## Scope

Baseline features:

- Gemma-4-31B only;
- labeled-frame and/or compact grid perception;
- exactly one proposed environment action per policy call;
- strict JSON parsing, at most one bounded repair attempt, legal-action validation, and a safe fallback chain;
- reflection memory refreshed at a configurable interval;
- NumPy click-candidate heuristic and dead-signature tracking;
- action-budget telemetry and a configurable shift from probing to execution;
- flags/configuration that allow each feature to be disabled independently;
- a `ModelBackend` protocol that permits a development-only `cerebras_dev`
  implementation without weakening the Gemma-only submission contract.

Out of baseline scope:

- Murad/Forge VLM-style arbiter or multiple candidate generators as default behavior;
- multi-model abstraction beyond Colab/Kaggle/Cerebras hosting proxies of
  the same target model;
- code-writing or REPL agents inspired by The Duck;
- multi-action queues;
- training, fine-tuning, or distilling Gemma;
- any external inference API in the Kaggle runtime, under any condition;
- milestone-prize publication logistics.

An arbiter may be added later only as an off-by-default experiment with a stated hypothesis and baseline comparison.

Never, under any circumstance, at any phase:

- read hidden game/engine source, runtime fields, or internal state not
  exposed through the official frame/action API;
- clone or reconstruct the real environment's internal state for search;
- take unscored counterfactual actions against the real implementation;
- branch behavior on a specific game ID, or use a public-game lookup table
  or memorized solution;
- run unbounded search (BFS/A*/IDDFS without an explicit node/time budget);
- add a CNN or other trained-fallback component without a stated
  training/evaluation case;
- treat public-game aggregate score as evidence of private-leaderboard
  performance (see `STRATEGY.md`'s Tycho-numbers caveat — public-game
  scores in the 79–100 RHAE range are a real but different measurement
  than the hidden leaderboard).

These come from `STRATEGY.md`'s prior-art review (ProjectForty2 FORGE, in
particular, used source-assisted state cloning and hidden-field
inspection — excluded here on both competition-integrity and
generalization grounds).

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
- configuration serialization and feature disabling;
- competition-mode (`platform=kaggle`) configuration rejects `cerebras_dev`
  before any client is constructed;
- generated-artifact secret/endpoint scanning finds a planted fake key and
  passes on a clean artifact;
- a contract test proving `FakeModelBackend`, `CerebrasDevBackend`, and
  `GemmaModelBackend` all satisfy the same `ModelBackend` protocol/response
  shape.

Mock all Cerebras calls in the default local suite — no `CEREBRAS_API_KEY`
should be required to run `pytest`. Live Cerebras tests are opt-in and
clearly marked (e.g. `pytest -m cerebras_live`) so teammates without the key
still run the full required suite.

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

Treat a Kaggle compatibility/submission test as a long asynchronous job: the
notebook may run for up to ~9 hours, queueing and scoring time are not
guaranteed, and failure logs can be limited. Never schedule the first valid
submission attempt for the delivery day itself.

Validate `submission.parquet` by byte size, successful readback, schema, rows, nulls, and embedded/source identity. A file's existence alone does not prove success. Submission is a separate explicit approval after validation.

### 5-day delivery schedule (Day 1 today, due Day 5)

This window is tight against Kaggle's ~9-hour run + unpredictable queue
time — the schedule below deliberately front-loads Kaggle attempts instead
of saving them for the end. Detailed, bite-sized steps for Day 1's local
skeleton live in `docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md`;
this is the cross-day shape, see `docs/TEAM_WORKFLOW.md` for full detail.

- **Day 1 (today):** verify the real starter repo/API, record `baseline-000`,
  submit the unmodified starter as a known-working Kaggle smoke test before
  the day ends. In parallel, build the local `zerx/` skeleton (fake backend,
  `cerebras_dev` adapter + hard lockout, secret scan).
- **Day 2:** complete the model-free `zerx/` package and its full test
  suite; load Gemma-4-31B on Colab and complete one model-in-loop smoke
  game (`baseline-100`); use Cerebras in parallel to compare perception
  formats and prompts fast. Diagnose the Day 1 Kaggle result immediately.
- **Day 3:** add only the minimum memory/heuristic-fallback the design
  calls for; run small comparable Colab evaluations; **start the first full
  Kaggle compatibility run as early in the day as possible** — this is the
  latest point a first real (non-smoke) Kaggle result can start and still
  leave a day to react.
- **Day 4:** analyze the Day 3 Kaggle result; fix only demonstrated
  failures (legality, malformed output, reset, OOM, timeout); freeze the
  architecture; **start the intended final Kaggle submission by the end of
  Day 4** — not Day 5, which has no run-time margin left.
- **Day 5 (due):** monitor/confirm the final submission's terminal status;
  if it failed, one narrow recovery using the last known-good candidate —
  no new architecture; record the best scored submission; finalize
  `docs/HANDOFF.md`.

If Day 3's first Kaggle run is delayed or fails in a way that threatens Day
4's final-submission start, say so explicitly and re-scope rather than
silently slipping the schedule.

## Definition of done

A change is complete when its hypothesis is stated, focused and full tests pass, the resolved config is recorded, no unrelated edits or secrets are included, and remaining Colab/Kaggle verification is explicit.

A Colab experiment is complete when results and environment metadata are persisted and compared with the unchanged baseline.

A Kaggle experiment is complete only when the kernel reaches a terminal status, logs and outputs are downloaded and checked, the artifact is structurally valid, and the experiment record is updated. A submission is complete when Kaggle reports its terminal evaluation result and that result is recorded.
