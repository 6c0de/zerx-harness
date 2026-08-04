# Team workflow — ARC-AGI-3 Zerx baseline

Tool-neutral: Codex and Claude Code follow the same repository contract in
[`AGENTS.md`](../AGENTS.md). This file is the day-by-day shape of the work;
`AGENTS.md` is the rulebook; [`STRATEGY.md`](../STRATEGY.md) is what
strategic hypotheses to test and why (prior-art adoption decisions,
experiment ladder). For Day 1's local model-free skeleton, the
authoritative bite-sized steps are in
[`docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md`](superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md)
— this file doesn't repeat them, it says where they sit in the schedule.
None of these three substitute for `HANDOFF.md`, experiment records,
commits, resolved configurations, or test results.

## Operating model

```text
Local Codex / Claude Code + Git
  model-free implementation and tests
            v local gate
Cerebras (dev-only, network, no local GPU)
  fast prompt/perception/JSON/memory experiments
            v promote winners only
Colab Pro A100/L4
  Gemma-4-31B load, inference, public-game ablations
            v Colab gate
Kaggle RTX Pro 6000
  offline compatibility, packaging, official evaluation
```

Local Git is the code source of truth. Cerebras is a fast, cheap proxy lane
for rejecting weak ideas before they cost Colab or Kaggle time — it is not
a replacement for the Colab Gemma gate and is never called from a Kaggle
submission (see `AGENTS.md`'s "Cerebras development boundary" for exactly
what it can and can't tell you; it now supports image input in preview,
which is a stronger proxy than a text-only lane, but still not Kaggle-parity
proof). Colab is the primary model-in-the-loop laboratory. Kaggle is the
deployment and scoring source of truth.

AI subscriptions provide workers, not project memory. All decisions, code,
configurations, results, and handoffs must be reconstructable from the
repository and linked experiment artifacts — not from a chat log.

## 5-day delivery schedule

**Today is Day 1. Delivery is due Day 5.** A serious Kaggle test can consume
the notebook's ~9-hour run allowance plus additional queue/scoring time, and
runs can fail with limited logs — so this schedule front-loads Kaggle
attempts instead of saving them for the end, and has no dedicated recovery
day after delivery. If Day 3's first real Kaggle run slips, say so and
re-scope rather than silently compressing Day 4/5.

### Day 1 — Foundation, skeleton, and a smoke submission

- Verify the actual official starter, agent entrypoint, build command,
  scoring/toolkit version, and submission mechanism (Task 1 of the local
  skeleton plan). Record `baseline-000`.
- Put `AGENTS.md`, this file, and `docs/HANDOFF.md` at the repository root
  / `docs/` — done.
- Start a known-working Kaggle smoke submission (the unmodified starter)
  before the day ends — don't wait for a "real" candidate; queue/scoring
  time is unpredictable and Day 1 is the cheapest day to eat that latency.
- In parallel, build the local `zerx/` skeleton and fake backend (Tasks
  2–10 of the local skeleton plan), the `ModelBackend` protocol, the
  `cerebras_dev` adapter with its hard competition-mode lockout, and the
  secret-scan test.
- Confirm the Cerebras account's actually-available model IDs (query, don't
  assume — public preview vs. dedicated catalogs differ) without exposing
  the key anywhere in the repo.

Exit condition: one reproducible baseline commit, and one Kaggle run in
progress or completed.

### Day 2 — Minimal Gemma baseline

- Finish perception, single-action schema/parser, legal guard, fallback
  chain (remaining tasks of the local skeleton plan); full local suite
  green.
- Wire the thin harness adapter (`agent/my_agent.py`) to the real upstream
  API discovered on Day 1; run `make verify-local`.
- Load the exact Gemma-4-31B revision on Colab A100 and complete one
  model-in-loop smoke game — this is `baseline-100`.
- Before/alongside the Colab run, use Cerebras for rapid ASCII-grid **and**
  labeled-image prompt experiments (both are viable inputs to
  `gemma-4-31b` on Cerebras now); keep only candidates that pass the same
  parser/legal guard, and re-run the winner on Gemma/Colab before trusting
  it.
- Diagnose the Day 1 Kaggle smoke result immediately; fix packaging or
  environment failures before any feature work.

Exit condition: `baseline-100` works locally and in Colab; Kaggle packaging
is known-valid, or has one clearly owned blocker.

### Day 3 — First complete candidate, first real Kaggle run

- Add only the minimum memory and heuristic-fallback the design calls for.
- Run small, comparable Colab evaluations across the candidates Cerebras
  narrowed down.
- Select the simplest candidate and **start its first full Kaggle
  compatibility/submission test as early in the day as possible** — this
  is the last point a first real Kaggle result can start and still leave
  Day 4 to react to it.
- Don't wait idle on the Kaggle run: review logs, tests, timeout paths, and
  environment parity in parallel.

Exit condition: a complete Zerx candidate is running on Kaggle; all
source/config identifiers are recorded in the experiment table.

### Day 4 — Evidence-driven fixes and the final submission

- Analyze the Day 3 Kaggle result.
- Fix only demonstrated failures: legality, malformed output, reset
  behavior, OOM, timeout, memory isolation, or clearly poor heuristic
  behavior. No new architecture.
- Freeze dependencies, model revision, dtype/quantization, perception
  format, and prompt structure.
- **Start the intended final Kaggle submission by the end of Day 4** — not
  Day 5, which has no run-time margin left given the ~9-hour + queue
  window.

Exit condition: the intended final submission is running, with a
previously validated fallback candidate on hand.

### Day 5 — Delivery

- Monitor the final submission to terminal status; do not launch
  overlapping duplicate runs.
- If it fails, one narrow repair or fallback to the last known-good
  candidate — do not redesign the agent.
- Validate `submission.parquet` (byte size, readback, schema, rows, nulls,
  source identity) — existence alone is not success.
- Record the best scored submission and select final submissions in Kaggle
  if the competition requires it.
- Complete `docs/HANDOFF.md` so the state is reproducible without this
  chat.

Exit condition: at least one confirmed valid submission and a reproducible
final repository state.

## Team and PC switching protocol

```text
repository/
├── AGENTS.md
├── CLAUDE.md
├── docs/
│   ├── TEAM_WORKFLOW.md
│   ├── HANDOFF.md
│   └── superpowers/
│       ├── specs/
│       └── plans/
├── zerx/
├── agent/
├── eval/
└── tests/
```

`CLAUDE.md` points to `AGENTS.md`; it does not maintain a second rule set.
Codex and Claude may suggest different implementations, but use the same
branch, tests, configuration schema, and definition of done.

Before handoff, the current owner updates `docs/HANDOFF.md`: branch/commit,
experiment/config ID, completed changes and tests, result-artifact
locations, active Colab/Kaggle job (notebook version, start time, account
owner), blockers, and the exact next action.

The receiving owner does not continue from an uncommitted folder copied
between PCs — pull the named branch/commit, recreate the environment, run
the smoke suite first.

Parallel work happens on separate branches with non-overlapping ownership
(e.g. core policy, perception/heuristics, evaluation, Colab/Kaggle
deployment). Merge through review; never run two AI agents on the same
files at the same time.

## Cerebras rapid-development lane

Use Cerebras only with public/local game data and derived representations:

1. Store `CEREBRAS_API_KEY` outside the repository.
2. Query the authenticated models endpoint once; record only the selected
   model ID and API version, never the key or raw auth response.
3. Either compact ASCII/grid text or the same labeled image sent to Gemma
   is a valid input — `gemma-4-31b` on Cerebras supports both as of this
   writing (image input is itself flagged "Preview" by Cerebras; re-verify
   before depending on it for anything time-sensitive).
4. Request strict structured JSON output where supported; local extraction,
   schema validation, coordinate checks, and fallback logic stay
   authoritative regardless of backend.
5. Cache responses by sanitized input/config hash for repeatable tests and
   lower API usage. Never cache credentials or authorization headers.
6. Sweep prompt/perception/parser/memory variants with fixed public games
   and seeds.
7. Promote only the best one or two ideas to Colab with the exact
   Gemma-4-31B backend before trusting them.

Good uses this sprint: checking whether a prompt/perception format produces
legal single actions; stress-testing structured JSON and repair paths;
comparing reflection-summary formats; high-throughput regression tests of
the backend/harness boundary; narrowing candidates before a Colab run.

Invalid uses: calling Cerebras from the Kaggle evaluation notebook;
treating a Cerebras result as a Gemma-Kaggle score; sending private
evaluation frames or secrets; using it to skip the required Colab Gemma
validation; expanding the sprint into training/distillation work.

## Team controls

- One owner, branch, and immutable experiment ID at a time.
- One shared result table: prior-art feature under test, baseline
  experiment ID, primary changed factor, model/prompt/image configuration,
  commit, config hash, backend, GPU, games/seeds, scored actions, resets,
  completion/RHAE, model calls, latency, inference cost, repairs,
  fallbacks, transition-evidence completeness, world-model
  exactness/coverage (once `world_model_on` is exercised — see
  `STRATEGY.md`), planner recommendation/validation/following (once
  `planner_on` is exercised), and a decision: promote, reject, or rerun.
- Peer review before any Colab sweep or Kaggle run.
- Never put credentials in Git, Drive notebooks, prompts, logs, or result
  files.
- Follow the competition's current team/data/model/submission rules.
- Use public games for development; treat private-game generalization as
  the actual objective.

## Kaggle integration choice

Use the official Kaggle CLI and the starter's verified `make` build/submit
flow first — don't assume `make play-local`/`make submit` exist until
confirmed in the checked-out repository (starter kits evolve). A
third-party Kaggle MCP/skill is optional, not required; only install one
after reviewing source, maintenance, credential handling, network access,
and submission permissions, and it must never be allowed to submit or
start quota-consuming runs without confirmation.

The Cerebras API is not a Kaggle integration and cannot turn a hosted API
model into an offline Kaggle competition model. It may generate code,
prompts, tests, or development evidence before packaging. The submitted
agent must load and run its permitted local Gemma weights and dependencies
from attached Kaggle inputs, with zero external API calls.
