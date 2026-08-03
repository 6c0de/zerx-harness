# CLAUDE.md

Project rules and operating constraints live in [`AGENTS.md`](AGENTS.md) —
they apply to this Claude Code session too (mission, 5-day delivery
schedule, team contract, scope, control-flow requirements,
config/reproducibility rules, Cerebras development boundary,
environment-promotion gates). Read it before making changes. Do not
duplicate or override `AGENTS.md` here.

Also read before touching agent behavior or switching PCs/owners:

- [`docs/TEAM_WORKFLOW.md`](docs/TEAM_WORKFLOW.md) — day-by-day schedule and promotion-gate detail.
- [`docs/HANDOFF.md`](docs/HANDOFF.md) — current branch, experiment, active Colab/Kaggle runs, next action.
- [`docs/superpowers/specs/2026-08-03-arc-agi3-baseline-design.md`](docs/superpowers/specs/2026-08-03-arc-agi3-baseline-design.md) — design rationale.
- [`docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md`](docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md) — bite-sized Phase 0/1 implementation plan.

Do not rely on a previous Claude conversation as project memory — record
durable decisions and results in the repository. If repository documents
conflict, stop and ask the human owner to resolve it before making a broad
change.

Cerebras (`cerebras_dev` backend) is development-only. Never put
`CEREBRAS_API_KEY` in this file, source code, commits, notebooks, logs,
prompts, or handoffs. Never call it from competition/Kaggle mode. Any idea
selected using it must pass the Colab Gemma gate before Kaggle promotion —
see `AGENTS.md`'s "Cerebras development boundary" for the current, verified
capability boundary (it now supports image input in preview, not
text-only — don't assume otherwise without re-checking Cerebras's docs).
