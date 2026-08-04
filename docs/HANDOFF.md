# Project handoff

Copy this template's structure for each handoff entry (newest first, or one
file per handoff under `docs/handoffs/` if the history grows long — not
needed yet). See `docs/TEAM_WORKFLOW.md` for the 5-day schedule this feeds
into.

- Updated at: 2026-08-04
- Current owner: (local session, Claude Code)
- Next owner: unassigned
- Branch: `day1-local-skeleton`
- Commit: `4b8ba8e`
- Experiment ID: `baseline-000` (starter smoke test); no model-in-loop experiment yet
- Config ID/hash: n/a — no `Config` instance has been run against a real model yet (default `Config().config_hash()` is deterministic but no experiment record has been written)
- Sprint day (1–5): Day 1

## Objective

Execute `docs/superpowers/plans/2026-08-03-arc-agi3-local-skeleton.md` Tasks
1–15: import the real upstream ARC-AGI-3-Kaggle-Starter, record
`baseline-000`, and build the complete model-free `zerx/` package (types,
config, perception, heuristics, memory, budget, model backend protocol,
dev-only Cerebras backend, secret scanner, JSON policy parsing + `decide()`
orchestrator, evidence-first transition ledger) plus a thin `agent/my_agent.py`
harness adapter wired to the real upstream API. No GPU, no model load,
nothing Kaggle-submission-related — the "Local → Colab" promotion gate from
`AGENTS.md`.

## Completed changes

All 15 plan tasks done, each with an individual TDD implementer pass and an
independent task-reviewer pass (see `.superpowers/sdd/progress.md`, gitignored
local ledger, for the full commit-range-per-task record). Plus three
follow-on fixes from the final whole-branch review, all human-approved
before implementation, each independently re-reviewed:

- `zerx/types.py`, `zerx/config.py`, `zerx/perception.py`, `zerx/heuristics.py`,
  `zerx/memory.py`, `zerx/budget.py`, `zerx/model_backend.py`,
  `zerx/backends/cerebras_dev.py`, `zerx/secret_scan.py`, `zerx/policy.py`
  (`parse_action` + `decide()`), `zerx/transitions.py`, `eval/run_ablation.py`.
- `agent/my_agent.py` replaces the vendored random-baseline agent, wiring
  `decide()`/`TransitionLedger`/`DeadSignatureTracker` together using the
  **real** upstream `FrameData`/`GameAction`/`GameState` API (verified
  against the installed `arcengine` package and a live probe, not assumed —
  see `docs/superpowers/experiments/baseline-000.md`). Two of the plan's own
  illustrative placeholders were wrong and corrected: `state` is a
  `GameState` enum (not a string), and `is_done` must stop only on `WIN`,
  not `GAME_OVER`. A 10th, previously-undocumented bug was found and fixed:
  `GameAction(id)` raises `ValueError` in the vendored package due to a
  stale `_value2member_map_`; use `GameAction.from_id(id)` instead.
- `choose_action` now has a top-level exception boundary (never raises,
  even if `decide()`/perception/transitions themselves fail) with a
  from-scratch safe fallback that works off raw upstream data only.
- `scripts/build_notebook.py` now bundles `zerx/*.py` (never
  `zerx/backends/`, the Cerebras-only module) into the Kaggle notebook
  build, gated by a build-time `zerx/secret_scan.py` call that fails the
  build on any finding — closes a real packaging break Task 14 introduced
  (nothing previously shipped `zerx/` alongside `agent/my_agent.py`, which
  now imports it).
- `decide()`'s budget signal now additionally triggers the heuristic path
  when `should_favor_execution=True` and a candidate exists (purely
  additive, byte-for-byte preserves the existing `heuristic_first` branch).
  Decision telemetry (`source`/`repaired`/config hash) now reaches
  `GameAction.reasoning`, which the vendored framework's recorder reads.
- One Windows-only upstream script bug fixed
  (`scripts/slim_framework.py`'s `write_text` missing `encoding="utf-8"`).

## Tests executed and results

`.venv\Scripts\pytest.exe tests/ -q` → **114 passed, 0 failed** (Windows
venv; `.venv/bin/pytest` from the plan's own commands is the Mac/Linux path
and doesn't exist on this checkout — see
`docs/superpowers/experiments/baseline-000.md`'s "Windows-native environment
deviations" section).

`scripts/build_notebook.py` run directly (not via `make`, not `make submit`)
— completes without the secret-scan `SystemExit`; `notebooks/submission.ipynb`
(gitignored) inspected and confirmed to contain real `zerx/*.py` source with
zero `zerx/backends/` content.

`agent/my_agent.py` live-smoke-tested twice against real local games:
during Task 14 (`ls20`, `cn04`, `vc33`, several actions each) and again
after both follow-on fix commits (`8ade927`, `4b8ba8e`) via
`scripts/play_local.py --game ls20,cn04,vc33 --max-steps 50` — all three
ran the full 51 actions with no `AttributeError`/`ImportError`/`TypeError`,
`NOT_FINISHED`/`levels=0` (expected: `GemmaModelBackend.generate()` is
still a stub, so every action falls through `decide()`'s fallback chain).

## Colab state

Not started. `GemmaModelBackend.generate()` is still `NotImplementedError`
by design (Task 8) — no model has been loaded anywhere in this branch.

- Account owner: n/a
- Notebook URL/version: n/a
- Git commit checked out: n/a
- GPU/backend profile: n/a
- Status/results location: n/a

## Cerebras development state

Not started. No `CEREBRAS_API_KEY` exists in this environment; no live
Cerebras call has ever been made — every `zerx/backends/cerebras_dev.py`
test injects a fake `http_post` or a literal string (verified during Task 9's
review).

- Account/key owner (name only, never the key): n/a
- Model ID and API version (query the account, don't assume): n/a
- Perception mode used (ascii / image): n/a
- Backend/config hash: n/a
- Public games/seeds: n/a
- Results location: n/a
- Gemma reproduction status (has this been re-validated on Colab?): n/a

## Kaggle state

Not started. No `make submit`, no Kaggle CLI call, no notebook push. Day 1's
plan called for "submit the unmodified starter as a known-working Kaggle
smoke test before the day ends" (`docs/TEAM_WORKFLOW.md`) — **this has not
happened yet** and needs explicit owner approval per `AGENTS.md`'s Kaggle
gate before it does.

- Account owner: n/a
- Kernel/notebook slug and version: n/a
- Submission ID: n/a
- Started at: n/a
- Expected maximum run window (~9h + queue): n/a
- Current status: not started
- Output/results location: n/a

## Known failures or risks

Recorded during per-task and final whole-branch review; all judged
non-blocking for Day 1, but real and worth Day 2's attention:

1. **`zerx/backends/cerebras_dev.py`'s `platform` kwarg defaults to
   `"local"` and is never wired to the real `Config.platform`** — currently
   inert because nothing in this branch ever constructs
   `CerebrasDevBackend` outside its own tests (`agent/my_agent.py` always
   uses `GemmaModelBackend`). `Config.__post_init__`'s own
   `cerebras_dev`+`kaggle` rejection is the only enforcement that's actually
   wired today. **Before any future task adds a backend-selection factory**
   (e.g. `make_backend(config)`), it must forward `platform=config.platform`
   explicitly, or this second defense-in-depth layer silently becomes a
   no-op the day it's needed.
2. **No true rate-limit backoff** in `CerebrasDevBackend.generate()`'s
   retry loop (immediate retry, no `Retry-After` handling) — inherited
   verbatim from the plan's own Task 9 code. Only matters once a live
   (opt-in, separately-marked) Cerebras test exists; none does yet.
3. **`parse_action(None, ...)` raises `AttributeError`**, technically
   breaching its own "never raises" docstring contract — inherited from the
   plan's own Task 11 code. Inert in practice: `decide()` (Task 12) wraps
   the only real call site (`backend.generate()` + `parse_action()`) in one
   shared `try/except Exception`.
4. **`history` is computed in `agent/my_agent.py` (last 4 frames converted
   per action) and passed to `decide()`/`perceive()`, but `perceive()`
   currently ignores it entirely** — four wasted full-grid conversions per
   action today. Documented as deliberate interface stability for future
   movement-delta perception, not a bug, but worth knowing before
   optimizing anything.
5. Two Windows-only local dev-tooling issues were found and fixed (not
   upstream API bugs): `scripts/slim_framework.py` needed `encoding="utf-8"`
   on its `write_text` call, and running any script producing Unicode
   console output needs `PYTHONIOENCODING=utf-8` on this machine's
   (Turkish, `cp1254`) console codepage. Full detail in
   `docs/superpowers/experiments/baseline-000.md`.

## Exact next action

1. Per `docs/TEAM_WORKFLOW.md`'s Day 1 exit condition ("one Kaggle run in
   progress or completed"): get explicit owner approval, then push a known-
   working Kaggle smoke submission via `make submit` — this has not
   happened yet and is Day 1's other, still-open exit condition. Decide
   first whether that smoke submission should be the unmodified starter
   (matches `baseline-000` exactly, lowest risk) or the current
   `zerx`-wired agent (exercises the new packaging path end-to-end,
   including the secret-scan gate, for the first time against real Kaggle
   infra) — owner's call, not decided in this session.
2. Day 2 per `AGENTS.md`/`docs/TEAM_WORKFLOW.md`: load the exact
   Gemma-4-31B revision on Colab Pro A100/L4 and complete one
   model-in-loop smoke game (`baseline-100`) — `GemmaModelBackend.generate()`
   is intentionally still a stub (`NotImplementedError`) until then.
3. Address known-risk #1 above (Cerebras `platform` wiring) if/when a
   backend-selection factory is added — don't let it get wired without the
   `config.platform` forward.

## Uncommitted or external artifacts

None tracked or required. `.venv/`, `vendor/ARC-AGI-3-Agents/`,
`environment_files/`, `notebooks/submission.ipynb`, and `.superpowers/`
(this session's SDD ledger/briefs/reports) all exist locally and are
gitignored — reproducible from `docs/superpowers/experiments/baseline-000.md`'s
recorded setup commands, not source-of-truth. No credentials of any kind
exist in this environment (no `CEREBRAS_API_KEY`, no Kaggle token) — every
test that would need one injects a fake instead.
