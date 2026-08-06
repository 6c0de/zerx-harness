# ARC-AGI-3 Full Repository Audit — 2026-08-06

**Auditor role:** principal research engineer / red-team reviewer / competition engineer
**Branch audited:** `master` @ `b405d3b`
**Work branch:** `feat/policy-prompt-legal-budget` (master never modified)
**Method:** every tracked file accounted for; all claims below are backed by executed
commands, not by reading comments or documentation.

---

## 1. Executive Summary

**Is master healthy? No. As of `b405d3b`, a Kaggle submission built from master
scores zero — it cannot even import our agent.**

Three independent submission-breaking defects were confirmed by execution, not
inspection:

1. The Kaggle bundle omits `zerx/backends/` (correct, for secret hygiene) while
   `zerx/model_backend.py` imports it at module level. Importing the bundle as
   Kaggle would raises `ModuleNotFoundError: No module named 'zerx.backends'`.
   `agent/my_agent.py` imports `zerx.model_backend`, so **the agent cannot load at
   all**.
2. The notebook writes `%%writefile /tmp/zerx/<mod>.py` without ever creating
   `/tmp/zerx`. IPython's writefile magic is a plain `open(path,'w')` with no
   `makedirs` (verified against IPython 9.16.1 source), so the **first** such cell
   raises `FileNotFoundError` on a fresh kernel.
3. Even had those worked, **the submission contains no model.** `kernel-metadata.json`
   has `model_sources: []` and `dataset_sources: []`, no cell starts vLLM, and no
   `ZERX_BACKEND` is ever set — so `Config.backend` defaults to `"fake"`, whose
   every `generate()` raises, and the agent silently degrades to heuristics-only.

Defects 1 and 2 are fixed on this branch with regression tests. Defect 3 is a
**strategy/workstream gap**, not a localized bug, and is reported rather than
silently rewritten (see §16).

**Biggest likely leaderboard bottleneck:** the agent has no reasoning in the
submission path at all (finding 003), and until 005 it also had **no progress
signal** — `levels_completed` was discarded at the adapter, hardcoding every
`score_delta` to 0.

**Biggest hidden-game generalization risk:** low from overfitting (see §13 — the
repo is genuinely clean of game IDs and magic coordinates in production code), but
**high from context overflow**: a legal 64×64 frame could render a 49,392-token
prompt (finding 006).

**Biggest Kaggle deployment risk:** findings 001/002 (now fixed), then the
untested end-to-end packaging path generally — the pre-existing packaging tests
asserted the *mechanism* (which files are globbed) and never the *outcome* (does
the bundle import).

**Most under-appreciated risk:** **four of the ablation flags cannot change
behaviour at all** (findings 008–011). Any A/B run using them would report "no
effect" and could lead the team to discard good ideas for the wrong reason. This
directly undermines AGENTS.md's promotion-gate methodology.

---

## 2. Repository / Git Reality

| Item | Value |
|---|---|
| master SHA | `b405d3b19f2b7b5c87d5bf541813835cbb6c6c0e` |
| branch SHA (start) | `e402a0d6598cf335286034d472ef11448bbdb65a` |
| merge-base | `b405d3b` (branch is a clean descendant; no divergence) |
| working tree at audit start | clean |
| master modified? | **No** |
| Kaggle submission made? | **No** |

Verified Day-1/2/3 status (reconstructed from git, not from docs):

- **Day 1 / Day 2** — plans and experiment records exist (`baseline-000`,
  `baseline-100`); code is present and tested. **Category A (completed)**, with the
  caveat that "tested" means model-free unit tests.
- **Day 3** — four tracks merged via `integration/day3`: `d0f3dfb` (exp-150),
  `6ed3fc2` (exp-140), `f3e9e57` (baseline-130), plus baseline-115. All merges are
  real and conflict-free. **"Merged" ≠ "integrated"** — see §4.
- **baseline-120** — a *second* four-track wave merged via `integration/baseline-120`
  (`f44e631`, `661d636`, `e020554`, `be202b4`). This is newer than the Day-3 wave and
  is what master actually points at.

Note: commit `d619dab` already recorded "the missing-legal-actions prompt finding"
during Cerebras validation — the fix landed on this branch as `e402a0d`, closing a
gap the team had already observed but not yet acted on.

---

## 3. File Coverage

**85 tracked files, 85 accounted for (100%).**

| Classification | Count | Files |
|---|---|---|
| REVIEWED-DEEPLY | 20 | `agent/my_agent.py`; `zerx/{__init__,budget,candidates,config,exact_state_memory,heuristics,memory,model_backend,perception,policy,scene,secret_scan,transitions,types}.py`; `zerx/backends/{__init__,cerebras_dev}.py`; `eval/run_ablation.py`; `scripts/build_notebook.py`; `scripts/build_colab_notebook.py` |
| REVIEWED-DEEPLY (tests) | 24 | all `tests/*.py` — read for coverage gaps and for tests asserting obsolete/insufficient contracts |
| REVIEWED-CONFIG | 6 | `Makefile`, `notebooks/kernel-metadata.json`, `requirements-zerx.txt`, `.gitignore`, `eval/__init__.py`, `tests/__init__.py` |
| REVIEWED-DEEPLY (scripts) | 2 | `scripts/play_local.py`, `scripts/slim_framework.py` |
| REVIEWED-DOC | 33 | `AGENTS.md`, `CLAUDE.md`, `STRATEGY.md`, `docs/HANDOFF.md`, `docs/TEAM_WORKFLOW.md`, `docs/superpowers/**` (3 experiments, 10 plans, 2 plan-set dirs with 13 files), `docs/superpowers/specs/*` |
| GENERATED | 0 tracked | `notebooks/submission.ipynb` is gitignored (built by `make notebook`) |
| VENDORED | 0 tracked | `vendor/ARC-AGI-3-Agents` is gitignored; inspected on disk for API/lifecycle facts but not shipped by us |
| BINARY | 0 | none |

No directory was skipped. `vendor/` and `.venv/` are untracked and excluded from
the ledger by design, but were read to verify upstream API contracts.

---

## 4. Day-3 Integration Matrix

"Merged" was verified against "actually reachable in the live `decide()` loop."

| Track | Plan | Implementation | Tests | Integrated? | Default on? | Evidence | Problem | Sev |
|---|---|---|---|---|---|---|---|---|
| **baseline-115** exact-state suppression | `parallel-day3/person-1-baseline-115.md` | `zerx/exact_state_memory.py`, wired in `my_agent.py:191-258` | `test_exact_state_memory.py`, `test_my_agent_exact_state.py` | **Yes** | No (`exact_state_suppression_on=False`) | grep: two call sites in `my_agent` | Its `level_delta` channel was inert because `score` was hardcoded 0 | P1 → **fixed** (005) |
| **baseline-130** structured/hypothesis memory | `parallel-day3/person-2-baseline-130.md` | `zerx/memory.py:56-266` | `test_structured_memory.py` | **Partially — state updates, but never reaches the model** | No | `render_for_prompt` has **0 production callers**; `build_prompt` never receives structured memory | Computes state each step, then discards it; costs a full `perceive()` per action for nothing | P1 (009) |
| **exp-140** candidate generation/arbiter | `parallel-day3/person-3-exp-140.md` | `zerx/candidates.py`, wired at `policy.py:232` | `test_candidates.py` | **Partially** | No (`candidate_count=1`) | `decide()` calls `select_candidate(candidates, config)` with **no arbiter arg** | `arbiter_on` gated on `arbiter is not None` → can never activate in production | P1 (010) |
| **exp-150** Duck object tools | `parallel-day3/person-4-exp-150.md` | `zerx/scene.py` (549 lines) | `test_scene_objects.py`, `test_transition_classification.py` | **No — by design** | No | module docstring: "nothing in the live decide() loop calls this module yet" | Legitimate **category G**. But `duck_objects_on` is in the ablation matrix while controlling nothing | P2 (011) |

**No merge-conflict artifacts, no semantic conflicts, no obsolete-interface tests
were found.** The four tracks compose cleanly. The systemic issue is not conflict —
it is that three of four tracks terminate in code that never influences an action.

---

## 5. Findings

### ARC-AUDIT-001 — Kaggle bundle cannot be imported (`zerx.backends`)

- **Severity:** P0 · **Confidence:** CONFIRMED · **Category:** Kaggle packaging
- **File:** `zerx/model_backend.py:17` (master)
- **Observed:** importing the bundle exactly as Kaggle would raises
  `ModuleNotFoundError: No module named 'zerx.backends'`.
- **Evidence:** reconstructed master's bundle from `git show master:zerx/*.py` into a
  clean dir (no `backends/`, matching `zerx_bundle_files()`), then
  `PYTHONPATH=$bundle python -c "import zerx.model_backend"` → ModuleNotFoundError.
- **Root cause:** `build_notebook.zerx_bundle_files()` deliberately uses a
  non-recursive glob so `zerx/backends/cerebras_dev.py` is never shipped (correct —
  AGENTS.md requires it). `model_backend.py` nonetheless imported it unconditionally
  at module scope. The two correct-in-isolation decisions are mutually incompatible.
- **Impact / Kaggle impact:** total. `agent/my_agent.py` imports
  `zerx.model_backend`, so the agent never loads; the run scores 0.
- **Hidden-game impact:** n/a (fails before any game starts).
- **Fix implemented:** yes — import moved inside the `cerebras_dev` branch of
  `select_backend`. Preserves both the secret-hygiene exclusion and importability;
  the branch is unreachable on Kaggle anyway (`Config` rejects it).
- **Regression test:** `tests/test_kaggle_bundle_importable.py::test_bundled_zerx_package_imports_without_the_backends_subpackage`
  (materializes the real bundle, imports it in a subprocess so the repo's own
  `zerx` cannot satisfy it).
- **Remaining risk:** none for this path.

### ARC-AUDIT-002 — Notebook writes into a directory it never creates

- **Severity:** P0 · **Confidence:** CONFIRMED · **Category:** Kaggle packaging
- **File:** `scripts/build_notebook.py` (cell ordering)
- **Observed:** the first `%%writefile /tmp/zerx/__init__.py` cell fails on a fresh
  kernel with `FileNotFoundError`.
- **Evidence:** IPython 9.16.1 `OSMagics.writefile` body contains
  `with open(filename, mode, encoding='utf-8')` and **no** `makedirs`/`dirname` call
  (read from installed source). Confirmed the underlying semantics:
  `open('<missing dir>/x.py','w')` → `FileNotFoundError`.
- **Root cause:** missing directory-creation step; `/tmp/zerx` does not pre-exist.
- **Impact / Kaggle impact:** total, independent of 001.
- **Fix implemented:** yes — an `os.makedirs('/tmp/zerx', exist_ok=True)` cell is
  emitted before the write cells.
- **Regression test:** `..::test_notebook_creates_tmp_zerx_directory_before_writing_into_it`,
  plus `..::test_writefile_into_a_missing_directory_really_does_fail` which pins the
  assumption the fix rests on.
- **Remaining risk:** none.

### ARC-AUDIT-003 — The submission runs with no model at all

- **Severity:** P0 · **Confidence:** CONFIRMED · **Category:** strategy / packaging
- **Files:** `notebooks/kernel-metadata.json`, `scripts/build_notebook.py`,
  `zerx/config.py:45`
- **Observed:** after fixing 001/002, the bundle imports and
  `select_backend(Config())` returns `FakeModelBackend` — whose `generate()` raises
  on every call.
- **Evidence:** `model_sources: []` and `dataset_sources: []` in kernel metadata; no
  vLLM install cell; no `ZERX_BACKEND` anywhere in the generated notebook; the `.env`
  the run cell writes contains only ARC gateway vars. `select_backend("fake")`
  returns `FakeModelBackend()` with an empty response list *by design*.
- **Root cause:** the Gemma serving path exists only in the Colab notebook builder
  (`scripts/build_colab_notebook.py`); it was never extended to the Kaggle
  submission notebook. `ACCELERATOR = "t4"` (16 GB) also contradicts AGENTS.md's
  RTX Pro 6000 (48 GB) target for a 31B model.
- **Impact:** the agent plays heuristics-only — the entire Gemma thesis is absent
  from the scored artifact. Failure is **silent**: no exception, no log, just a low
  score.
- **Fix implemented:** **partially, deliberately.** Wiring vLLM + model weights into
  the Kaggle notebook is a Day-4 workstream and a strategy decision, not a localized
  bug fix — I did not silently rewrite the submission path. What I did fix is the
  *silence*: `select_backend` now logs at ERROR when `backend="fake"` on any
  non-`local` platform. I deliberately did **not** make it raise, because turning a
  misconfiguration into a hard crash mid-competition is worse than a degraded run.
- **Regression test:** `tests/test_audit_regressions.py::test_fake_backend_on_kaggle_platform_logs_an_error`
  and `..::test_fake_backend_on_local_platform_is_silent`.
- **Remaining risk:** **high and unresolved.** See §16 P0-3 and §19.

### ARC-AUDIT-004 — `_diff` crashes, or lies, on a grid-shape change

- **Severity:** P1 · **Confidence:** CONFIRMED · **Category:** correctness
- **File:** `zerx/transitions.py:_diff`
- **Observed:** `_diff(64×64, empty)` → `IndexError: tuple index out of range`;
  `_diff(empty, 64×64)` → `(0, None)` i.e. "nothing changed".
- **Evidence:** executed both directly against master's code.
- **Root cause:** loop bounds derive from `before` only, with no shape check.
  Both shapes occur in a real run: `FrameData.frame == []` on `NOT_PLAYED`/`GAME_OVER`.
- **Impact:** the IndexError propagates out of `TransitionLedger.finalize()` into
  `_choose_action_inner`, where `MyAgent.choose_action`'s catch-all swallows it and
  emits `_safe_fallback_action` — so an entire decision step is silently discarded
  and the transition record is lost. The reverse case makes **the first real frame of
  every game look like a dead action**, poisoning `DeadSignatureTracker` from step one.
- **Hidden-game impact:** every game, every reset.
- **Fix implemented:** yes — `_diff` now detects a shape change and reports it as a
  real change with no bbox. (A prior review had deliberately put a `_shapes_match`
  guard in `my_agent.py` instead of touching shared infrastructure; that guard only
  covered the exact-state block and left `finalize()` exposed. Fixing the root is
  correct and the local guard remains as defence in depth.)
- **Regression test:** three tests in `tests/test_audit_regressions.py`.

### ARC-AUDIT-005 — The benchmark's progress signal was discarded

- **Severity:** P1 · **Confidence:** CONFIRMED · **Category:** correctness / strategy
- **File:** `agent/my_agent.py:_to_game_frame` (master line 75)
- **Observed:** `GameFrame(score=0)` hardcoded; every `TransitionRecord.score_delta`
  is `0 - 0 = 0`.
- **Evidence:** `arcengine.enums.FrameData` **does** define
  `levels_completed: int = Field(0, ge=0, le=254)` and `win_levels: int`. The code
  comment claimed "FrameData has no `.score` field (only level-completion counters, a
  different concept)" — literally true about the field *name*, but in ARC-AGI-3
  level completion **is** the scored progress signal.
- **Root cause:** a defensible-sounding comment that encoded a wrong judgement, then
  went unchallenged.
- **Impact:** `TransitionRecord.effective` collapsed to `changed_pixels > 0`;
  `ExactStateMemory.level_delta` was permanently 0 (acknowledged in a stale code
  comment); `classify_transition`'s `LEVEL_BOUNDARY` branch was unreachable. **The
  agent could not tell "I completed a level" from "nothing happened."**
- **Fix implemented:** yes — `score=frame.levels_completed`.
- **Regression test:** `test_score_delta_reports_level_completion`,
  `test_to_game_frame_maps_levels_completed_onto_score`.
- **Remaining risk:** low. `win_levels` and `full_reset` remain unused — see §17.

### ARC-AUDIT-006 — Unbounded object table can blow the context window

- **Severity:** P1 · **Confidence:** CONFIRMED · **Category:** token budget
- **File:** `zerx/policy.py:build_prompt`
- **Observed:** a two-colour 64×64 checkerboard — an entirely legal frame — segments
  into **4096** single-cell objects and renders a **197,571-character (~49,392-token)**
  prompt.
- **Evidence:** executed against master's `build_prompt`.
- **Root cause:** click candidates were capped at 5; the object table was not capped
  at all.
- **Impact:** exceeds Gemma's context window → the call errors or silently truncates,
  most likely dropping the JSON schema instruction at the end of the prompt and
  producing unparseable output → fallback chain → wasted actions. Cost scales with
  visual noise, so it degrades exactly on the busiest, most interesting frames.
- **Fix implemented:** yes — capped at `_MAX_PROMPT_OBJECTS = 60`, **with the
  truncation disclosed to the model** ("+N more objects not listed") so it cannot
  falsely infer the board is empty. *Hypothesis:* bounds worst-case prompt size by
  ~97% on pathological frames with no loss on normal ones, since the ranked candidate
  list — not this table — is what the model acts on.
- **Regression test:** `test_prompt_object_table_is_bounded_on_a_pathological_frame`,
  `test_prompt_lists_every_object_when_under_the_cap`.
- **Remaining risk:** 60 is a judgement call, not a calibrated value. Ablatable.

### ARC-AUDIT-007 — Concurrent agent threads share mutable `GameAction` singletons

- **Severity:** P1 · **Confidence:** HIGH (mechanism confirmed; impact not yet
  measured on a live run) · **Category:** concurrency
- **Files:** `agent/my_agent.py:_to_game_action`; upstream `agents/swarm.py:76-95`,
  `arcengine.enums.GameAction.set_data`
- **Observed:** the Kaggle run cell executes `python main.py --agent myagent` with no
  `--game`. `main.py`: "If none specified, an agent swarm will play all available
  games." `Swarm.main()` builds one agent per game and starts them all as concurrent
  `Thread`s.
- **Evidence:** `GameAction.set_data` is `self.action_data = self.action_type(**data)`
  — mutating a process-wide enum singleton. `_to_game_action` mutates
  `GameAction.ACTION6.action_data` and `.reasoning`, returns the shared member, and
  the framework only reads `action.action_data` later, in `do_action_request`.
- **Root cause:** upstream architecture (their own `Random` agent does the same); we
  inherit it and add `.reasoning`.
- **Impact:** between our mutation and the framework's read, another game's thread can
  overwrite the coordinates — **game A can submit game B's click**. Corrupts ACTION6,
  the most information-rich action, across all games simultaneously, and pollutes
  `TransitionLedger`/`ExactStateMemory` with actions that were never actually sent.
- **Fix implemented:** **no — deliberately.** This cannot be robustly fixed from
  inside `my_agent.py`: we do not control the window between `choose_action`
  returning and `take_action` reading, so no lock we can hold covers it. The real
  options are (a) patch the bundled framework copy, or (b) run one game per process.
  Both are architectural decisions with runtime/scheduling consequences that belong to
  the owner, not to an audit. See §17 EXP-R1.
- **Remaining risk:** high, and it is invisible in single-game local testing —
  `make verify-local` uses `--game`, so this never reproduces in the dev loop.

### ARC-AUDIT-008 — Reflection memory is permanently empty (`memory_on` is inert)

- **Severity:** P1 · **Confidence:** CONFIRMED · **Category:** dead feature / experiment integrity
- **File:** `zerx/policy.py:213-219`
- **Evidence:** `decide()` calls `maybe_refresh(..., summarizer=lambda prev, ctx: prev, ...)`
  — an explicit no-op, commented "deterministic no-op for the local skeleton".
  `MemoryState.summary` therefore never becomes non-empty; the prompt permanently
  reads `What you've learned so far: (nothing yet)`.
- **Impact:** `memory_on` is **True by default** and controls nothing. It costs prompt
  tokens for a constant string and makes any memory ablation meaningless.
- **Fix implemented:** no — supplying a real summarizer is a model-call design
  decision (cost, cadence, latency accounting) explicitly in `baseline-130` scope.
- **Remaining risk:** medium. Chief danger is *believing* memory is on.

### ARC-AUDIT-009 — Structured memory is computed, then thrown away

- **Severity:** P1 · **Confidence:** CONFIRMED · **Category:** dead feature / performance
- **Files:** `agent/my_agent.py:262-270`, `zerx/memory.py:207` (`render_for_prompt`)
- **Evidence:** `render_for_prompt` has **zero** production callers;
  `build_prompt` has no structured-memory parameter. `my_agent` calls
  `maybe_refresh_structured(...)` with the same no-op summarizer and stores the result.
- **Impact:** worse than dead code — when `structured_memory_on` is enabled it runs a
  **full `perceive(frame)` flood-fill every single action** purely to feed a no-op
  whose output is never read.
- **Fix implemented:** no — wiring it into the prompt is the substance of
  `baseline-130`, needs a hypothesis and an ablation, and would change model input.
- **Remaining risk:** medium (pure wasted compute today).

### ARC-AUDIT-010 — `arbiter_on` can never activate

- **Severity:** P1 · **Confidence:** CONFIRMED · **Category:** dead flag
- **Files:** `zerx/policy.py:236`, `zerx/candidates.py:130`
- **Evidence:** `select_candidate` gates on `config.arbiter_on and arbiter is not None`;
  `decide()` calls `select_candidate(model_candidates, config)` — **never passing an
  arbiter**. The condition is unsatisfiable in production.
- **Impact:** an `arbiter_on` A/B is guaranteed to show zero difference. `candidates.py`
  is honest that the arbiter is speculative, but the flag reaching the ablation matrix
  makes it look testable when it is not.
- **Fix implemented:** no — either wire an arbiter backend or drop the flag; that is a
  research decision.

### ARC-AUDIT-011 — `duck_objects_on` controls nothing but is in the ablation matrix

- **Severity:** P2 · **Confidence:** CONFIRMED · **Category:** experiment integrity
- **Files:** `zerx/config.py:48`, `eval/run_ablation.py:39`
- **Evidence:** grep — `duck_objects_on` appears only in `config.py`, its tests, and
  the ablation env map. Nothing in `zerx/` or `agent/` reads it. `zerx/scene.py` is
  unwired **by design** (category G, per its docstring and the parallel-day3 README's
  additive-only etiquette).
- **Impact:** the unwired module is fine; the **ablation entry is not**. Toggling it
  produces a confidently-reported null result.
- **Fix implemented:** no (one-line removal, but it is the owner's call whether the
  next step is "remove the flag" or "wire the feature").

### ARC-AUDIT-012 — Cross-game state leakage: **verified NOT a defect**

- **Severity:** — · **Confidence:** CONFIRMED · **Category:** false alarm, documented
- **Evidence:** no `.reset()` on `MemoryState` / `DeadSignatureTracker` /
  `TransitionLedger` / `ExactStateMemory` is called anywhere in production, despite all
  four docstrings insisting state "must never leak into the next game". However,
  `Swarm.main()` constructs **a fresh agent per game**, so `MyAgent.__init__` gives
  each game its own state.
- **Conclusion:** correct today, but **only by accident of the framework's lifecycle**.
  The `reset()` methods are untriggered safety equipment. Recorded so a future change to
  agent reuse (or intra-game level transitions) does not silently reintroduce leakage.

### ARC-AUDIT-013 — `history` is threaded everywhere and used nowhere

- **Severity:** P2 · **Confidence:** CONFIRMED · **Category:** strategic gap / performance
- **Files:** `agent/my_agent.py:210`, `zerx/perception.py:75-83`
- **Evidence:** `my_agent` builds `tuple(_to_game_frame(f) for f in frames[-4:])` each
  step and passes it to `decide()` → `perceive(frame, history)`, whose docstring says
  history "is accepted for interface stability … but the baseline only looks at `frame`".
- **Impact:** four full frame conversions per action, discarded. More importantly the
  agent has **no temporal/delta perception** — it cannot see what its last action
  changed, which is the core evidence loop ARC-AGI-3 rewards. `zerx/scene.py`'s
  `compare_frames` exists and would supply exactly this, unwired (see 011).
- **Fix implemented:** no — this is `exp-150`/world-model scope.

### ARC-AUDIT-014 — Redundant perception per step

- **Severity:** P2 · **Confidence:** CONFIRMED · **Category:** performance
- **Evidence:** per action the code may run `perceive()` up to three times over the same
  or adjacent frames (`decide()`; `_find_object_by_label`; the structured-memory block),
  plus 4 `_to_game_frame` conversions for the unused history.
- **Impact:** CPU only, no GPU cost — modest next to inference latency, but it is pure
  waste and grows with grid busyness. Not quadratic; no unbounded cache found.

### ARC-AUDIT-015 — Accelerator/model mismatch

- **Severity:** P2 · **Confidence:** HIGH · **Category:** Kaggle config
- **Evidence:** `build_notebook.ACCELERATOR = "t4"` (2×16 GB); AGENTS.md targets RTX
  Pro 6000 48 GB, and `build_colab_notebook.py` itself notes 31B bf16 ≈ 61 GB.
- **Impact:** moot while finding 003 stands (no model is loaded), but it will silently
  become the blocker the moment the model is wired.

### ARC-AUDIT-016 — `Config.from_env` raises on malformed input

- **Severity:** P3 · **Confidence:** CONFIRMED · **File:** `zerx/config.py:21-28`
- **Evidence:** `_env_int`/`_env_float` call bare `int()`/`float()`. `ZERX_BUDGET_SOFT_CAP=abc`
  raises `ValueError` inside `MyAgent.__init__`, which is **outside** `choose_action`'s
  catch-all → the agent fails to construct and the whole game aborts.
- **Impact:** low in practice (we set our own env), but it is a crash on a typo.

### ARC-AUDIT-017 — Cerebras lockout implements one of three required conditions

- **Severity:** P3 · **Confidence:** CONFIRMED · **File:** `zerx/config.py:54`
- **Evidence:** AGENTS.md requires rejection "whenever `platform=kaggle`, competition
  mode is active, or internet is disabled". Only the first is implemented.
- **Mitigation already present:** defence in depth — `CerebrasDevBackend.__init__` has
  its own guard, and the build never ships `zerx/backends/`. Residual risk low.

### ARC-AUDIT-018 — Secret scanner covers only two literal patterns

- **Severity:** P3 · **Confidence:** CONFIRMED · **File:** `zerx/secret_scan.py`
- **Evidence:** only `api.cerebras.ai` and `CEREBRAS_API_KEY`. No generic detection of
  key-shaped strings (`sk-…`, bearer tokens, Kaggle tokens).
- **Impact:** a leaked *value* without its variable name would pass. `.kaggle/` is
  correctly gitignored; **no secrets are tracked in git** (verified).

### ARC-AUDIT-019 — Unregistered pytest marks

- **Severity:** P4 · **Confidence:** CONFIRMED
- **Evidence:** `PytestUnknownMarkWarning: Unknown pytest.mark.slow_local_engine`. There
  is no `pytest.ini`/`pyproject.toml` in the repo root, so `cerebras_live` and
  `slow_local_engine` are unregistered and **`-m` filtering silently depends on
  convention**. A typo'd mark name would silently select nothing.

### ARC-AUDIT-020 — Root `pytest` invocation is broken

- **Severity:** P3 · **Confidence:** CONFIRMED
- **Evidence:** bare `pytest` from the repo root fails collection:
  `ERROR vendor/ARC-AGI-3-Agents/tests - ModuleNotFoundError: No module named 'tests.conftest'`
  — it tries to collect the *vendored* framework's test suite. Everyone must know to run
  `pytest tests`. AGENTS.md's documented command is `uv run pytest -q`, which would fail.
  A `testpaths = tests` setting would fix both this and 019.

---

## 6. Kaggle Submission Readiness

| Check | Master | This branch |
|---|---|---|
| Offline execution (no network at eval) | **PASS** — `zerx/` reaches the network only via localhost vLLM; `backends/` never shipped | PASS |
| Packaging — directory creation | **FAIL** (002) | **PASS** |
| Packaging — imports | **FAIL** (001) | **PASS** (verified by subprocess import of the real bundle) |
| Model assets attached | **FAIL** (003) — `model_sources: []` | **FAIL** (unchanged; reported) |
| Python 3.12 compatibility | PASS — `list[str]`/`X | Y` syntax used consistently; `from __future__ import annotations` throughout | PASS |
| GPU/CPU assumptions | **WARN** (015) — t4 vs 48 GB target | WARN |
| Paths | PASS — no Mac-specific or absolute-local paths in shipped code | PASS |
| Timeout / resource risk | **WARN** (006 context blowout) | improved |
| Output artifacts | PASS — dummy `submission.parquet` on commit; gateway emits the real one on rerun | PASS |
| Secrets in artifact | PASS — build-time scan gate, verified to fire on a planted key | PASS |

**Verdict: master is NOT submission-ready. This branch fixes the two mechanical
blockers; the missing model (003) still makes a submission near-pointless for score.**

No Kaggle submission was made, no notebook was pushed, no quota consumed. Packaging
was verified **locally only**, via `scripts/build_notebook.py` + bundle
reconstruction — never `make submit`.

---

## 7. ARC Prize Compliance / License Review

I did not have verified live access to the current official rules during this audit, so
every item below is classified conservatively and **none of these are legal conclusions**.

| Item | Status | Evidence |
|---|---|---|
| Repo LICENSE | **NEEDS VERIFICATION** | **There is no `LICENSE` file in `git ls-files`.** Prize eligibility has open-source requirements; this should be resolved before any prize-eligible submission. |
| Vendored framework | LIKELY OK | `vendor/ARC-AGI-3-Agents` is gitignored — we redistribute nothing; Kaggle copies it from the competition dataset. Upstream ships its own LICENSE. |
| Model weights | NEEDS VERIFICATION | `model_backend.py` cites Kaggle handle `google/gemma-4/Transformers/gemma-4-31b-it`, "Apache 2.0". Gemma is normally under the **Gemma Terms of Use**, not Apache 2.0 — this comment should be re-checked against the actual model card. |
| Cerebras (dev-only) | CLEAR | Never in the Kaggle runtime; excluded from the bundle; build-time scan gate; no key in git. |
| Copied source | CLEAR | No copied prior-art implementation found. `STRATEGY.md` cites ReKi/Tycho/FORGE/Duck as *inspiration* with commit-pinned links; the implementations are original. |
| Tracked secrets | CLEAR | `.kaggle/` gitignored; no tokens/keys in tracked files. |
| Competition-integrity rules | CLEAR | No hidden-state access, no engine-source reading, no game-ID branching, no unbounded search (see §13). |

## 8. Prompt / Policy Review

The single prompt is `zerx/policy.py:build_prompt` (plus a small arbiter prompt in
`candidates.py`, currently unreachable — 010). Assessed against §8 of the brief:

**Fixed this session (with hypotheses):**
- Missing legal-action list (`e402a0d`) — *reduces invalid actions and wasted
  repair/fallback cycles*. AGENTS.md control-flow step 2 required it.
- Missing budget signal (`e402a0d`) — *lets the model shift probing→execution*, AGENTS.md
  step 7.
- Unbounded object table (006) — *bounds worst-case prompt from ~49k tokens to ~2k*.

**Correctly resisted:**
- The prompt says only "a grid-based puzzle game" and never explains ACTION1–5. This
  is **right**, not a weakness: AGENTS.md forbids hard-coding their semantics because
  they vary per game. An earlier instinct to "improve" this was withdrawn.

**Genuine remaining weaknesses (not changed — each needs a hypothesis + ablation):**
- **No state delta.** The model sees the current grid but never "what your last action
  changed". This is the highest-value prompt improvement available and the machinery
  already exists unwired (`scene.compare_frames`, 013).
- **No fact/hypothesis separation reaches the model** (009) — `render_for_prompt` builds
  exactly this and is never called.
- **`memory.summary` is a permanent constant** (008).
- Output schema is sound: strict JSON, one action, `data` only for ACTION6, and it now
  closes by re-binding the model to the legal set.

## 9. Memory / Context Review

- **Bounded, no unbounded growth in the live path:** `TransitionLedger._recent_hashes`
  is a `deque(maxlen=20)`; history is `frames[-4:]`; candidates capped at 5; objects now
  capped at 60.
- **`ExactStateMemory._records` grows without bound** within a game (one entry per
  distinct `(state_hash, action_sig)`). Bounded in practice by the framework's
  `MAX_ACTIONS`, so not a live leak — noted, not filed.
- **`StructuredMemoryState.notable_failures` is explicitly never deduped** and would grow
  unboundedly *if* it were ever populated. Inert today (009).
- **The FACT / HYPOTHESIS / REJECTED / GOAL / PLAN hierarchy the brief asks about exists
  and is well-built** in `zerx/memory.py` — including correct belief-reversal semantics in
  `contradict_hypothesis`. It is simply never connected to anything.

## 10. Exploration / World Model Review

Present and working: graded negative affordances (`DeadSignatureTracker`, correctly soft
rather than a hard ban), exact `(state, action)` no-op suppression (baseline-115, off by
default), repeated-state detection, evidence-first transition pairing.

Absent: novelty-driven target selection, unexplored-region tracking, causal probing,
action-result association reaching the model, hypothesis falsification in the live loop.
`zerx/scene.py` implements object correspondence and a full change taxonomy
(`OBJECT_MOVE` / `RECOLOR_OR_TRANSFORM` / `HUD_ONLY` / `LEVEL_BOUNDARY` …) — all unwired.

**Net:** the agent reacts to the current frame; it does not yet learn mechanics. Before
005 it could not even perceive progress. This — not any individual bug — is the
strategic ceiling.

## 11. Legal Action / Parser Review

**This layer is the strongest part of the codebase.** The four-stage separation the brief
asks for is real and correctly ordered: model proposal → `parse_action` → `_validate_payload`
(name in enum, name ∈ legal set, ACTION6 x/y present and int) → `Action.__post_init__`
(0–63 bounds, rejects x/y on non-ACTION6) → fallback chain.

Verified by existing tests plus inspection: invalid JSON, prose-wrapped JSON, markdown
fences, wrong action name, illegal-but-valid action, missing x/y, non-int x/y,
out-of-bounds and negative coordinates are all rejected. Repair is bounded to exactly one
deterministic extraction — no second reasoning loop.

Fallback order matches AGENTS.md: model → heuristic → deterministic → random-from-legal →
RESET. Random is genuinely last-resort and samples only from the legal set. `choose_action`
cannot raise (outer boundary + independent `_safe_fallback_action`).

One nuance worth stating: because the default backend is `fake` (003), **every step
currently exercises the fallback chain**, which is why local runs look stable — the
robust part is carrying the whole system.

## 12. Compute & Runtime Budget

- **No model, so no GPU load today** (003): startup ≈ import time, no weights, no VRAM.
- **Per action, model-free:** up to 3 `perceive()` flood-fills + 4 frame conversions +
  1 grid hash + 1 diff. All O(H·W) = O(4096). Negligible against inference.
- **No quadratic behaviour found:** no re-rendering of all history, no re-segmentation of
  unchanged regions, no model/tokenizer reloading, no unbounded cache, no duplicate
  inference. `scene.py`'s `_find_children` explicitly bounds its flood-fill to the parent
  bbox to avoid O(objects × grid_area).
- **Token budget was the real risk** (006): ~49k tokens worst case, now ~2k.
- **Retry budget:** `GemmaModelBackend.max_retries=2` with **no backoff** — two immediate
  attempts. Fine for a local sidecar; would be wrong against a rate-limited remote.
- **`candidate_count > 1` multiplies model calls linearly**; correctly defaults to 1.

## 13. Public-Game Overfitting Risks

**This is the repository's strongest result, and it should be stated plainly: production
code is clean.**

- `grep` for game IDs (`ls20`, `vc33`, …) across `zerx/` and `agent/` → **zero hits**.
  Occurrences exist only in `scripts/play_local.py` (dev CLI help) and
  `scripts/build_colab_notebook.py` (`GAME_SAMPLE`, an eval-only list). Neither ships.
- No magic coordinates, no hard-coded colours, no game-specific transition rules, no
  score hacks, no lookup tables.
- Heuristic constants classified: `penalty_step=0.35`, `recovery_step=0.5`,
  `0.5·size + 0.5·rarity`, `_HUD_MAX_AREA=4`, `_MIN_MATCH_CONFIDENCE=0.5` — all
  **category A/B (general inductive bias)**: "small, rare-coloured things are clickable"
  is a domain prior, not a public-game memorization. None are tuned against specific games.
- `heuristic_first` is correctly **off by default**, exactly as AGENTS.md requires until
  calibrated.
- `_MAX_PROMPT_OBJECTS = 60` (added here) is a new uncalibrated constant — flagged as
  ablatable rather than presented as tuned.

## 14. Test Coverage Gaps

287 → **302 tests, all passing.** The suite is genuinely good on units. Its systematic
blind spot is **outcomes vs mechanisms**:

- `test_build_notebook.py`'s own docstring says a real Kaggle run "would fail with
  `ModuleNotFoundError`" — and it tested that `backends/` is excluded and that files are
  written, but **never that the result imports**. Both P0s lived directly under this test
  file. Now closed by `test_kaggle_bundle_importable.py`.
- **No offline/packaging integration test** existed. Added.
- **No malformed-shape transition test** existed (004). Added.
- **No token/context-size bound test** existed (006). Added.
- Still missing (recommended, not added): a long-horizon test (>200 steps) asserting
  bounded memory and no action loops; a multi-threaded test reproducing 007; a
  no-legal-actions test; a `full_reset` / level-transition test.
- Two marks are unregistered and the root `pytest` invocation is broken (019, 020).

## 15. Dead Code / Technical Debt

| Item | Lines | Status |
|---|---|---|
| `zerx/scene.py` | 549 | Unwired **by design** (category G). Fully tested. Highest-value unlock in the repo. |
| `memory.render_for_prompt` + 7 structured-memory mutators | ~140 | Implemented, tested, **zero production callers** (009) |
| `candidates._build_arbiter_prompt` / `_select_with_arbiter` | ~25 | Unreachable (010) |
| 4× `reset()` methods | ~25 | Never called; currently redundant (012) |
| `history` parameter chain | — | Threaded through 3 layers, used by none (013) |
| `GameFrame.score` | — | Was inert; **now live** (005) |

No duplicate algorithms, no competing state representations, no shadowed functions, no
merge-conflict artifacts, no TODO/FIXME/HACK/XXX anywhere. Notably, `heuristics.size_rarity_scores`
was correctly factored out so `scene.list_salient_objects` reuses one formula rather than
duplicating it — the codebase is disciplined about single sources of truth.

## 16. Ranked Fix List

**P0**
1. ~~`zerx.backends` import breaks the bundle~~ — **FIXED** (001)
2. ~~`/tmp/zerx` never created~~ — **FIXED** (002)
3. **Wire a real model into the Kaggle notebook** — attach Gemma weights as a Kaggle model
   source, add a vLLM serving cell, set `ZERX_BACKEND=gemma_kaggle` + `ZERX_PLATFORM=kaggle`,
   switch `ACCELERATOR` to `rtx6000`. **Not fixed — owner decision, Day-4 workstream** (003, 015)

**P1**
4. ~~`_diff` shape crash / false no-change~~ — **FIXED** (004)
5. ~~`levels_completed` discarded~~ — **FIXED** (005)
6. ~~Unbounded prompt object table~~ — **FIXED** (006)
7. `GameAction` singleton race across concurrent game threads — **not fixed, architectural** (007)
8. `memory_on` inert — supply a real summarizer or stop defaulting it on (008)
9. `structured_memory_on` computes and discards; costs a `perceive()` per action (009)
10. `arbiter_on` unsatisfiable (010)

**P2**
11. `duck_objects_on` in the ablation matrix while controlling nothing (011)
12. `history` unused → no temporal perception (013)
13. Redundant `perceive()` calls per step (014)
14. Add `LICENSE` (§7) — eligibility-relevant

**P3**
15. `Config.from_env` crashes on malformed env values (016)
16. Cerebras guard covers 1 of 3 required conditions (017)
17. Secret scanner is literal-pattern-only (018)
18. Root `pytest` collection broken; add `testpaths`/`markers` (019, 020)
19. Re-verify the Gemma licence comment (§7)

**P4**
20. `MemoryState`/`ExactStateMemory` reset methods are untriggered safety equipment (012)

## 17. Ranked Experiment Backlog

| ID | Category | Hypothesis | Targets failure | Metric | Cost | Conf | Pri |
|---|---|---|---|---|---|---|---|
| **EXP-K1** | KAGGLE PACKAGING | Attaching Gemma + vLLM turns a heuristic-only run into a reasoning run | 003 | RHAE vs heuristic-only baseline | High (Kaggle hrs) | HIGH | **1** |
| **EXP-R1** | ROBUSTNESS | One game per process (or a patched framework copy) eliminates cross-game action corruption | 007 | mismatch rate between intended and recorded actions | Low | HIGH | **2** |
| **EXP-P1** | PROMPT | Showing the last transition's delta ("your ACTION6 at (x,y) moved obj3 left") cuts repeated dead actions | 013 | repeat-action rate, actions/level | Low | HIGH | **3** |
| **EXP-M1** | MEMORY | A real summarizer at interval 10 beats the constant-empty summary | 008 | levels completed; latency delta | Med | MEDIUM | 4 |
| **EXP-M2** | MEMORY | Rendering structured memory (fact/hypothesis/rejected) beats free-text | 009 | belief-reversal rate, RHAE | Med | MEDIUM | 5 |
| **EXP-W1** | WORLD MODEL | Wiring `scene.compare_frames` + `classify_transition` lets `effective` distinguish HUD-only animation from real change | STRATEGY §5.4's stated limitation | false-effective rate | Med | MEDIUM | 6 |
| **EXP-A1** | ACTION POLICY | Calibrate `heuristic_confidence_threshold` before enabling `heuristic_first` | uncalibrated gate | model-call savings vs quality loss | Low | MEDIUM | 7 |
| **EXP-C1** | PERFORMANCE | Cache `perceive()` per frame; drop unused history conversion | 013, 014 | wall-clock/action | Low | HIGH | 8 |
| **EXP-T1** | PROMPT | Ablate `_MAX_PROMPT_OBJECTS` ∈ {20, 60, 200} | 006's uncalibrated constant | RHAE, prompt tokens | Low | MEDIUM | 9 |

**Do not use public-game aggregate score alone as evidence any of these generalize**
(STRATEGY.md's Tycho-numbers caveat).

## 18. Strategy Roadmap Reconciliation

| Milestone | Classification | Evidence |
|---|---|---|
| Tasks 1–15 local skeleton | **DONE + VERIFIED** | 302 tests; all modules present |
| `baseline-000` starter verification | DONE + VERIFIED | experiment record + upstream API confirmed against vendored source |
| `baseline-100` Colab Gemma | **DONE BUT UNVERIFIED** on Kaggle | Colab-only; Kaggle path proven broken (001/002) |
| `baseline-110` transition ledger | DONE, **had a latent defect** | shipped; 004 fixed here |
| `baseline-115` exact-state memory | **DONE + VERIFIED**, off by default | wired, tested; its level channel was inert until 005 |
| `baseline-120` real-game validation | **PARTIAL** | four tracks merged; eval harness + regression tests exist; **the "real game" path still runs model-free** |
| `baseline-130` hypothesis memory | **PARTIAL / STALE** | data structures done and tested; never reaches the model (009) |
| `exp-140` VLM refinement | **PARTIAL** | candidates wired; arbiter unreachable (010) |
| `exp-150` Duck tools A+B | **DONE, NOT STARTED BY DESIGN (integration)** | `scene.py` complete + tested, intentionally unwired |
| `exp-150` C/D variants | NOT STARTED BY DESIGN | — |
| `exp-200+` Tycho world model | NOT STARTED BY DESIGN | correctly gated behind `baseline-130` |

**Assessment: the ladder is ahead of the foundation.** Three tracks of sophisticated,
well-tested machinery sit above a base that, until this branch, could not import on
Kaggle, could not perceive progress, and had no model attached. **Do not start
`baseline-125` or `exp-200` until EXP-K1 and EXP-R1 land.**

## 19. Recommended Next 5 Actions

1. **Attach Gemma + vLLM to the Kaggle notebook and set `ZERX_BACKEND`/`ZERX_PLATFORM`;
   switch `ACCELERATOR` to `rtx6000`.** Nothing else on this list changes the score while
   the submission runs model-free. (003/015, EXP-K1)
2. **Decide and implement the concurrency mitigation for 007** — simplest credible option
   is one game per process. This is invisible in local single-game testing and can corrupt
   every ACTION6 on Kaggle.
3. **Re-run the Day-3 packaging path end-to-end on Kaggle as a smoke test** now that
   001/002 are fixed — *before* spending the intended final submission slot. AGENTS.md's
   own schedule says the last safe start for a first real run was Day 3; it is Day 5.
4. **Resolve the two experiment-integrity issues before any further A/B**: either wire or
   remove `arbiter_on` and `duck_objects_on`, and stop defaulting `memory_on=True` while it
   is a no-op. Otherwise the next ablation produces confident null results.
5. **Add a `LICENSE` file** and re-verify the Gemma weight-licence claim in
   `model_backend.py` against the actual model card — eligibility-relevant, cheap, and
   currently unresolved.

---

## Appendix A — Fix Log (before/after)

Every change below is on `feat/policy-prompt-legal-budget` only.

**FIX-1 — ARC-AUDIT-001**
- *Root cause:* module-level `from zerx.backends.cerebras_dev import CerebrasDevBackend`
  in a module that ships without `backends/`.
- *Files:* `zerx/model_backend.py`
- *Before:* `import zerx.model_backend` from the bundle → `ModuleNotFoundError`.
- *Change:* import moved inside the `cerebras_dev` branch of `select_backend`.
- *Test:* `test_bundled_zerx_package_imports_without_the_backends_subpackage` (subprocess),
  `test_selecting_cerebras_backend_still_works_when_backends_is_present`.
- *After:* bundle imports; `select_backend(Config(backend="cerebras_dev"))` still returns
  `CerebrasDevBackend`.
- *Risk:* one existing test monkeypatched the now-absent module attribute; retargeted to the
  defining module, assertion strength unchanged.
- *Impact:* submission goes from "cannot start" to "starts".

**FIX-2 — ARC-AUDIT-002**
- *Root cause:* `%%writefile` does not create parent directories.
- *Files:* `scripts/build_notebook.py`
- *Before:* first zerx write cell → `FileNotFoundError` on a fresh kernel.
- *Change:* emit an `os.makedirs('/tmp/zerx', exist_ok=True)` cell before the write cells.
- *Test:* ordering test + a test pinning the `open()`-into-missing-dir assumption.
- *Risk:* none.

**FIX-3 — ARC-AUDIT-004**
- *Files:* `zerx/transitions.py`
- *Before:* `IndexError` (64×64→empty) / false `(0, None)` (empty→64×64).
- *Change:* shape check; a shape change reports "everything changed", bbox `None`.
- *Test:* 3 tests incl. ledger-level `effective`.
- *Risk:* low — `changed_pixels` on a shape change is now a magnitude, not a cell count;
  only `> 0` is consumed.

**FIX-4 — ARC-AUDIT-005**
- *Files:* `agent/my_agent.py`
- *Before:* `score=0` hardcoded → every `score_delta` 0.
- *Change:* `score=frame.levels_completed`; stale comments corrected.
- *Test:* `test_score_delta_reports_level_completion`, `test_to_game_frame_maps_levels_completed_onto_score`.
- *Risk:* low. Turns on a previously-dead channel in `ExactStateMemory`; that feature is
  off by default, so exposure is bounded.
- *Impact:* the agent can perceive progress for the first time.

**FIX-5 — ARC-AUDIT-006**
- *Files:* `zerx/policy.py`
- *Before:* 4096 objects → 197,571 chars (~49,392 tokens).
- *Change:* cap at 60, disclose truncation to the model.
- *Test:* pathological-frame bound + under-cap completeness.
- *Risk:* the model no longer sees every object on very busy frames; mitigated because the
  ranked candidate list is what it acts on, and truncation is stated explicitly.

**FIX-6 — ARC-AUDIT-003 (partial)**
- *Files:* `zerx/model_backend.py`
- *Change:* ERROR log when `backend="fake"` on a non-`local` platform. Deliberately does
  **not** raise.
- *Test:* logs-on-kaggle / silent-on-local.
- *Remaining:* the actual model wiring — owner decision.
