# baseline-120 Colab validation, experiment record, and status docs — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `scripts/build_colab_notebook.py`'s one-game smoke-test notebook
into `baseline-120-reki-core`'s real multi-game validation run (real
per-game outcome + RHAE capture), run the cheap `cerebras_dev` dev-lane
sweep through the real harness once Track 1's backend factory is
reachable, and write `docs/superpowers/experiments/baseline-120.md` with
an honest `keep`/`revert`/`investigate` conclusion decided from the
**Colab Gemma** result, not the Cerebras proxy.

**Architecture:** No new modules. `scripts/build_colab_notebook.py` gains
two new module constants (`GAME_SAMPLE`, `MAX_STEPS_PER_GAME`) and its
`smoke_game_cell`/`save_results_cell` generators are rewritten:
`smoke_game_cell` stops shelling out to `scripts/play_local.py` (a child
process's in-memory `Arcade`/scorecard state is unreachable from a later
notebook cell) and instead drives `MyAgent` directly, in-process, once per
sampled game, so `save_results_cell` — running in the **same** kernel —
can call `arc.get_scorecard()` and read real per-game RHAE. A standalone,
never-committed scratch script exercises `CerebrasDevBackend` +
`zerx.policy.build_prompt`/`parse_action` in isolation. `eval/run_ablation.py`
(Track 2) and `zerx/model_backend.py`'s `select_backend` (Track 1) are
read-only dependencies of the gated Part B tasks, never edited here.

**Tech Stack:** Python 3.12, pytest, the `arc-agi` PyPI package
(`arc_agi.Arcade`, `arc_agi.scorecard.EnvironmentScorecard`/`EnvironmentScoreList`),
the vendored `ARC-AGI-3-Agents` framework's `agents.agent.Agent`, Jupyter
notebook JSON (no live execution locally — this repo has no GPU).

## Global Constraints

- Base master commit for this branch: `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb`.
  Branch: `feat/baseline-120-colab-validation` (already created and checked
  out from that exact SHA).
- Full local suite must stay green throughout: **261 passed, 0 failed**
  confirmed on this exact commit before any change in this plan
  (`.venv/Scripts/pytest.exe tests/ -q`, Windows-native venv layout per
  `docs/superpowers/experiments/baseline-000.md`'s "Windows-native
  environment deviations").
- Touch **only**: `scripts/build_colab_notebook.py`,
  `tests/test_build_colab_notebook.py`,
  `docs/superpowers/experiments/baseline-120.md` (new file),
  `docs/HANDOFF.md`, and this plan file. Do **not** touch
  `scripts/build_notebook.py` (the Kaggle notebook builder — a different
  file), `zerx/model_backend.py`, `agent/my_agent.py`, `zerx/config.py`,
  `eval/run_ablation.py`, `scripts/play_local.py`, `STRATEGY.md`, or
  `notebooks/kernel-metadata.json`.
- Never put `CEREBRAS_API_KEY` in code, commits, notebook cells, logs, or
  this plan/experiment doc's contents — only ever read from the
  executor's own shell environment at run time (`AGENTS.md`'s Cerebras
  development boundary).
- Do not merge to `master`. Push only to
  `origin/feat/baseline-120-colab-validation` when explicitly asked.
- Regenerate `notebooks/colab_gemma_smoke.ipynb`
  (`.venv/Scripts/python.exe scripts/build_colab_notebook.py`) **after**
  committing a generator-script change, never before/during the same
  commit — otherwise the embedded `COMMIT_SHA` goes stale
  (`docs/HANDOFF.md`'s recorded Day-2 gotcha). The notebook file itself is
  gitignored; regeneration is a verification step, not something to commit.
- The actual Colab GPU run (Part B step 6) requires a human to upload and
  run the generated notebook in a real Colab browser session attached to
  an A100/L4 runtime — no tool available in this session can drive that.
  Prepare everything needed for it; do not fabricate its result.

---

## Task 1: Multi-game sample + in-process `smoke_game_cell`

**Files:**
- Modify: `scripts/build_colab_notebook.py` (add `GAME_SAMPLE`/`MAX_STEPS_PER_GAME`
  module constants after `PINNED_INSTALL`; rewrite `smoke_game_cell`'s body
  inside `build()`; small `intro_cell` text update)
- Test: `tests/test_build_colab_notebook.py`

**Interfaces:**
- Consumes: nothing new — this task only edits notebook-generator text.
- Produces: `build_colab_notebook.GAME_SAMPLE: List[str]` (8 game ids:
  `["ls20", "vc33", "su15", "tn36", "ka59", "lf52", "tr87", "sc25"]`),
  `build_colab_notebook.MAX_STEPS_PER_GAME: int` (`100`). Task 2 reads
  both by name (they appear verbatim in the generated `smoke_game_cell`
  source, and Task 2's `save_results_cell` source references the same two
  names plus `per_game_play_results`, a variable this task's generated
  cell defines — not a Python object either task's *generator* touches at
  import time, only a name both cells' *generated source text* share).

- [ ] **Step 1: Write the failing tests**

Open `tests/test_build_colab_notebook.py`. Delete this now-superseded test
(the cell it checks is being rewritten to no longer shell out to
`play_local.py` — see Task 1's rationale in the Architecture section
above):

```python
def test_build_runs_one_public_game_via_play_local():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "play_local.py" in combined
```

Add these in its place (anywhere after the existing tests, before
`test_main_writes_notebook_file`):

```python
def test_game_sample_includes_the_existing_ls20_vc33_precedent():
    """docs/superpowers/plans/parallel-baseline-120/README.md's own
    'concrete, empirical finding' section measured baseline-120's
    fallback-only reference (0.0 aggregate score, 0 levels completed,
    all-ACTION6) by running ls20+vc33 -- the Colab game sample must
    include both so this track's real-model result is comparable to that
    measured reference, not a disjoint game set.
    """
    assert "ls20" in build_colab_notebook.GAME_SAMPLE
    assert "vc33" in build_colab_notebook.GAME_SAMPLE


def test_game_sample_is_larger_than_baseline_100s_single_game_sample():
    """baseline-100.md's own conclusion ('investigate', not 'keep') was
    partly because only one game (ls20) was ever played. AGENTS.md's
    'repeated seeds/configurations' and 'per-game regressions' language
    argues for more than that before this rung can be promoted.
    """
    assert len(build_colab_notebook.GAME_SAMPLE) >= 6


def test_smoke_game_cell_plays_every_sampled_game_directly_via_myagent():
    """Replaces the old subprocess call to scripts/play_local.py: capturing
    real per-game RHAE requires arc.get_scorecard() to be queried in the
    SAME Python process that played the games (a child process's Arcade/
    scorecard state is unreachable from a later notebook cell), so this
    cell now drives MyAgent directly instead of shelling out.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "arc_agi.Arcade" in combined
    assert "agent.main()" in combined
    for game_id in build_colab_notebook.GAME_SAMPLE:
        assert f'"{game_id}"' in combined


def test_smoke_game_cell_caps_steps_below_play_locals_default_for_colab_time_budget():
    """8 games x play_local.py's 200-step default risked exceeding a
    single Colab session at an unmeasured 31B per-decision latency -- see
    docs/superpowers/experiments/baseline-120.md's wall-clock estimate.
    """
    assert build_colab_notebook.MAX_STEPS_PER_GAME < 200
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "MAX_STEPS_PER_GAME" in combined


def test_smoke_game_cell_isolates_one_games_exception_from_the_rest():
    """A single game's unhandled exception must not lose the results
    already collected for earlier games in the sample -- each game is
    wrapped in its own try/except that records the failure and continues
    to the next game.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "except Exception as exc" in combined
    assert '"exception": repr(exc)' in combined


def test_smoke_game_cell_still_documents_the_gemma_backend_and_vllm_server():
    """Preserves the existing test_build_wires_gemma_model_backend_against_local_vllm_server
    guarantee under the new cell structure: a reader must still be able to
    see that ZERX_BACKEND=gemma_local resolves to GemmaModelBackend against
    the local vLLM server this notebook just started.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "GemmaModelBackend" in combined
    assert "localhost:8000" in combined
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_build_colab_notebook.py -v`

Expected: the 6 new tests FAIL (`AttributeError: module 'build_colab_notebook'
has no attribute 'GAME_SAMPLE'` for most; the deleted test is simply gone).
`test_build_wires_gemma_model_backend_against_local_vllm_server` (the
pre-existing test) still PASSES at this point — Step 3 hasn't touched
`smoke_game_cell` yet.

- [ ] **Step 3: Add the module constants**

In `scripts/build_colab_notebook.py`, immediately after the `PINNED_INSTALL`
`dedent(...)` block (after its closing `)`), add:

```python
# baseline-120-reki-core's real-game validation sample (see
# docs/superpowers/plans/parallel-baseline-120/README.md's "concrete,
# empirical finding" and docs/superpowers/experiments/baseline-120.md for
# the full game-list/wall-clock justification). Keeps the existing
# ls20+vc33 precedent (baseline-100's smoke game plus this plan's own
# measured "before" reference) and adds 6 more games spread across the
# documented 25-game public list, for per-game regression coverage per
# AGENTS.md's "repeated seeds/configurations" language.
GAME_SAMPLE = ["ls20", "vc33", "su15", "tn36", "ka59", "lf52", "tr87", "sc25"]
# Deliberately below play_local.py's 200-step default: 8 games x 200 steps
# risked exceeding a single Colab session at an unmeasured 31B
# per-decision latency -- see the experiment doc's wall-clock trade-off.
MAX_STEPS_PER_GAME = 100
```

- [ ] **Step 4: Rewrite `smoke_game_cell`**

Replace the existing `smoke_game_cell = code_cell(...)` assignment inside
`build()` with:

```python
    smoke_game_cell = code_cell(
        f'GAME_SAMPLE = {json.dumps(GAME_SAMPLE)}\n'
        f'MAX_STEPS_PER_GAME = {MAX_STEPS_PER_GAME}\n'
        + dedent(
            """
            import os
            import sys
            import time
            import importlib.util

            os.environ["ZERX_BACKEND"] = "gemma_local"
            os.environ["ZERX_PLATFORM"] = "colab"
            os.environ["ZERX_MODEL_REVISION"] = "gemma-4-31b-it"

            sys.path.insert(0, "")
            sys.path.insert(0, "vendor/ARC-AGI-3-Agents")

            import arc_agi
            from arc_agi import OperationMode

            def _load_my_agent_class():
                spec = importlib.util.spec_from_file_location(
                    "user_agent_module", "agent/my_agent.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.MyAgent

            # Loading agent/my_agent.py here constructs its backend via
            # ZERX_BACKEND=gemma_local -- once Track 1's select_backend
            # lands this resolves through it to
            # GemmaModelBackend(self._config.model_revision), pointed at
            # the vLLM server started above via its default base_url
            # (localhost:8000). Before Track 1 lands, agent/my_agent.py's
            # __init__ hardcodes GemmaModelBackend directly regardless of
            # Config.backend -- gemma_local already resolves correctly
            # either way (see docs/superpowers/plans/parallel-baseline-120/
            # README.md's "concrete, empirical finding").
            MyAgentCls = _load_my_agent_class()
            MyAgentCls.MAX_ACTIONS = min(
                getattr(MyAgentCls, "MAX_ACTIONS", MAX_STEPS_PER_GAME),
                MAX_STEPS_PER_GAME,
            )

            arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)

            per_game_play_results = []
            for i, game_id in enumerate(GAME_SAMPLE, 1):
                print(f"=== [{i}/{len(GAME_SAMPLE)}] {game_id} ===")
                start = time.monotonic()
                try:
                    env = arc.make(game_id)
                    agent = MyAgentCls(
                        card_id="colab-baseline-120",
                        game_id=game_id,
                        agent_name=f"MyAgent.colab.{game_id}",
                        ROOT_URL="http://localhost",
                        record=False,
                        arc_env=env,
                        tags=["colab", "baseline-120"],
                    )
                    agent.main()
                    final = agent.frames[-1]
                    per_game_play_results.append({
                        "game_id": game_id,
                        "state": str(final.state),
                        "levels_completed": final.levels_completed,
                        "actions": agent.action_counter,
                        "wall_time_seconds": time.monotonic() - start,
                        "exception": None,
                    })
                    print(
                        f"  -> state={final.state}, "
                        f"levels_completed={final.levels_completed}, "
                        f"actions={agent.action_counter}"
                    )
                except Exception as exc:  # noqa: BLE001 - one bad game must not lose the rest
                    per_game_play_results.append({
                        "game_id": game_id,
                        "state": None,
                        "levels_completed": None,
                        "actions": None,
                        "wall_time_seconds": time.monotonic() - start,
                        "exception": repr(exc),
                    })
                    print(f"  -> EXCEPTION: {exc!r}")
            """
        )
    )
```

- [ ] **Step 5: Update `intro_cell`'s text to describe the new behavior**

Replace the existing `intro_cell = markdown_cell(...)` assignment with:

```python
    intro_cell = markdown_cell(
        "# baseline-120 — Colab Gemma-4-31B real-game validation\n\n"
        "Extends Day 2's one-game smoke test "
        "(`docs/superpowers/experiments/baseline-100.md`) into a real, "
        "scored `baseline-120-reki-core` validation run across a "
        "documented multi-game sample. Development notebook, not the "
        "Kaggle submission (see `scripts/build_notebook.py` for that). "
        "Attach an A100 or L4 GPU runtime before running (Runtime > Change "
        "runtime type).\n\n"
        "1. Install pinned deps + vLLM\n"
        "2. Clone this repo at the exact commit and check out `zerx/`\n"
        "3. Print the resolved environment (GPU, package versions — no secrets)\n"
        "4. Start a local vLLM server for `google/gemma-4-31B-it`\n"
        "5. Play each game in `GAME_SAMPLE` directly via `MyAgent`, capped at "
        "`MAX_STEPS_PER_GAME` actions each\n"
        "6. Save each game's real outcome (state, levels completed, actions) "
        "plus its RHAE from `arc.get_scorecard()` to Google Drive (outside "
        "ephemeral runtime storage)"
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_build_colab_notebook.py -v`

Expected: all tests PASS, including the 6 new ones and the pre-existing
`test_build_wires_gemma_model_backend_against_local_vllm_server`,
`test_build_saves_structured_results_outside_ephemeral_storage`, and
`test_build_prints_environment_without_secrets` (which must still hold —
no `CEREBRAS_API_KEY`/`KAGGLE_API_TOKEN` appear anywhere in the rewritten
cell).

- [ ] **Step 7: Commit**

```bash
git add scripts/build_colab_notebook.py tests/test_build_colab_notebook.py
git commit -m "feat(colab-notebook): play a documented 8-game sample in-process instead of one-game subprocess smoke test"
```

---

## Task 2: Real per-game RHAE capture in `save_results_cell`

**Files:**
- Modify: `scripts/build_colab_notebook.py` (`save_results_cell` inside `build()`)
- Test: `tests/test_build_colab_notebook.py`

**Interfaces:**
- Consumes: `per_game_play_results` (a list of dicts with keys `game_id`,
  `state`, `levels_completed`, `actions`, `wall_time_seconds`, `exception`
  — produced by Task 1's rewritten `smoke_game_cell`), `GAME_SAMPLE`,
  `MAX_STEPS_PER_GAME`, and the module-level `arc` object Task 1's cell
  constructs — all as **generated-source-level** names shared between two
  cells in the same notebook kernel, not a Python import.
- Produces: a `per_game_full` list (each entry = the corresponding
  `per_game_play_results` entry plus `rhae` and `rhae_message`) written to
  the Drive JSON under the `"per_game"` key.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_build_colab_notebook.py`:

```python
def test_save_results_cell_captures_real_rhae_via_get_scorecard():
    """docs/superpowers/experiments/baseline-100.md's own 'Known gap' --
    the old save_results_cell recorded only environment/setup metadata,
    never the actual per-game outcome or RHAE. Must now query
    arc.get_scorecard()'s EnvironmentScorecard per game (README.md's
    frozen interface: EnvironmentScorecard.environments, each an
    EnvironmentScoreList with .score/.actions/.levels_completed, matched
    by .id).
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "arc.get_scorecard()" in combined
    assert "find_environment" in combined


def test_save_results_cell_saves_full_per_game_breakdown_not_just_aggregate():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert '"per_game": per_game_full' in combined
    assert '"game_id": "ls20"' not in combined  # no longer a single hardcoded game
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/pytest.exe tests/test_build_colab_notebook.py -v -k save_results`

Expected: both new tests FAIL (`find_environment` and
`"per_game": per_game_full` are absent from the current `save_results_cell`,
which still only writes a flat single-game metadata dict).

- [ ] **Step 3: Rewrite `save_results_cell`**

Replace the existing `save_results_cell = code_cell(...)` assignment with:

```python
    save_results_cell = code_cell(
        dedent(
            """\
            import json
            import subprocess
            from google.colab import drive

            drive.mount("/content/drive")

            scorecard = arc.get_scorecard()

            def _rhae_for(game_id):
                env_score_list = scorecard.find_environment(game_id)
                if env_score_list is None or not env_score_list.runs:
                    return None, "no EnvironmentScoreList found for this game_id"
                latest_run = env_score_list.runs[-1]
                return latest_run.score, latest_run.message

            per_game_full = []
            for entry in per_game_play_results:
                if entry["exception"] is not None:
                    rhae, rhae_message = None, "game raised an exception before scoring"
                else:
                    rhae, rhae_message = _rhae_for(entry["game_id"])
                per_game_full.append({**entry, "rhae": rhae, "rhae_message": rhae_message})

            result = {
                "experiment_id": "baseline-120",
                "model_revision": "gemma-4-31b-it",
                "base_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True
                ).stdout.strip(),
                "gpu": subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True,
                ).stdout.strip(),
                "dtype": "bfloat16",
                "game_sample": GAME_SAMPLE,
                "max_steps_per_game": MAX_STEPS_PER_GAME,
                "per_game": per_game_full,
                "aggregate_score": scorecard.score,
            }
            out_path = "/content/drive/MyDrive/zerx-baseline-120-result.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print("Saved:", out_path)
            """
        )
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/pytest.exe tests/test_build_colab_notebook.py -v`

Expected: all tests PASS (full file, not just the `-k save_results` subset).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_colab_notebook.py tests/test_build_colab_notebook.py
git commit -m "feat(colab-notebook): capture real per-game RHAE from arc.get_scorecard() instead of environment-only metadata"
```

---

## Task 3: Regenerate the notebook and run the full suite

**Files:**
- Generated (gitignored, not committed): `notebooks/colab_gemma_smoke.ipynb`

**Interfaces:**
- Consumes: Tasks 1+2's finished generator.
- Produces: nothing new tests depend on — this task is a verification gate before moving to Part A's remaining item (Task 4).

- [ ] **Step 1: Regenerate the notebook**

Run: `.venv/Scripts/python.exe scripts/build_colab_notebook.py`

Expected output: `[build_colab_notebook] Wrote notebooks/colab_gemma_smoke.ipynb`
with no traceback.

- [ ] **Step 2: Run the full local suite**

Run: `.venv/Scripts/pytest.exe tests/ -q`

Expected: all tests pass, count grown from 261 to 267 (261 baseline, minus
1 superseded `play_local.py` test, plus 7 new tests across Tasks 1–2).

- [ ] **Step 3: Spot-check the generated notebook's JSON is well-formed**

Run:

```bash
.venv/Scripts/python.exe -c "import json; nb = json.load(open('notebooks/colab_gemma_smoke.ipynb', encoding='utf-8')); print(nb['nbformat'], len(nb['cells']))"
```

Expected: prints `4 7` (unchanged cell count — only cell *content*
changed, not cell *count*) with no exception.

---

## Task 4: Standalone Cerebras prompt/parse sanity check (Part A step 3)

**Files:**
- Create (scratch, NOT part of this repo, never committed): a script under
  this session's scratchpad directory, e.g.
  `<scratchpad>/cerebras_sanity_check.py` — per
  `docs/superpowers/plans/parallel-baseline-120/person-4-colab-validation.md`'s
  own instruction ("a scratch script you keep local"), this file is
  deliberately outside the repo and outside version control.

**Interfaces:**
- Consumes: `zerx.backends.cerebras_dev.CerebrasDevBackend`,
  `zerx.policy.build_prompt`/`parse_action`, `zerx.perception.PerceptionResult`/`LabeledObject`,
  `zerx.memory.MemoryState`, `zerx.types.ActionName` — all existing,
  unmodified.
- Produces: a printed pass/fail sanity result for this session's status
  update; not consumed by any other task.

- [ ] **Step 1: Check for a Cerebras credential before writing anything that would need it**

Run (do not print the value):

```bash
if [ -n "$CEREBRAS_API_KEY" ]; then echo "present"; else echo "absent"; fi
```

If `absent`: **stop this task here.** Record in this session's status
update (and later in Task 8's experiment doc) that this check was skipped
for lack of a `CEREBRAS_API_KEY` in this environment — per
`person-4-colab-validation.md`'s explicit instruction: "if you genuinely
cannot get a key, say so explicitly in your status update rather than
silently skipping it." Do not fabricate a result. Move on to Task 5.

If `present`: continue to Step 2.

- [ ] **Step 2: Write the scratch script**

```python
"""Standalone, NEVER-COMMITTED sanity check: constructs CerebrasDevBackend
directly (bypassing agent/my_agent.py's backend wiring entirely) and
exercises zerx.policy.build_prompt / parse_action against a real Cerebras
response, in isolation from the game loop, heuristics, and memory
refresh. Requires CEREBRAS_API_KEY in this shell's environment.

Per AGENTS.md's Cerebras development boundary: this file, its output, and
CEREBRAS_API_KEY itself must never be committed, logged into the repo, or
pasted into any notebook cell.
"""
import os
import sys

REPO_ROOT = r"C:\Users\iefey\Documents\GitHub\zerx-harness"
sys.path.insert(0, REPO_ROOT)

from zerx.backends.cerebras_dev import CerebrasDevBackend
from zerx.memory import MemoryState
from zerx.perception import LabeledObject, PerceptionResult
from zerx.policy import build_prompt, parse_action
from zerx.types import ActionName

if not os.environ.get("CEREBRAS_API_KEY"):
    raise SystemExit("CEREBRAS_API_KEY not set -- nothing to sanity-check")

# AGENTS.md's documented public preview model id as of August 2026 --
# re-verify against the account's live model list before treating this as
# authoritative for anything beyond this one-off sanity check.
backend = CerebrasDevBackend(model_id="gemma-4-31b", platform="local")
print("credential_present:", backend.credential_present)

perception = PerceptionResult(
    ascii_grid="\n".join("0" * 8 for _ in range(8)),
    objects=(LabeledObject(label="obj0", color=2, cells=((3, 3), (3, 4))),),
)
memory = MemoryState(summary="")
legal_actions = frozenset({ActionName.ACTION1, ActionName.ACTION6, ActionName.RESET})

prompt = build_prompt(perception, memory, candidates=())
print("prompt chars:", len(prompt))

raw = backend.generate(prompt)
print("raw response (first 500 chars):", raw[:500])

parsed = parse_action(raw, legal_actions)
if parsed is None:
    print("PARSE FAILED -- raw response did not yield a valid legal action")
else:
    print("parsed action:", parsed.action, "repaired:", parsed.repaired)
print("last_latency_seconds:", backend.last_latency_seconds)
```

- [ ] **Step 3: Run it**

Run: `.venv/Scripts/python.exe <scratchpad>/cerebras_sanity_check.py`

Expected: prints `credential_present: True`, a nonzero prompt length, a
raw response excerpt, and either a successfully parsed action or an
explicit `PARSE FAILED` line (both are valid, recordable outcomes — a
parse failure here is itself useful signal about the prompt/schema, not a
task failure). Record the actual result (pass or fail, and why) for
Task 8's experiment doc. Never paste the raw `CEREBRAS_API_KEY` value or
the full unredacted response if it might echo the key back (it should
not, since the key is only sent as a bearer header, never echoed by a
well-behaved API, but eyeball the printed output before recording it).

---

## Task 5: Check Track 1 (`select_backend`) availability — Part B gate

**Files:** none modified — this is a read-only reconnaissance step.

**Interfaces:**
- Consumes: `origin/feat/baseline-120-backend-wiring` (Track 1's branch,
  if pushed) or a locally-merged copy of it.
- Produces: a go/no-go decision for Tasks 6–7. If this gate fails, Tasks
  6–7 do not run this session; Task 8 documents the block honestly
  instead of fabricating results.

- [ ] **Step 1: Fetch and check for Track 1's branch**

```bash
git fetch origin
git ls-remote --heads origin feat/baseline-120-backend-wiring
git branch --list feat/baseline-120-backend-wiring
```

- [ ] **Step 2: Branch on the result**

If **both** commands produce no output (branch not on origin, not local):
**stop here.** Do not proceed to Tasks 6–7 this session. Record in
Task 8/9 that Part B (the `cerebras_dev` sweep and the Colab run) is
blocked on Track 1's backend-selection factory
(`feat/baseline-120-backend-wiring`) not yet existing anywhere reachable,
exactly as `docs/superpowers/plans/parallel-baseline-120/person-4-colab-validation.md`'s
own "Failure-mode behavior" section anticipates as a valid, honest
hand-off state. Skip directly to Task 8.

If the branch **is** found (remote or local): merge it into this branch
and confirm `select_backend` is importable:

```bash
git merge origin/feat/baseline-120-backend-wiring   # or the local branch name if only local
.venv/Scripts/python.exe -c "from zerx.model_backend import select_backend; print(select_backend)"
.venv/Scripts/pytest.exe tests/ -q
```

Expected on success: the merge completes cleanly (or with a mechanical,
resolvable conflict confined to `zerx/config.py`/`agent/my_agent.py` per
`docs/superpowers/plans/parallel-baseline-120/README.md`'s "Ownership
matrix" — those files are Track 1's, resolve by keeping Track 1's side),
`select_backend` imports without error, and the full suite still passes
(261 + this branch's own new tests + Track 1's new tests, no regressions).
Proceed to Task 6.

---

## Task 6 (gated on Task 5): Full `cerebras_dev` sweep via the real harness

**Only run this task if Task 5 confirmed `select_backend` exists AND Task
4 confirmed a `CEREBRAS_API_KEY` is present in this environment.** If
either precondition is missing, skip to Task 8 and document which one.

**Files:** none modified — this is a measurement run, not a code change.

- [ ] **Step 1: Run the documented game sample through the real harness with `cerebras_dev`**

```bash
export ZERX_BACKEND=cerebras_dev
export ZERX_PLATFORM=local
export ZERX_MODEL_REVISION=gemma-4-31b
.venv/Scripts/python.exe scripts/play_local.py --game ls20,vc33,su15,tn36,ka59,lf52,tr87,sc25 --max-steps 100
```

(`CEREBRAS_API_KEY` stays whatever is already set in this shell — do not
re-export or print it.)

- [ ] **Step 2: Record the printed per-game summary and aggregate score**

Copy `play_local.py`'s `SUMMARY` block (per-game `state`/`levels`/`actions`
lines) and the final `Aggregate scorecard score: <value>` line verbatim
into this session's notes for Task 8 — this is the dev-lane proxy result,
labeled explicitly as such, never as the `baseline-120` score itself
(`AGENTS.md`'s hard backend-mismatch rule).

---

## Task 7 (gated on Task 5): Confirm the notebook still resolves `gemma_local` post-Track-1

**Files:** none — verification only, per
`person-4-colab-validation.md`'s step 5 ("confirm `ZERX_BACKEND=gemma_local`
still resolves correctly through Track 1's `select_backend`").

- [ ] **Step 1: Regenerate and re-run the suite once more on top of the merged Track 1 code**

```bash
.venv/Scripts/python.exe scripts/build_colab_notebook.py
.venv/Scripts/pytest.exe tests/ -q
```

Expected: no change in behavior — `gemma_local` already mapped correctly
even before Track 1's fix (per this plan's README's "concrete, empirical
finding"), so this step is a confirmation, not a code change.

---

## Task 8: Write `docs/superpowers/experiments/baseline-120.md`

**Files:**
- Create: `docs/superpowers/experiments/baseline-120.md`

**Interfaces:**
- Consumes: Task 3's regenerated-notebook confirmation, Task 4's Cerebras
  sanity-check result (pass/fail/skipped), Task 5's gate outcome, Task 6's
  sweep results (if run), Task 7's confirmation (if run).
- Produces: the experiment record other sessions read to decide
  `STRATEGY.md` §7's next rung.

- [ ] **Step 1: Write the record, following `baseline-100.md`'s field structure**

Use this structure (fill in the actual Task 4/5/6 outcomes from this
session — do not invent Colab numbers; if Task 6 didn't run, its section
says so plainly):

```markdown
# baseline-120 — real-game validation (Reki-core: reflection + click proposals + soft failure memory)

- Date: 2026-08-05
- Base commit: `8a8a01ad155227aee6f00a5844d1e1bd9da5f4cb` (branch `feat/baseline-120-colab-validation`)
- Game sample: `ls20, vc33, su15, tn36, ka59, lf52, tr87, sc25` (8 of the
  25 documented public games — keeps the existing `ls20`+`vc33` precedent
  this plan's own README measured its 0.0/all-ACTION6 "before" reference
  against, plus 6 more spread across the documented list for per-game
  regression coverage per `AGENTS.md`'s "repeated seeds/configurations"
  language). `max_steps_per_game=100` (below `play_local.py`'s 200
  default — no prior per-decision latency measurement exists for this
  31B model in this repo, so 8 games × 200 steps risked exceeding a
  single Colab session; 100 was chosen as a documented, conservative
  trade-off, to be revisited once real timing exists).

## Part A — notebook + tooling (this session, no GPU/Cerebras dependency)

- `scripts/build_colab_notebook.py`'s `smoke_game_cell` rewritten from a
  one-game subprocess call (`!python3.12 scripts/play_local.py --game
  ls20 --max-steps 50`) to an in-process loop over `GAME_SAMPLE` driving
  `MyAgent` directly, so `save_results_cell` (same kernel) can query
  `arc.get_scorecard()` for real per-game RHAE.
- `save_results_cell` now writes `per_game` (state, levels_completed,
  actions, wall_time_seconds, exception, rhae, rhae_message) for every
  sampled game, not just environment/setup metadata — closes the exact
  gap `baseline-100.md` flagged.
- Full local suite: [FILL IN: pass count from Task 3 Step 2, e.g. "267
  passed, 0 failed"].
- Notebook regenerated cleanly (Task 3), JSON well-formed, 7 cells
  (unchanged count).

## Cerebras prompt/parse sanity check (Part A step 3)

[FILL IN one of:]
- Skipped: no `CEREBRAS_API_KEY` present in this session's environment
  (Task 4 Step 1). Per-teammate credential per `AGENTS.md` — whoever runs
  this plan next with a key available should complete this step and
  update this section.
- Completed: credential present, prompt length N chars, parse
  [succeeded producing action X | failed — raw response excerpt: "..."],
  latency N seconds.

## Part B — dev-lane `cerebras_dev` sweep via the real harness

[FILL IN one of:]
- Blocked: Track 1's backend-selection factory
  (`feat/baseline-120-backend-wiring`, shipping `zerx.model_backend.select_backend`)
  was not found on `origin` or locally as of this session (Task 5 Step 1
  — `git ls-remote --heads origin feat/baseline-120-backend-wiring`
  returned nothing). Without it, `agent/my_agent.py`'s `__init__` still
  unconditionally constructs `GemmaModelBackend` regardless of
  `Config.backend`, so a `cerebras_dev` sweep through the real harness
  would silently fall through to fallback heuristics/deterministic
  actions exactly like this plan's own measured "before" reference — not
  a meaningful signal. [If also blocked on the Cerebras key: "Also no
  `CEREBRAS_API_KEY` was available this session (see above), so this
  step is doubly blocked."]
- Completed: [FILL IN Task 6's recorded per-game summary and aggregate
  score, labeled explicitly as a dev-lane proxy result, never as the
  baseline-120 score itself].

## Part B — authoritative Colab Gemma-4-31B-it run

**Not performed this session.** Running the generated notebook on a real
Colab A100/L4 GPU runtime requires a human to open
`notebooks/colab_gemma_smoke.ipynb` in a browser at colab.research.google.com,
attach a GPU runtime, and run all cells — no tool available to this
Claude Code session can drive that. [If Part B's harness sweep also
didn't run: "The harness-level `cerebras_dev` sweep above did not run
either, so no real per-game signal exists yet for this rung — see
'blocked' notes above."]

## Conclusion

`investigate` per `STRATEGY.md` §7.1 — closest fit to "a
measurement/logging defect exists" (following `baseline-100.md`'s own
precedent for this exact situation): the notebook/tooling gap
`baseline-100.md` flagged is now closed (Part A), but no real per-game
model-in-loop measurement exists yet for `baseline-120` itself — neither
the `cerebras_dev` dev-lane proxy nor the authoritative Colab Gemma run.
This is explicitly **not** `keep` (no evidence supports it) and explicitly
**not** `revert` (nothing was measured to revert). Re-run this experiment
record's Part B once Track 1 merges and a Cerebras key is available for
the dev-lane sweep, then have a human execute the Colab run before
re-deciding this conclusion.
```

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/experiments/baseline-120.md
git commit -m "docs(experiment): record baseline-120 Part A completion and Part B blockers"
```

---

## Task 9: Update `docs/HANDOFF.md`

**Files:**
- Modify: `docs/HANDOFF.md`

- [ ] **Step 1: Add this track's one-line status update**

Per `docs/superpowers/plans/parallel-baseline-120/README.md`'s ownership
matrix ("all 4 (one-line status each)... append/edit only your own status
line during your track") — since Part B/Colab isn't complete, do **not**
rewrite "Exact next action" (that's reserved for once the real run is
complete, per `person-4-colab-validation.md`). Add a status line under a
suitable existing section, e.g.:

```markdown
- Track 4 (`feat/baseline-120-colab-validation`, Colab validation):
  Part A complete (multi-game notebook + real RHAE capture, full suite
  green) — commit `[FILL IN actual HEAD SHA]`. Part B blocked this
  session: Track 1's `select_backend`
  (`feat/baseline-120-backend-wiring`) not yet reachable on origin/local,
  [and/or] no `CEREBRAS_API_KEY` available in this environment. See
  `docs/superpowers/experiments/baseline-120.md` for full detail.
```

- [ ] **Step 2: Commit**

```bash
git add docs/HANDOFF.md
git commit -m "docs(handoff): record baseline-120 colab-validation Part A status and Part B blockers"
```

---

## Task 10: Push the branch

**Files:** none — git operation only.

- [ ] **Step 1: Confirm the full suite one final time**

Run: `.venv/Scripts/pytest.exe tests/ -q`

Expected: all tests pass, matching Task 3 Step 2's count.

- [ ] **Step 2: Push to the track's own branch only**

```bash
git push -u origin feat/baseline-120-colab-validation
```

Do **not** merge to `master` — per this plan's Global Constraints and the
parallel-baseline-120 README's merge-order (`INTEGRATION.md` in that
directory governs the actual master merge, done by the integration owner
after all 4 tracks land).

---

## Self-review notes

- **Spec coverage:** Part A step 1 (game list) → Task 1. Part A step 2
  (RHAE capture) → Task 2. Part A step 3 (Cerebras sanity check) → Task 4.
  Part B steps 4–7 (sweep, notebook regen, Colab run, experiment doc) →
  Tasks 5–8 (gated honestly where real infrastructure is unavailable this
  session). `docs/HANDOFF.md` update → Task 9. Push-only-own-branch → Task
  10. Ownership/scope restrictions from the README/person-4 file are
  encoded in "Global Constraints" above.
- **No placeholders:** every step has real, runnable code. The only
  bracketed `[FILL IN: ...]` spots are in Task 8's experiment-doc template,
  which is inherently only fillable with this session's actual
  measurement/test-count results at execution time — not vague TODOs, but
  explicit "insert this session's real number here" markers, consistent
  with `baseline-100.md`'s own precedent of an honest, partially-measured
  record.
- **Type/name consistency:** `GAME_SAMPLE`/`MAX_STEPS_PER_GAME` (Task 1)
  are read by name in Task 2's `save_results_cell` and Task 8's doc.
  `per_game_play_results` (Task 1) → `per_game_full` (Task 2) → `"per_game"`
  JSON key (Task 2) → cited in Task 8. `select_backend` (Task 5's gate) is
  the exact frozen signature name from
  `docs/superpowers/plans/parallel-baseline-120/README.md`.
