"""Tests for scripts/build_colab_notebook.py — verifies the generated
Colab notebook's cell CONTENT satisfies AGENTS.md's Colab gate. Cannot
verify GPU execution itself (no GPU in this environment) — that happens
when the human runs the generated notebook on Colab.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_colab_notebook  # noqa: E402


def _all_cell_sources(notebook: dict) -> str:
    return "\n".join(cell.get("source", "") for cell in notebook["cells"])


def test_build_produces_valid_notebook_structure():
    notebook = build_colab_notebook.build()
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) > 0
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("code", "markdown")


def test_build_pins_dependency_versions():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "vllm==" in combined
    assert "pip install" in combined


def test_build_does_not_pin_the_pre_gemma4_vllm_version():
    """vllm==0.11.0 (Oct 2025) predates Gemma 4's release (2026-03-26) by
    ~5 months and cannot parse its rope_scaling config -- real Colab run
    (2026-08-04) hit exactly this ("rope_scaling should have a 'rope_type'
    key"). Must never regress to actually INSTALLING that known-broken
    pin (the version number may still appear in an explanatory comment
    about why it's avoided -- only the quoted install-target string matters).
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert '"vllm==0.11.0"' not in combined


def test_build_installs_vllm_via_uv_torch_backend_auto():
    """Plain `pip install vllm` pulls vLLM's default CUDA-12.9-compiled
    binary regardless of the actual driver's CUDA version -- real Colab
    run (2026-08-04, driver reporting CUDA 13.0) hit exactly this:
    "ImportError: libcudart.so.13: cannot open shared object file". Per
    vLLM's own install docs, `uv pip install --torch-backend=auto` detects
    the installed driver's CUDA version and selects a matching build.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "uv pip install" in combined
    assert "--torch-backend=auto" in combined


def test_build_forces_reinstall_to_avoid_colabs_preexisting_torch():
    """--torch-backend=auto alone was NOT enough (real Colab run,
    2026-08-04): the same libcudart.so.13 error recurred, because pip/uv
    treated Colab's pre-existing torch as already satisfying the
    requirement and left it untouched, pairing it with a freshly
    installed, differently-CUDA-linked vLLM extension -- exactly the
    "binary incompatibility" vLLM's own docs warn about. --reinstall
    forces uv to actually replace the pre-existing state.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "--reinstall" in combined


def test_env_print_cell_reports_torchs_actual_cuda_build():
    """nvidia-smi's "CUDA Version" is the driver's max-supported ceiling,
    not what torch/vllm actually linked against -- confusing the two
    caused two rounds of misdiagnosis on 2026-08-04. The env-print cell
    must print torch's own resolved CUDA build directly so a future
    libcudart-style failure is diagnosable from this cell alone.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "torch.version.cuda" in combined
    assert "torch.cuda.is_available()" in combined


def test_build_checks_out_exact_commit():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "git checkout" in combined
    assert "git clone" in combined


def test_build_clones_and_slims_the_agents_framework():
    """agent/my_agent.py does `from agents.agent import Agent` -- that
    package is the arcprize/ARC-AGI-3-Agents framework (not a pip package)
    and must actually be cloned into vendor/ARC-AGI-3-Agents before the
    smoke-game cell imports MyAgent. Real Colab run confirmed the notebook
    previously never cloned it at all, raising 'ModuleNotFoundError: No
    module named agents'. Must also run scripts/slim_framework.py, matching
    `make setup`'s local flow, so the upstream __init__.py's eager
    langgraph/langsmith/smolagents imports (never installed here) don't
    raise a second, different ImportError once the clone itself is fixed.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "git clone" in combined
    assert "ARC-AGI-3-Agents.git" in combined
    assert "vendor/ARC-AGI-3-Agents" in combined
    assert "slim_framework.py" in combined


def test_build_prints_environment_without_secrets():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "nvidia-smi" in combined or "torch.cuda" in combined
    assert "CEREBRAS_API_KEY" not in combined
    assert "KAGGLE_API_TOKEN" not in combined


def test_build_loads_exact_gemma_revision():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert '"--model", "google/gemma-4-31B-it"' in combined


def test_build_does_not_pass_kaggle_ui_slug_as_the_vllm_model_argument():
    """The Kaggle Models UI labels this "google/gemma-4/Transformers/gemma-4-31b-it"
    (owner/model/framework/variant -- Kaggle's own organizational path), but
    that string is NOT a valid Hugging Face Hub repo id: vLLM/transformers
    reject it outright with HFValidationError. Real Colab run (2026-08-04)
    hit exactly this. The Kaggle label may still appear in prose/comments
    for context, but must never be the string actually passed to --model.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert '"--model", "google/gemma-4/Transformers/gemma-4-31b-it"' not in combined


def test_build_wires_gemma_model_backend_against_local_vllm_server():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "GemmaModelBackend" in combined
    assert "localhost:8000" in combined


def test_build_runs_bf16_unquantized_matching_what_kaggle_actually_runs():
    """Colab must load the precision Kaggle will actually deploy, so a Colab
    result is comparable to the deployment rather than to a model that never
    ships. That requirement has not changed; the precision it implies has.

    The previous fp8 setting rested on "Kaggle's RTX Pro 6000 has 48GB, so
    bf16 (~61.4GB) does not fit". That number was never measured. The
    environment probe measured it (docs/superpowers/experiments/kaggle-env-probe.md):
    97887 MiB -- ~96GB, not 48GB -- against 62.58GB of bf16 weights. So bf16
    fits on both cards (Colab's A100-SXM4-80GB has ~17GB spare, Kaggle ~33GB),
    parity holds at the higher precision, and the quantization was a cost paid
    for a constraint that does not exist.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert '"--quantization"' not in combined, "bf16 fits on both cards"
    assert '"--dtype", "bfloat16"' in combined
    assert "--max-model-len" in combined


def test_build_does_not_install_bitsandbytes_now_that_fp8_is_used():
    """bitsandbytes was only ever needed for the old 4-bit in-flight path;
    FP8 quantization is native to vLLM. Must not regress to actually
    INSTALLING the now-unused package (the name may still appear in an
    explanatory comment about why 4-bit/bitsandbytes was rejected in favor
    of fp8 -- only the install command matters, same convention as
    test_build_does_not_pin_the_pre_gemma4_vllm_version above).
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert 'pip install -q "bitsandbytes' not in combined


def test_build_surfaces_real_vllm_server_log_on_startup_failure():
    """The original cell raised a bare 'did not become ready in time'
    SystemExit with no visibility into vLLM's actual error -- real Colab
    run (2026-08-04) hit exactly this, with no way to diagnose the cause.
    The server's stdout/stderr must be captured to a log file and tailed
    into the failure output.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "vllm_server.log" in combined
    assert "_tail_log" in combined
    assert "poll()" in combined  # detects the process dying early, not just a timeout


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


def test_smoke_game_cell_sets_max_actions_via_config_env_not_via_min():
    """`min(getattr(MyAgentCls, "MAX_ACTIONS", ...), MAX_STEPS_PER_GAME)` can
    only ever LOWER the cap, never raise it above the vendored base Agent
    class's own default (80) -- confirmed by a real Colab run capping at
    81 actions/game instead of the requested 100 (docs/HANDOFF.md "Known
    failures or risks" item 7).

    The cap is now a Config field that `MyAgent.__init__` applies to the
    *instance* (`self.MAX_ACTIONS = config.max_actions`), so setting the
    class attribute here would be silently overwritten at construction --
    the cell must go through ZERX_MAX_ACTIONS instead.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert 'os.environ["ZERX_MAX_ACTIONS"] = str(MAX_STEPS_PER_GAME)' in combined
    assert "MyAgentCls.MAX_ACTIONS =" not in combined
    assert 'getattr(MyAgentCls, "MAX_ACTIONS"' not in combined


def test_smoke_game_cell_wires_trace_export_for_diagnosability():
    """A 0.0 aggregate score with no exceptions is otherwise unexplainable
    after the fact -- MyAgent.__init__ already wires a JsonlTraceWriter off
    Config.trace_export_path (zerx/trace.py) whenever ZERX_TRACE_EXPORT_PATH
    is set; this cell must actually set it, and save_results_cell must copy
    the resulting trace files off ephemeral Colab storage the same way it
    already does for the result JSON.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert 'os.environ["ZERX_TRACE_EXPORT_PATH"]' in combined
    assert "shutil.copytree" in combined
    assert "TRACE_DST" in combined


def test_smoke_game_cell_raises_budget_soft_cap_for_this_diagnostic_run():
    """zerx/budget.py's should_favor_execution flips at 80% of
    budget_soft_cap (default 50, i.e. action 40) and zerx/policy.py's
    decide() then skips the model call in favor of the top heuristic click
    candidate -- silently turning the back half of every
    MAX_STEPS_PER_GAME=100 game into heuristic-only play. This diagnostic
    run needs the model exercised for the whole game, so the cap must be
    raised well above MAX_STEPS_PER_GAME here (a per-run override, not a
    zerx/config.py default change).
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert 'os.environ["ZERX_BUDGET_SOFT_CAP"]' in combined


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


def test_save_results_cell_records_the_actual_quantization_method():
    """The result JSON previously recorded only 'dtype', never which
    quantization was actually active -- silently indistinguishable between
    a 4-bit run and a bf16 run. Must record 'quantization' explicitly so a
    saved result is self-describing without cross-referencing the notebook
    source that produced it.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    # The value changed from "fp8" to None when the probe showed bf16 fits on
    # both cards, but the property being tested did not: a saved result must
    # still say which quantization was active, so "no quantization" is
    # recorded explicitly rather than left to be inferred from a missing key.
    assert '"quantization": None' in combined


def test_build_saves_structured_results_outside_ephemeral_storage():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "drive.mount" in combined
    assert "json.dump" in combined


def test_main_writes_notebook_file(tmp_path, monkeypatch):
    target = tmp_path / "colab_gemma_smoke.ipynb"
    monkeypatch.setattr(build_colab_notebook, "NOTEBOOK_PATH", target)
    build_colab_notebook.main()
    assert target.exists()
    import json
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
