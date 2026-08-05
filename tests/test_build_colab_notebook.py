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


def test_build_quantizes_the_model_to_fit_a_40gb_gpu():
    """31B dense in bf16/fp16 is ~2 bytes/param, ~61GB of weights alone --
    does not fit an A100-SXM4-40GB's 40GB VRAM. Real Colab run (2026-08-04)
    confirmed the unquantized bf16 config never brought the vLLM server up
    within the wait window. Must load quantized (bitsandbytes) instead, and
    the quantization package must actually be installed.
    """
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "bitsandbytes" in combined
    assert "--quantization" in combined
    assert "--max-model-len" in combined


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
