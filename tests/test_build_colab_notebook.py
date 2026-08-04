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
    assert "google/gemma-4/Transformers/gemma-4-31b-it" in combined


def test_build_wires_gemma_model_backend_against_local_vllm_server():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "GemmaModelBackend" in combined
    assert "localhost:8000" in combined


def test_build_runs_one_public_game_via_play_local():
    combined = _all_cell_sources(build_colab_notebook.build())
    assert "play_local.py" in combined


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
