"""Tests for `scripts/build_probe_notebook.py` — the Kaggle environment probe.

The probe exists to answer questions about the real Kaggle runtime that
nobody has ever measured (see
`docs/superpowers/specs/2026-08-06-kaggle-p0-model-attach-design.md`). Two
properties make it worth anything, and both are easy to break silently:

  1. It must mirror the *submission* environment. A probe that runs on a T4
     with internet on, or with different weights attached, answers questions
     about a machine we will never submit from.
  2. One failing section must not cost the other answers. Kaggle aborts a
     notebook at the first uncaught exception, and a probe that dies partway
     through wastes the whole run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_notebook  # noqa: E402
import build_probe_notebook  # noqa: E402


def _sources(notebook: dict) -> list[str]:
    return [
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
    ]


def _joined(notebook: dict) -> str:
    return "\n".join(_sources(notebook))


def test_probe_notebook_is_structurally_valid_and_json_serializable():
    notebook = build_probe_notebook.build()
    json.dumps(notebook)  # a notebook Kaggle cannot parse is worthless
    assert notebook["nbformat"] == 4
    assert notebook["cells"], "probe notebook has no cells"
    for cell in notebook["cells"]:
        assert cell["cell_type"] in ("code", "markdown")


def test_probe_mirrors_the_submission_accelerator():
    """Probing a card we will never run on answers the wrong question."""
    probe_accel = build_probe_notebook.build()["metadata"]["kaggle"]["accelerator"]
    submission_accel = build_notebook.build()["metadata"]["kaggle"]["accelerator"]
    assert probe_accel == submission_accel


def test_probe_runs_with_internet_disabled_like_the_real_submission():
    kaggle_meta = build_probe_notebook.build()["metadata"]["kaggle"]
    assert kaggle_meta["isInternetEnabled"] is False
    assert kaggle_meta["isGpuEnabled"] is True

    metadata = build_probe_notebook.build_metadata()
    assert metadata["enable_internet"] is False
    assert metadata["enable_gpu"] is True


def test_probe_attaches_the_same_weights_the_submission_will_use():
    """If the two drift apart, the probe's mount-path answer does not transfer."""
    assert build_probe_notebook.MODEL_SOURCE == build_notebook.MODEL_SOURCE
    assert build_probe_notebook.build_metadata()["model_sources"] == [
        build_notebook.MODEL_SOURCE
    ]


def test_probe_metadata_has_a_real_kernel_id():
    metadata = build_probe_notebook.build_metadata()
    assert "REPLACE_WITH_YOUR_USERNAME" not in metadata["id"]
    assert metadata["id"].count("/") == 1, "kernel id must be <username>/<slug>"
    assert metadata["code_file"] == "probe.ipynb"


def test_probe_attaches_the_competition_so_the_wheels_directory_is_listable():
    metadata = build_probe_notebook.build_metadata()
    assert metadata["competition_sources"] == [build_probe_notebook.COMPETITION_SLUG]


def test_every_probe_section_is_error_trapped():
    """Kaggle stops at the first uncaught exception. Each section must record
    its own failure and let the rest of the run continue, or one missing
    package costs every other answer.
    """
    source = _joined(build_probe_notebook.build())
    assert "def section(name)" in source, "the error-trapping helper is gone"

    section_count = source.count("@section(")
    assert section_count >= 6, f"expected several probe sections, found {section_count}"

    # The helper's except branch is what makes the decorator safe; a refactor
    # that drops it would leave the decorators in place but stop protecting.
    helper = next(s for s in _sources(build_probe_notebook.build()) if "def section(" in s)
    assert "except Exception as exc:" in helper


def test_probe_persists_results_outside_the_log():
    """A result that only exists in kernel stdout is easy to lose and awkward
    to diff; the probe writes a downloadable artifact too.
    """
    source = _joined(build_probe_notebook.build())
    assert "/kaggle/working/probe.json" in source
    assert "json.dump(PROBE" in source


def test_probe_asks_the_questions_the_serving_decision_depends_on():
    source = _joined(build_probe_notebook.build())
    for probe_target in (
        "vllm",                    # can the planned serving path exist at all
        "bitsandbytes",            # the 4-bit fallback
        "compute_capability",      # >= 8.9 decides true FP8 W8A8
        "float8_e4m3fn",           # transformers-native FP8 viability
        "/kaggle/input",           # where the weights really mount
        "arc_agi_3_wheels",        # what the competition already ships offline
    ):
        assert probe_target in source, f"probe never checks {probe_target!r}"


def test_probe_neither_loads_a_model_nor_plays_a_game():
    """It is a measurement, not a run: it must stay cheap and side-effect free."""
    source = _joined(build_probe_notebook.build())
    for forbidden in ("from_pretrained", "main.py --agent", "LLM(", "vllm serve"):
        assert forbidden not in source, f"probe should not {forbidden!r}"


def test_probe_leaks_no_credentials():
    source = _joined(build_probe_notebook.build())
    assert "CEREBRAS_API_KEY" not in source
    assert "api.cerebras.ai" not in source
    assert "KAGGLE_API_TOKEN" not in source
    # os.environ is never dumped wholesale — that would print Kaggle's own
    # injected secrets straight into a notebook output. Reading one named
    # variable (os.getenv(...)) is fine and is what the probe actually does.
    assert "dict(os.environ)" not in source
    assert "os.environ.items()" not in source
    assert "os.environ)" not in source
