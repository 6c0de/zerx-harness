"""Regression tests for the Kaggle submission bundle produced by
`scripts/build_notebook.py`.

`tests/test_build_notebook.py` already checks the packaging *mechanism*
(which files the glob picks up, that `zerx/backends/` is excluded, that a
secret scan gates the build). It never checked the *outcome*: that the
bundle Kaggle actually receives can be imported. Both bugs below were live
on master and invisible to the existing suite, and each one alone makes a
submission score zero:

  1. `zerx/model_backend.py` imported `zerx.backends.cerebras_dev` at module
     level while the bundle deliberately ships only `zerx/*.py`, so
     `import zerx.model_backend` — and therefore `agent/my_agent.py` — died
     with `ModuleNotFoundError: No module named 'zerx.backends'`.
  2. The `%%writefile /tmp/zerx/<mod>.py` cells ran before anything created
     `/tmp/zerx`, and IPython's writefile magic calls plain `open(path,'w')`
     with no `makedirs`, so the first such cell raised FileNotFoundError on a
     fresh kernel.

These tests reconstruct the bundle exactly as the notebook writes it and
import it in a subprocess, so they fail if either regression returns.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_notebook  # noqa: E402


def _cell_sources(notebook: dict) -> list[str]:
    return [
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
    ]


def _materialize_bundle(destination: Path) -> Path:
    """Write out exactly the files the notebook's %%writefile cells create."""
    notebook = build_notebook.build()
    package_dir = destination / "zerx"
    package_dir.mkdir(parents=True, exist_ok=True)
    for source in _cell_sources(notebook):
        match = re.match(r"%%writefile (/tmp/zerx/[\w.]+)\n(.*)", source, re.DOTALL)
        if match:
            (package_dir / os.path.basename(match.group(1))).write_text(match.group(2))
        agent_match = re.match(r"%%writefile /tmp/my_agent\.py\n(.*)", source, re.DOTALL)
        if agent_match:
            (destination / "my_agent.py").write_text(agent_match.group(1))
    return destination


def test_bundled_zerx_package_imports_without_the_backends_subpackage(tmp_path):
    """The bundle omits `zerx/backends/` by design; importing what IS shipped
    must still succeed. Run in a subprocess so the repo's own fully-populated
    `zerx` package (which does have `backends/`) cannot satisfy the import.
    """
    bundle = _materialize_bundle(tmp_path)
    assert not (bundle / "zerx" / "backends").exists(), "bundle must not ship backends/"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import zerx.model_backend as m; "
            "from zerx.policy import decide; "
            "print(m.select_backend.__name__)",
        ],
        cwd=bundle,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(bundle)},
    )
    assert result.returncode == 0, (
        "bundled zerx package failed to import as Kaggle would import it:\n"
        f"{result.stderr}"
    )


def test_selecting_cerebras_backend_still_works_when_backends_is_present():
    """Making the cerebras import lazy must not disable the dev backend in the
    full repo checkout, where `zerx/backends/` does exist.
    """
    from zerx.config import Config
    from zerx.model_backend import select_backend

    backend = select_backend(Config(backend="cerebras_dev", platform="local"))
    assert backend.__class__.__name__ == "CerebrasDevBackend"


def test_notebook_creates_tmp_zerx_directory_before_writing_into_it():
    """IPython's `%%writefile` does not create parent directories, so a cell
    must create /tmp/zerx before the first write into it.
    """
    sources = _cell_sources(build_notebook.build())

    mkdir_indices = [
        i
        for i, source in enumerate(sources)
        if "makedirs" in source and "/tmp/zerx" in source
    ]
    write_indices = [
        i for i, source in enumerate(sources) if source.startswith("%%writefile /tmp/zerx/")
    ]

    assert mkdir_indices, "no cell creates /tmp/zerx before the %%writefile cells"
    assert write_indices, "expected %%writefile /tmp/zerx/... cells"
    assert min(mkdir_indices) < min(write_indices), (
        "/tmp/zerx must be created before the first file is written into it"
    )


def test_writefile_into_a_missing_directory_really_does_fail():
    """Pin the assumption the fix rests on: IPython's writefile magic is a
    plain open(), so writing into a missing directory raises. If a future
    IPython starts creating parents, this test tells us the guard cell is
    merely belt-and-braces rather than load-bearing.
    """
    import pytest

    missing = Path(__file__).resolve().parent / "_definitely_missing_dir" / "x.py"
    with pytest.raises(FileNotFoundError):
        with open(missing, "w", encoding="utf-8") as handle:
            handle.write("x = 1")


def _run_cell_source() -> str:
    """The cell that actually starts the scored run."""
    sources = _cell_sources(build_notebook.build())
    matches = [s for s in sources if "main.py --agent myagent" in s]
    assert len(matches) == 1, "expected exactly one cell to launch the agent"
    return matches[0]


def test_run_cell_selects_a_real_backend_not_the_silent_fake_one():
    """`Config.backend` defaults to "fake", whose `select_backend` returns a
    `FakeModelBackend` with no scripted responses — every `generate()` raises
    and the agent plays heuristics-only through its own fallback chain,
    without crashing. That is precisely how a submission can score ~nothing
    while looking healthy (docs/HANDOFF.md, ARC-HANDOFF-001). The run cell
    must set a real backend explicitly.
    """
    source = _run_cell_source()

    assert build_notebook.ZERX_BACKEND != "fake"
    assert f"ZERX_BACKEND={build_notebook.ZERX_BACKEND}" in source
    assert f"ZERX_PLATFORM={build_notebook.ZERX_PLATFORM}" in source
    assert build_notebook.ZERX_PLATFORM == "kaggle"


def test_run_cell_refuses_to_play_when_the_backend_resolves_to_fake():
    """A readiness gate, not just an env var: the env var could be right while
    the resolution still lands on the fake backend. The gate must run before
    the agent starts, and must raise rather than merely print — IPython
    swallows a shell command's non-zero exit.
    """
    source = _run_cell_source()

    assert "select_backend" in source, "run cell never resolves the backend"
    assert "FakeModelBackend" in source, "run cell never checks for the fake backend"
    assert "raise SystemExit" in source, "gate must abort, not just warn"

    gate_at = source.index("select_backend")
    launch_at = source.index("main.py --agent myagent")
    assert gate_at < launch_at, "the readiness gate must precede the run"


def test_run_cell_refuses_to_play_when_no_model_directory_is_configured():
    """Until the probe resolves the real /kaggle/input mount path,
    KAGGLE_MODEL_DIR is None and the notebook must decline to run at all —
    a submission that quietly has no weights is the failure being fixed.
    """
    source = _run_cell_source()
    assert "KAGGLE_MODEL_DIR" in source
    assert repr(build_notebook.KAGGLE_MODEL_DIR) in source
    assert "os.path.isdir(KAGGLE_MODEL_DIR)" in source


def test_kernel_metadata_attaches_weights_and_has_a_real_id():
    metadata = json.loads(build_notebook.METADATA_PATH.read_text(encoding="utf-8"))

    assert "REPLACE_WITH_YOUR_USERNAME" not in metadata["id"]
    assert metadata["id"].count("/") == 1, "kernel id must be <username>/<slug>"
    assert metadata["model_sources"], "no model attached: the agent would run modelless"
    assert metadata["model_sources"] == [build_notebook.MODEL_SOURCE]
    # Internet stays off; nothing may be downloaded at evaluation time.
    assert metadata["enable_internet"] is False


def test_accelerator_matches_the_documented_arc_agi_3_target():
    """AGENTS.md targets the ARC-AGI-3-exclusive RTX Pro 6000 (48GB). The
    starter's default of 2x16GB T4 cannot hold this model at any precision
    we are willing to run.
    """
    assert build_notebook.ACCELERATOR == "rtx6000"
    kaggle_meta = build_notebook.build()["metadata"]["kaggle"]
    assert kaggle_meta["accelerator"] == "nvidiaRtx6000"
    assert kaggle_meta["isGpuEnabled"] is True
    assert kaggle_meta["isInternetEnabled"] is False


def test_accelerator_is_exposed_as_a_push_flag_not_only_notebook_metadata():
    """Kaggle's push API reads only `enable_gpu` from kernel-metadata.json and
    ignores the notebook's own `metadata.kaggle.accelerator` entirely, so
    selecting a GPU that way silently does nothing — measured 2026-08-06,
    see docs/superpowers/experiments/kaggle-env-probe.md. The accelerator
    must reach the CLI as a `--accelerator` argument.
    """
    assert build_notebook.push_accelerator_flag(), (
        "a GPU accelerator must produce a --accelerator flag for the push"
    )
    # kernel-metadata.json carries no accelerator key at all; asserting it
    # stays absent keeps anyone from "fixing" this by adding one there, which
    # would look right and do nothing.
    metadata = json.loads(build_notebook.METADATA_PATH.read_text(encoding="utf-8"))
    assert "accelerator" not in metadata


def test_cpu_accelerator_produces_no_push_flag(monkeypatch):
    monkeypatch.setattr(build_notebook, "ACCELERATOR", "cpu")
    assert build_notebook.push_accelerator_flag() == ""


def test_run_cell_installs_nothing_from_the_network():
    """Internet is disabled at evaluation time; every install must be offline."""
    for source in _cell_sources(build_notebook.build()):
        if "pip install" in source:
            assert "--no-index" in source, (
                f"network-dependent install in a submission cell:\n{source}"
            )


def test_bundle_contains_no_cerebras_endpoint_or_credential_reference():
    """Secret hygiene must survive the lazy-import change: the endpoint and
    key name still must not appear anywhere in the shipped bundle except
    inside secret_scan.py's own detection patterns.
    """
    notebook = build_notebook.build()
    for source in _cell_sources(notebook):
        if source.startswith("%%writefile /tmp/zerx/secret_scan.py"):
            continue
        assert "api.cerebras.ai" not in source
        assert "CEREBRAS_API_KEY" not in source
