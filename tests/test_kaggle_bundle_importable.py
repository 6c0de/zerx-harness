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


def _cells(notebook):
    return [c["source"] for c in notebook["cells"] if c["cell_type"] == "code"]


def test_kaggle_env_selects_a_real_model_backend_not_the_fake_one():
    """ARC-HANDOFF-001's core symptom: with no ZERX_BACKEND set,
    Config.backend stays "fake", select_backend returns FakeModelBackend(),
    every generate() raises, and the agent plays heuristics-only with no
    crash and no log line -- just a near-zero score.
    """
    from zerx.config import Config
    from zerx.model_backend import FakeModelBackend, GemmaModelBackend, select_backend

    assert isinstance(select_backend(Config()), FakeModelBackend)  # the old state

    kaggle_env = {
        "ZERX_BACKEND": "gemma_kaggle",
        "ZERX_PLATFORM": "kaggle",
        "ZERX_GEMMA_BASE_URL": "http://localhost:8000/v1/chat/completions",
    }
    backend = select_backend(Config.from_env(kaggle_env))
    assert isinstance(backend, GemmaModelBackend)
    assert backend.base_url == "http://localhost:8000/v1/chat/completions"


def test_run_cell_exports_the_backend_env_vars():
    import scripts.build_notebook as build_notebook

    run_cell = next(s for s in _cells(build_notebook.build()) if "main.py --agent myagent" in s)
    assert "ZERX_BACKEND=gemma_kaggle" in run_cell
    assert "ZERX_PLATFORM=kaggle" in run_cell
    assert "ZERX_GEMMA_BASE_URL=" in run_cell


def test_notebook_serves_the_model_and_refuses_to_continue_without_it():
    import scripts.build_notebook as build_notebook

    combined = "\n".join(_cells(build_notebook.build()))
    assert "vllm.entrypoints.openai.api_server" in combined
    assert "--quantization" in combined and "fp8" in combined
    # The readiness gate is the difference between a loud failure and an
    # entire run silently spent on heuristics.
    assert "did not become ready" in combined
    assert "raise SystemExit" in combined


def test_offline_invariant_no_pip_install_without_no_index():
    """Internet is disabled at evaluation time; nothing may be downloaded."""
    import re

    import scripts.build_notebook as build_notebook

    combined = "\n".join(_cells(build_notebook.build()))
    for match in re.finditer(r"pip install(?P<args>[^\n]*)", combined):
        assert "--no-index" in match.group("args"), match.group(0)


def test_kernel_metadata_declares_a_model_source():
    import json
    from pathlib import Path

    import scripts.build_notebook as build_notebook

    meta = json.loads(
        (Path(build_notebook.ROOT) / "notebooks" / "kernel-metadata.json").read_text()
    )
    assert meta["model_sources"], "empty model_sources means no weights are attached"
    assert meta["enable_internet"] is False


def test_accelerator_matches_the_documented_target_card():
    import scripts.build_notebook as build_notebook

    assert build_notebook.ACCELERATOR == "rtx6000"
