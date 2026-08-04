"""Tests for `scripts/build_notebook.py` (final whole-branch review Fix 2):
Kaggle submission packaging must bundle `zerx/` (Task 14 made
`agent/my_agent.py` import from `zerx.*`, but nothing ever bundled the
package itself — a real Kaggle run would fail with `ModuleNotFoundError`,
since internet is disabled at eval time and there is no pip rescue), and
must never bundle `zerx/backends/` (the Cerebras dev-only module), gated by
a build-time secret scan.

`scripts/build_notebook.py` is a script, not a package module — imported
here the same way `scripts/play_local.py` and this script's own `main()`
expect to be run: with the repo root as the working/import context.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_notebook  # noqa: E402

from zerx.secret_scan import scan_for_secrets  # noqa: E402


def _all_cell_sources(notebook: dict) -> str:
    return "\n".join(cell.get("source", "") for cell in notebook["cells"])


def test_zerx_bundle_files_excludes_backends_subdirectory():
    """The glob must be non-recursive (`zerx/*.py`, not `zerx/**/*.py`), so
    `zerx/backends/cerebras_dev.py` is never a candidate in the first place
    — verify this directly rather than assuming the glob pattern behaves.
    """
    names = {p.name for p in build_notebook.zerx_bundle_files()}
    assert "cerebras_dev.py" in {
        p.name for p in (ROOT / "zerx" / "backends").glob("*.py")
    }  # sanity: the excluded file really exists on disk
    assert "cerebras_dev.py" not in names
    assert all(p.parent.name == "zerx" for p in build_notebook.zerx_bundle_files())


def test_zerx_bundle_files_enumerated_programmatically_includes_known_modules():
    names = {p.name for p in build_notebook.zerx_bundle_files()}
    # A hand-typed list would drift; this just confirms the current known
    # top-level modules are actually picked up by the glob.
    for expected in ("config.py", "policy.py", "heuristics.py", "transitions.py"):
        assert expected in names


def test_build_bundles_recognizable_source_from_at_least_two_zerx_modules():
    notebook = build_notebook.build()
    combined = _all_cell_sources(notebook)
    assert "class Config" in combined  # zerx/config.py
    assert "class ClickCandidate" in combined  # zerx/heuristics.py


def test_build_never_bundles_cerebras_backend_content():
    """Confirm `zerx/backends/` (the Cerebras module) was never bundled.

    Note: `zerx/secret_scan.py` IS legitimately bundled (it's a top-level
    zerx module) and its own body necessarily contains the literal strings
    "CEREBRAS_API_KEY" / "api.cerebras.ai" as its own detection
    patterns/descriptions — that is the scanner naming what it looks for,
    not a leaked credential. So this test checks the actual thing that
    matters (the real backends/cerebras_dev.py body is absent, and no
    *other* cell leaks these substrings) rather than a bare substring
    check that would false-positive on secret_scan.py's own source.
    """
    notebook = build_notebook.build()
    cells = notebook["cells"]
    combined = _all_cell_sources(notebook)

    backend_src = (ROOT / "zerx" / "backends" / "cerebras_dev.py").read_text()
    assert backend_src not in combined

    for cell in cells:
        source = cell.get("source", "")
        if cell["cell_type"] == "code" and source.startswith(
            "%%writefile /tmp/zerx/secret_scan.py"
        ):
            continue  # the scanner's own pattern definitions, not a leak
        assert "CEREBRAS_API_KEY" not in source
        assert "api.cerebras.ai" not in source


def test_build_writes_zerx_files_before_agent_cell_and_copies_zerx_in_run_cell():
    notebook = build_notebook.build()
    cells = notebook["cells"]
    zerx_cell_indices = [
        i
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "code" and cell["source"].startswith("%%writefile /tmp/zerx/")
    ]
    agent_cell_index = next(
        i
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "code" and cell["source"].startswith("%%writefile /tmp/my_agent.py")
    )
    assert zerx_cell_indices, "expected at least one %%writefile /tmp/zerx/... cell"
    assert max(zerx_cell_indices) < agent_cell_index

    run_cell_source = next(
        cell["source"]
        for cell in cells
        if cell["cell_type"] == "code" and "KAGGLE_IS_COMPETITION_RERUN" in cell["source"]
    )
    assert "cp -r /tmp/zerx" in run_cell_source
    assert "/kaggle/working/ARC-AGI-3-Agents/zerx" in run_cell_source


def test_secret_scan_gate_mechanism_fires_on_a_planted_fake_secret():
    """Prove the gate would actually fire if it needed to — real bundled
    content never contains a Cerebras credential (covered above), so this
    exercises the underlying `scan_for_secrets` call point directly with a
    planted fake secret.
    """
    findings = scan_for_secrets("CEREBRAS_API_KEY = 'x'")
    assert findings

    findings_url = scan_for_secrets("endpoint = 'https://api.cerebras.ai/v1/chat/completions'")
    assert findings_url


def test_secret_scan_exemption_is_scoped_to_secret_scan_py_only(monkeypatch):
    """The gate strips the two KNOWN pattern-literal strings
    ("CEREBRAS_API_KEY", "api.cerebras.ai") from secret_scan.py's own body
    ONLY (final whole-branch review Fix, narrowed from exempting the whole
    file) — every OTHER bundled zerx file must still be scanned verbatim.
    Plant a real-shaped leak in a different top-level module (budget.py)
    and confirm build() still raises SystemExit; the exemption must not
    accidentally widen to cover it.
    """
    real_budget = ROOT / "zerx" / "budget.py"
    poisoned_text = real_budget.read_text() + '\nCEREBRAS_API_KEY = "sk-leaked-for-real"\n'

    original_read_text = Path.read_text

    def _patched_read_text(self, *args, **kwargs):
        if self == real_budget:
            return poisoned_text
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _patched_read_text)

    try:
        build_notebook.build()
    except SystemExit as exc:
        assert "CEREBRAS_API_KEY" in str(exc)
    else:
        raise AssertionError("expected build() to raise SystemExit on the planted secret")


def test_secret_scan_gate_raises_systemexit_when_findings_present(monkeypatch):
    """Exercise build()'s actual gate call point (not just scan_for_secrets
    in isolation): monkeypatch AGENT_SRC to a planted fake secret and
    confirm build() raises SystemExit instead of silently producing a
    notebook.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        fake_agent = Path(tmp) / "my_agent.py"
        fake_agent.write_text("CEREBRAS_API_KEY = 'leaked'\n")
        monkeypatch.setattr(build_notebook, "AGENT_SRC", fake_agent)
        try:
            build_notebook.build()
            assert False, "expected SystemExit from the secret scan gate"
        except SystemExit as exc:
            assert "secret scan failed" in str(exc)
