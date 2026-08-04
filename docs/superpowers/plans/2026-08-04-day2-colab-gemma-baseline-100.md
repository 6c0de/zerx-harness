# Day 2: Colab Gemma-4-31B Load + baseline-100 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `GemmaModelBackend.generate()` for real, generate a Colab
notebook that loads the exact competition-attached Gemma-4-31B model on an
A100/L4 GPU and runs one model-in-loop smoke game, and record the result as
`baseline-100` — the Day 2 exit condition from `docs/TEAM_WORKFLOW.md`.

**Architecture:** `GemmaModelBackend` becomes an HTTP client to a local
vLLM OpenAI-compatible chat-completions server — the same proven pattern
`STRATEGY.md` §3.1 documents ReKi using (`milestone1-2nd-solution.ipynb`
starts a local vLLM server for Gemma-4-31B) — via an injected `http_post`
callable, structurally identical to `zerx/backends/cerebras_dev.py`'s
`CerebrasDevBackend`, so it stays fully unit-testable on this machine
without a GPU. `scripts/build_colab_notebook.py` (mirrors the existing
`scripts/build_notebook.py` pattern) programmatically generates
`notebooks/colab_gemma_smoke.ipynb`, which the human uploads to Colab and
runs on an attached A100/L4 runtime — GPU execution itself cannot happen
through this session's tools, only the notebook generation and the local
zerx-side backend code are directly testable here.

**Tech Stack:** Python 3.12, pytest (existing), stdlib `urllib`/`json`/`time`
for the HTTP client (matches `cerebras_dev.py`), vLLM (`pip install vllm`,
Colab-side only) serving the exact Kaggle-attached model
`google/gemma-4/Transformers/gemma-4-31b-it` (Apache 2.0, verified live
against `kaggle.com/models/google/gemma-4/Transformers/gemma-4-31b-it` and
the competition's own Models tab — not assumed).

## Global Constraints

- Never load Gemma-4-31B locally on this machine (RTX 4060, insufficient
  VRAM) — see `AGENTS.md`. All GPU execution in this plan happens only in
  the generated Colab notebook, run by the human on Colab Pro A100/L4.
- `GemmaModelBackend`'s local unit tests must never start a real server,
  make a real network call, or import `vllm`/`torch`/`transformers` — same
  discipline as `zerx/backends/cerebras_dev.py`'s tests (Task 9 of the
  local-skeleton plan).
- Record model revision, precision/quantization, backend settings, GPU
  type, and package/runtime versions for every model run (`AGENTS.md`,
  "Configuration and reproducibility").
- Colab notebook must: install pinned versions, clone/check out an exact
  commit, print the resolved environment without secrets, load the exact
  Gemma revision, and save structured results outside ephemeral runtime
  storage (`AGENTS.md`'s "Colab gate"). Never treat Drive as source
  control — code stays in Git; Drive is only for non-secret result
  artifacts.
- `baseline-100` is a Colab result — provisional, not deployment source of
  truth. Document known differences from Kaggle (GPU memory,
  dtype/quantization, model path, internet availability) once results
  exist (`AGENTS.md`).
- This plan does not touch Kaggle in any way — no `make submit`, no
  Kaggle CLI. Out of scope, per `AGENTS.md`'s Kaggle gate.
- No `CEREBRAS_API_KEY` involved anywhere in this plan — this is the
  Gemma-only path, no Cerebras dev-proxy usage.

---

## Task 1: `zerx/model_backend.py` — real `GemmaModelBackend.generate()`

**Files:**
- Modify: `zerx/model_backend.py`
- Modify: `tests/test_model_backend.py` (the existing
  `test_gemma_backend_generate_not_yet_implemented` test asserted
  `NotImplementedError` — that's no longer true after this task, so it
  must be replaced, not left broken)

**Interfaces:**
- Consumes: nothing new — `ModelBackend` protocol shape is unchanged
  (`.generate(prompt: str) -> str`).
- Produces: `GemmaModelBackend(model_revision, base_url="http://localhost:8000/v1/chat/completions", request_timeout_seconds=60.0, max_retries=2, http_post=None)` —
  `http_post` injected exactly like `CerebrasDevBackend` (Task 9 of the
  local-skeleton plan) so tests never make a real network call.
  `.generate(prompt) -> str`, `.last_latency_seconds: Optional[float]`.
  Existing call site `agent/my_agent.py:130`
  (`GemmaModelBackend(self._config.model_revision)`) keeps working
  unchanged — all new parameters have defaults.

- [ ] **Step 1: Write the failing tests**

Replace the existing `test_gemma_backend_generate_not_yet_implemented`
test and add new ones. Open `tests/test_model_backend.py` and replace this
function:

```python
def test_gemma_backend_generate_not_yet_implemented():
    backend = GemmaModelBackend(model_revision="gemma-4-31b-it")
    with pytest.raises(NotImplementedError):
        backend.generate("prompt")
```

with:

```python
def _fake_http_post(response_json, captured=None):
    def _post(url, headers, json_body, timeout):
        if captured is not None:
            captured.append({"url": url, "headers": headers, "json_body": json_body, "timeout": timeout})
        return response_json
    return _post


def _ok_response(text='{"action": "ACTION1"}'):
    return {"choices": [{"message": {"content": text}}]}


def test_gemma_backend_generate_returns_message_content():
    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it",
        http_post=_fake_http_post(_ok_response()),
    )
    assert backend.generate("prompt text") == '{"action": "ACTION1"}'


def test_gemma_backend_generate_records_latency():
    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it",
        http_post=_fake_http_post(_ok_response()),
    )
    backend.generate("prompt text")
    assert backend.last_latency_seconds is not None
    assert backend.last_latency_seconds >= 0.0


def test_gemma_backend_sends_model_revision_and_prompt_in_body():
    captured = []
    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it",
        http_post=_fake_http_post(_ok_response(), captured=captured),
    )
    backend.generate("prompt text")
    assert captured[0]["json_body"]["model"] == "gemma-4-31b-it"
    assert captured[0]["json_body"]["messages"] == [{"role": "user", "content": "prompt text"}]


def test_gemma_backend_uses_configured_base_url():
    captured = []
    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it",
        base_url="http://localhost:9000/v1/chat/completions",
        http_post=_fake_http_post(_ok_response(), captured=captured),
    )
    backend.generate("prompt text")
    assert captured[0]["url"] == "http://localhost:9000/v1/chat/completions"


def test_gemma_backend_retries_on_transient_failure_then_succeeds():
    calls = {"count": 0}

    def flaky_post(url, headers, json_body, timeout):
        calls["count"] += 1
        if calls["count"] < 2:
            raise TimeoutError("simulated transient failure")
        return _ok_response()

    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it", http_post=flaky_post, max_retries=2
    )
    assert backend.generate("prompt text") == '{"action": "ACTION1"}'
    assert calls["count"] == 2


def test_gemma_backend_raises_after_exhausting_retries():
    def always_fails(url, headers, json_body, timeout):
        raise TimeoutError("simulated permanent failure")

    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it", http_post=always_fails, max_retries=1
    )
    with pytest.raises(TimeoutError):
        backend.generate("prompt text")


def test_gemma_backend_module_never_imports_vllm_torch_or_transformers():
    """The whole point of the injected-http_post pattern is that
    zerx/model_backend.py itself stays GPU/model-library-free, exactly
    like zerx/backends/cerebras_dev.py — verify by reading the module's
    own source, not by asserting something that's always true.
    """
    import zerx.model_backend as mb

    source = open(mb.__file__, encoding="utf-8").read()
    assert "import vllm" not in source
    assert "import torch" not in source
    assert "import transformers" not in source
```

- [ ] **Step 2: Run tests to verify the new ones fail**

```bash
.venv\Scripts\pytest.exe tests/test_model_backend.py -v
```
Expected: the 7 new/changed tests FAIL (old behavior asserted
`NotImplementedError`; new ones expect real HTTP-client behavior that
doesn't exist yet). The 5 pre-existing `FakeModelBackend`/
`test_gemma_backend_constructs_without_loading_model` tests still PASS.

- [ ] **Step 3: Implement the real `GemmaModelBackend`**

Replace the existing `GemmaModelBackend` class in `zerx/model_backend.py`
(keep `ModelBackend` and `FakeModelBackend` exactly as they are) with:

```python
"""The only module allowed to load/call the Gemma model. Defines a narrow
Protocol so every other module (and all local tests) can depend on
`ModelBackend` without ever importing a real model. `GemmaModelBackend`
talks to a local vLLM OpenAI-compatible chat-completions server via an
injected `http_post` callable — the same pattern
`zerx/backends/cerebras_dev.py` uses for Cerebras — so this module itself
never imports vllm/torch/transformers and every local test runs without a
GPU or a running server. The real server is started only in
`notebooks/colab_gemma_smoke.ipynb`, on Colab/Kaggle.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol


class ModelBackend(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class FakeModelBackend:
    """Test double: returns scripted responses in order."""

    responses: List[str] = field(default_factory=list)
    _calls: List[str] = field(default_factory=list, init=False)

    def generate(self, prompt: str) -> str:
        self._calls.append(prompt)
        if not self.responses:
            raise RuntimeError("FakeModelBackend: no scripted responses left")
        return self.responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def last_prompt(self) -> str:
        if not self._calls:
            raise RuntimeError("FakeModelBackend: generate() was never called")
        return self._calls[-1]


HttpPost = Callable[[str, dict, dict, float], dict]

_DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"


def _default_http_post(url: str, headers: dict, json_body: dict, timeout: float) -> dict:
    import json
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(json_body).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class GemmaModelBackend:
    """Real backend — talks to a local vLLM OpenAI-compatible server
    serving Gemma-4-31B (Kaggle model handle
    `google/gemma-4/Transformers/gemma-4-31b-it`, Apache 2.0). Constructed
    and exercised with a fake `http_post` in local unit tests; the real
    vLLM server is started only in `notebooks/colab_gemma_smoke.ipynb`.
    """

    def __init__(
        self,
        model_revision: str,
        base_url: str = _DEFAULT_BASE_URL,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 2,
        http_post: Optional[HttpPost] = None,
    ) -> None:
        self.model_revision = model_revision
        self.base_url = base_url
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self._http_post = http_post if http_post is not None else _default_http_post
        self.last_latency_seconds: Optional[float] = None

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        json_body = {
            "model": self.model_revision,
            "messages": [{"role": "user", "content": prompt}],
        }
        last_error: Optional[Exception] = None
        for _ in range(self.max_retries):
            start = time.monotonic()
            try:
                response = self._http_post(
                    self.base_url, headers, json_body, self.request_timeout_seconds
                )
                self.last_latency_seconds = time.monotonic() - start
                return response["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised if exhausted
                last_error = exc
        assert last_error is not None
        raise last_error
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv\Scripts\pytest.exe tests/test_model_backend.py -v
```
Expected: 12 passed (5 pre-existing `FakeModelBackend`/constructor tests +
7 new/changed `GemmaModelBackend` tests).

- [ ] **Step 5: Run the full suite**

```bash
.venv\Scripts\pytest.exe tests/ -v
```
Expected: all tests still pass (114 from the local-skeleton plan + fixes,
plus the net change from this task — 1 test replaced, 6 added, so 119
total). `agent/my_agent.py` needs no changes — it already constructs
`GemmaModelBackend(self._config.model_revision)` positionally, which the
new signature still accepts.

- [ ] **Step 6: Commit**

```bash
git add zerx/model_backend.py tests/test_model_backend.py
git commit -m "feat(zerx): implement GemmaModelBackend as an injectable vLLM HTTP client"
```

---

## Task 2: `scripts/build_colab_notebook.py` — generate the Colab smoke-test notebook

**Files:**
- Create: `scripts/build_colab_notebook.py`
- Test: `tests/test_build_colab_notebook.py`
- Modify: `.gitignore` (add `notebooks/colab_gemma_smoke.ipynb` — a
  generated artifact, same treatment as `notebooks/submission.ipynb`)

**Interfaces:**
- Consumes: nothing from other tasks (standalone script, mirrors
  `scripts/build_notebook.py`'s existing pattern in this repo).
- Produces: `build() -> dict` (the notebook JSON structure), `main()`
  (writes it to `notebooks/colab_gemma_smoke.ipynb`).

This notebook is a **development tool**, not the Kaggle submission
notebook — it never touches Kaggle, never imports `agent/my_agent.py`'s
splicing logic, and its only job is satisfying `AGENTS.md`'s Colab gate:
install pinned versions, check out an exact commit, print the resolved
environment (no secrets), load the exact Gemma revision, run one local
public game, save structured results outside ephemeral runtime storage.

- [ ] **Step 1: Write the failing tests**

`tests/test_build_colab_notebook.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv\Scripts\pytest.exe tests/test_build_colab_notebook.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'build_colab_notebook'`.

- [ ] **Step 3: Implement `scripts/build_colab_notebook.py`**

```python
"""Generates notebooks/colab_gemma_smoke.ipynb — a Colab development
notebook (NOT the Kaggle submission notebook; that's
scripts/build_notebook.py) that satisfies AGENTS.md's Colab gate: pinned
installs, exact-commit checkout, environment print without secrets, exact
Gemma revision load, one local public-game smoke run, structured results
saved outside ephemeral Colab runtime storage.

Upload the generated .ipynb to Colab (colab.research.google.com > File >
Upload notebook), attach an A100 or L4 GPU runtime (Runtime > Change
runtime type), and run all cells. Results are written to Google Drive as
JSON — download or copy them back into this repo's
docs/superpowers/experiments/baseline-100.md (see Task 3 of this plan).
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "colab_gemma_smoke.ipynb"

# Pinned to match this repo's local venv (docs/superpowers/experiments/baseline-000.md)
# plus vLLM for serving the model. Bump deliberately, record the change.
PINNED_INSTALL = dedent(
    """\
    !pip install -q "arc-agi>=0.9.6" python-dotenv
    !pip install -q "vllm==0.11.0"
    """
)


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    intro_cell = markdown_cell(
        "# Day 2 — Colab Gemma-4-31B smoke test\n\n"
        "Development notebook, not the Kaggle submission (see "
        "`scripts/build_notebook.py` for that). Attach an A100 or L4 GPU "
        "runtime before running (Runtime > Change runtime type).\n\n"
        "1. Install pinned deps + vLLM\n"
        "2. Clone this repo at the exact commit and check out `zerx/`\n"
        "3. Print the resolved environment (GPU, package versions — no secrets)\n"
        "4. Start a local vLLM server for `google/gemma-4/Transformers/gemma-4-31b-it`\n"
        "5. Run one local public game with `GemmaModelBackend` wired in\n"
        "6. Save structured results to Google Drive (outside ephemeral runtime storage)"
    )

    install_cell = code_cell(PINNED_INSTALL)

    checkout_cell = code_cell(
        dedent(
            """\
            # Fill in the exact commit SHA you're validating (git log --oneline -1 locally).
            REPO_URL = "https://github.com/YOUR_ORG/YOUR_REPO.git"  # replace with this repo's real remote
            COMMIT_SHA = "REPLACE_WITH_EXACT_COMMIT_SHA"

            !git clone $REPO_URL repo
            %cd repo
            !git checkout $COMMIT_SHA
            !python3.12 -m pip install -q -r requirements-zerx.txt
            """
        )
    )

    env_print_cell = code_cell(
        dedent(
            """\
            import subprocess, sys, pkgutil
            print("Python:", sys.version)
            print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
            for pkg in ("vllm", "torch", "arc-agi"):
                spec = pkgutil.find_loader(pkg.replace("-", "_"))
                print(pkg, "installed:", spec is not None)
            # Deliberately never prints CEREBRAS_API_KEY / KAGGLE_API_TOKEN / any secret —
            # this backend never reads them; only confirms GPU + package versions.
            """
        )
    )

    start_vllm_cell = code_cell(
        dedent(
            """\
            import subprocess, time

            vllm_proc = subprocess.Popen([
                "python3.12", "-m", "vllm.entrypoints.openai.api_server",
                "--model", "google/gemma-4/Transformers/gemma-4-31b-it",
                "--served-model-name", "gemma-4-31b-it",
                "--port", "8000",
                # Precision/quantization: record whatever actually loads successfully on
                # this GPU tier (A100 preferred; L4 needs separately verified quantization
                # per STRATEGY.md) — bf16 shown as the A100 starting point.
                "--dtype", "bfloat16",
            ])
            # Wait for the server to report ready before the smoke game below runs.
            import urllib.request
            for _ in range(60):
                try:
                    urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
                    print("vLLM server ready")
                    break
                except Exception:
                    time.sleep(5)
            else:
                raise SystemExit("vLLM server did not become ready in time")
            """
        )
    )

    smoke_game_cell = code_cell(
        dedent(
            """\
            import os
            os.environ["ZERX_BACKEND"] = "gemma_local"
            os.environ["ZERX_PLATFORM"] = "colab"
            os.environ["ZERX_MODEL_REVISION"] = "gemma-4-31b-it"

            # play_local.py loads agent/my_agent.py, which constructs
            # GemmaModelBackend(self._config.model_revision) — pointed at the vLLM
            # server just started above via the default base_url (localhost:8000).
            !python3.12 scripts/play_local.py --game ls20 --max-steps 50
            """
        )
    )

    save_results_cell = code_cell(
        dedent(
            """\
            import json, subprocess
            from google.colab import drive

            drive.mount("/content/drive")

            result = {
                "experiment_id": "baseline-100",
                "model_revision": "gemma-4-31b-it",
                "base_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True
                ).stdout.strip(),
                "gpu": subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True,
                ).stdout.strip(),
                "dtype": "bfloat16",
                "game_id": "ls20",
            }
            out_path = "/content/drive/MyDrive/zerx-baseline-100-result.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print("Saved:", out_path)
            """
        )
    )

    return {
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU",
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            intro_cell,
            install_cell,
            checkout_cell,
            env_print_cell,
            start_vllm_cell,
            smoke_game_cell,
            save_results_cell,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"[build_colab_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv\Scripts\pytest.exe tests/test_build_colab_notebook.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Generate the notebook and eyeball it**

```bash
.venv\Scripts\python.exe scripts\build_colab_notebook.py
```
Expected: `notebooks/colab_gemma_smoke.ipynb` written. Open it (it's JSON)
and confirm the cells read sensibly — this file is gitignored (Step 6
below), so generating it is safe and doesn't need to be undone.

- [ ] **Step 6: Add the generated notebook to `.gitignore`**

Add this line to `.gitignore` (near the existing
`notebooks/submission.ipynb` entry):
```
notebooks/colab_gemma_smoke.ipynb
```

- [ ] **Step 7: Run the full suite**

```bash
.venv\Scripts\pytest.exe tests/ -v
```
Expected: all tests pass (119 from Task 1 + 9 new = 128).

- [ ] **Step 8: Commit**

```bash
git add scripts/build_colab_notebook.py tests/test_build_colab_notebook.py .gitignore
git commit -m "feat(scripts): generate a Colab Gemma-4-31B smoke-test notebook"
```

---

## Task 3: Run the Colab notebook and record `baseline-100`

**Files:**
- Create: `docs/superpowers/experiments/baseline-100.md`

**Interfaces:**
- Consumes: `eval/run_ablation.py`'s `ExperimentRecord`, `write_records()`
  (already built, local-skeleton plan Task 15).
- Produces: a filled-in `baseline-100.md` (no placeholders) and one
  appended JSONL record.

This task is **human-executed on Colab**, not runnable through this
session's tools — no GPU is available here. The steps below are exactly
what the human running the notebook needs to do and report back; nothing
here is a stand-in for actually running it.

- [ ] **Step 1: Generate and open the notebook**

```bash
.venv\Scripts\python.exe scripts\build_colab_notebook.py
```
Open `notebooks/colab_gemma_smoke.ipynb`, upload it to
[colab.research.google.com](https://colab.research.google.com) (File >
Upload notebook), and fill in the two placeholders in the "checkout" cell:
`REPO_URL` (this repo's real remote — if it isn't pushed anywhere yet,
push `day1-local-skeleton`/`day2-colab-gemma-baseline-100` first, or swap
the clone step for uploading a zip) and `COMMIT_SHA` (`git log --oneline -1`
on the branch you're validating).

- [ ] **Step 2: Attach an A100 or L4 runtime**

Runtime > Change runtime type > Hardware accelerator > GPU. Pick A100 if
available on the Colab Pro plan (preferred per `AGENTS.md`); L4 requires
separately verifying the quantization/dtype in the "start vLLM" cell
actually fits L4's smaller VRAM (drop `--dtype bfloat16` to an 8-bit/4-bit
quantized load if it OOMs — record whichever actually works).

- [ ] **Step 3: Run all cells top to bottom**

Expected, in order: pinned installs succeed; repo clones and checks out
the exact commit; the environment cell prints Python version, `nvidia-smi`
GPU info, and confirms `vllm`/`torch`/`arc-agi` are installed (no secrets
printed — confirm this by eye); the vLLM server cell prints "vLLM server
ready" within the 5-minute wait window; the smoke-game cell runs `ls20`
for up to 50 actions without a Python exception (a `NotImplementedError`
must NOT appear anymore — that would mean `GemmaModelBackend` isn't
actually wired to a live server); the results cell mounts Drive and saves
`zerx-baseline-100-result.json`.

If any cell fails: this is real, unscripted debugging — a wrong package
version, an OOM on the model load, a `GemmaModelBackend` request that
doesn't match vLLM's actual OpenAI-compatible response shape, etc. Record
whatever the actual failure was; don't guess a fix without seeing the real
error, per this project's "verify against reality" practice established in
Task 1 of the local-skeleton plan.

- [ ] **Step 4: Bring the results back into the repo**

Download `zerx-baseline-100-result.json` from Drive (or copy its
contents). Write `docs/superpowers/experiments/baseline-100.md` with this
structure, filled in with the real values from the notebook run (no
placeholders left in the committed file):

```markdown
# baseline-100 — first Gemma-4-31B model-in-loop smoke game

- Date: <today's date>
- Base commit: <the COMMIT_SHA the notebook actually checked out>
- Model: google/gemma-4/Transformers/gemma-4-31b-it (Apache 2.0)
- GPU: <nvidia-smi output from the notebook's environment cell>
- Precision/dtype: <whatever actually loaded successfully>
- Game: ls20, max-steps 50
- Backend: gemma_local (zerx/model_backend.py's GemmaModelBackend, vLLM OpenAI-compatible server on localhost:8000)
- Result: <state, levels_completed, actions taken, any exceptions hit and how they were resolved>
- Conclusion: <keep, revert, or investigate — per STRATEGY.md §7.1's rubric>
```

- [ ] **Step 5: Append the machine-readable experiment record**

```bash
.venv\Scripts\python.exe -c "
from pathlib import Path
from zerx.config import Config
from eval.run_ablation import ExperimentRecord, write_records

cfg = Config(experiment_id='baseline-100', backend='gemma_local', platform='colab', model_revision='gemma-4-31b-it')
record = ExperimentRecord(
    experiment_id='baseline-100',
    config_hash=cfg.config_hash(),
    game_id='ls20',
    actions_taken=50,  # replace with the real actions-taken count from the run
    levels_completed=0,  # replace with the real value
    rhae=None,  # local public-game runs are not RHAE-scored; leave None
    wall_time_seconds=0.0,  # replace with the real wall time
    invalid_outputs=0,  # replace with real counts if any repairs/fallbacks fired
    repairs=0,
    fallbacks=0,
    resets=0,
    exceptions=0,
)
write_records([record], Path('docs/superpowers/experiments/baseline-100.jsonl'))
print('wrote docs/superpowers/experiments/baseline-100.jsonl')
"
```
Expected: `docs/superpowers/experiments/baseline-100.jsonl` created with
one real (not placeholder) JSON line, using the actual numbers from
Step 4.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/experiments/baseline-100.md docs/superpowers/experiments/baseline-100.jsonl
git commit -m "docs(experiments): record baseline-100 (first Gemma-4-31B model-in-loop smoke game)"
```

---

## What this plan does not cover

- Any Kaggle interaction (`make submit`, packaging, official submission) —
  separate, explicitly-approved step per `AGENTS.md`'s Kaggle gate, not
  part of this plan.
- Tuning prompts, perception format, or memory/heuristic behavior based on
  the `baseline-100` result — that's Day 3 territory
  (`docs/TEAM_WORKFLOW.md`), and depends on what `baseline-100` actually
  shows.
- Cerebras dev-proxy comparisons — separate, optional, requires the
  user's own `CEREBRAS_API_KEY`, never part of this plan's default path.
- Everything in `STRATEGY.md`'s §7 ladder past `baseline-100`
  (`baseline-110` is already done; `baseline-115`/`120`/`125`/`130`,
  `exp-140`/`150`/`200`+) — not scaffolded here, per the standing
  instruction not to start those without being asked.
