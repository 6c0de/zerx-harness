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
            # Deliberately never prints any API key or auth token — this backend
            # does not read them; only confirms GPU + package versions.
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
    try:
        display_path = NOTEBOOK_PATH.relative_to(ROOT)
    except ValueError:
        display_path = NOTEBOOK_PATH
    print(f"[build_colab_notebook] Wrote {display_path}")


if __name__ == "__main__":
    main()
