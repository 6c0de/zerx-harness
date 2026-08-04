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


def _git(*args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _resolve_repo_url() -> str:
    try:
        return _git("remote", "get-url", "origin")
    except Exception:
        return "https://github.com/YOUR_ORG/YOUR_REPO.git"  # no origin configured — fill in by hand


def _resolve_commit_sha() -> str:
    try:
        return _git("rev-parse", "HEAD")
    except Exception:
        return "REPLACE_WITH_EXACT_COMMIT_SHA"  # not a git checkout — fill in by hand

# Pinned to match this repo's local venv (docs/superpowers/experiments/baseline-000.md)
# plus vLLM for serving the model. Bump deliberately, record the change.
PINNED_INSTALL = dedent(
    """\
    !pip install -q "arc-agi>=0.9.6" python-dotenv
    # vllm==0.11.0 (Oct 2025) predates Gemma 4's release (2026-03-26 per its
    # Kaggle model card) by ~5 months and cannot parse its rope_scaling
    # config (p-RoPE) -- real Colab run (2026-08-04) hit exactly this:
    # "rope_scaling should have a 'rope_type' key". Pinned to the latest
    # stable release as of 2026-08-04 instead.
    !pip install -q "vllm==0.26.0"
    !pip install -q "bitsandbytes>=0.43.0"
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
        "4. Start a local vLLM server for `google/gemma-4-31B-it`\n"
        "5. Run one local public game with `GemmaModelBackend` wired in\n"
        "6. Save structured results to Google Drive (outside ephemeral runtime storage)"
    )

    install_cell = code_cell(PINNED_INSTALL)

    checkout_cell = code_cell(
        # REPO_URL/COMMIT_SHA are resolved from this machine's actual git state
        # at generation time (git remote get-url origin / git rev-parse HEAD) —
        # not placeholders. Re-run this script after committing to refresh them.
        f'REPO_URL = "{_resolve_repo_url()}"\n'
        f'COMMIT_SHA = "{_resolve_commit_sha()}"\n'
        + dedent(
            """
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

            # Model identity: the Kaggle Models UI labels this
            # "google/gemma-4/Transformers/gemma-4-31b-it" (owner/model/framework/
            # variant -- Kaggle's own organizational path), but that string is NOT
            # a valid Hugging Face Hub repo id and vLLM/transformers reject it
            # outright (HFValidationError). Real Colab run (2026-08-04) hit exactly
            # this. The actual loadable repo id, confirmed live against
            # huggingface.co/google/gemma-4-31B-it (note capital B -- HF repo ids
            # are case-sensitive) and its own documented `vllm serve` usage
            # snippet, is "google/gemma-4-31B-it".
            #
            # Precision/quantization: 31B dense in bf16/fp16 is ~2 bytes/param ~= 61GB
            # of weights alone -- does NOT fit an A100-SXM4-40GB's 40GB VRAM (confirmed
            # against this notebook's own env-print cell's nvidia-smi output). Load
            # 4-bit (bitsandbytes nf4) instead: ~1/4 the weight footprint (~15GB),
            # leaving headroom for KV cache. Record whatever precision actually loads
            # successfully -- this is the A100 starting point, re-verify for L4 (24GB,
            # even tighter) per STRATEGY.md.
            VLLM_LOG_PATH = "/content/vllm_server.log"
            vllm_log = open(VLLM_LOG_PATH, "w")
            vllm_proc = subprocess.Popen(
                [
                    "python3.12", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", "google/gemma-4-31B-it",
                    "--served-model-name", "gemma-4-31b-it",
                    "--port", "8000",
                    "--quantization", "bitsandbytes",
                    "--load-format", "bitsandbytes",
                    "--dtype", "bfloat16",
                    # Smoke test only needs a short context (64x64 grid + a short
                    # prompt) -- capping this shrinks the KV cache's VRAM footprint.
                    "--max-model-len", "8192",
                    "--gpu-memory-utilization", "0.85",
                ],
                stdout=vllm_log,
                stderr=subprocess.STDOUT,
            )

            # Wait for the server to report ready before the smoke game below runs.
            # A cold 31B load (first-time HF download + quantized-load + CUDA graph
            # warmup) can take well past 5 minutes -- poll for up to 20 minutes, and
            # print the actual server log (not just a bare timeout) if it never comes up,
            # or if the process has already died, so the real error is visible instead
            # of a blind "did not become ready" message.
            import urllib.request

            def _tail_log(n_lines: int = 60) -> str:
                vllm_log.flush()
                with open(VLLM_LOG_PATH) as f:
                    lines = f.readlines()
                return "".join(lines[-n_lines:])

            ready = False
            for i in range(240):
                if vllm_proc.poll() is not None:
                    print(f"vLLM server process exited early with code {vllm_proc.returncode}")
                    print("---- last 60 lines of vllm_server.log ----")
                    print(_tail_log())
                    raise SystemExit("vLLM server process exited before becoming ready")
                try:
                    urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
                    ready = True
                    print("vLLM server ready")
                    break
                except Exception:
                    if i % 12 == 0:  # every ~60s
                        print(f"still waiting on vLLM server ({i * 5}s elapsed)...")
                    time.sleep(5)
            if not ready:
                print("---- last 60 lines of vllm_server.log ----")
                print(_tail_log())
                raise SystemExit(
                    f"vLLM server did not become ready in time; full log at {VLLM_LOG_PATH}"
                )
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
