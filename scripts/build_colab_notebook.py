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
    #
    # Plain `pip install vllm` pulls vLLM's default CUDA-12.9-compiled
    # binary regardless of the actual driver's CUDA version -- real Colab
    # run (2026-08-04, driver reporting CUDA 13.0) hit exactly this
    # mismatch: "ImportError: libcudart.so.13: cannot open shared object
    # file". Per vLLM's own install docs (docs.vllm.ai, GPU install page),
    # use `uv pip install --torch-backend=auto` instead, which detects the
    # installed driver's CUDA version and selects a matching build.
    #
    # --torch-backend=auto alone was NOT enough (real Colab run,
    # 2026-08-04): the SAME libcudart.so.13 error recurred, because Colab
    # ships a pre-existing torch install that pip/uv treats as "already
    # satisfies the requirement" and leaves untouched, so vLLM's freshly
    # installed CUDA-13-targeted extension ends up paired with whatever
    # (possibly differently-CUDA-linked) torch Colab already had --
    # exactly the "binary incompatibility with other CUDA versions" vLLM's
    # own docs warn about, recommending "a fresh new environment". `pip`
    # cannot create a venv inside a running Colab kernel, so the practical
    # equivalent is `--reinstall`: force uv to replace the pre-existing
    # torch/vllm state instead of silently trusting it.
    !pip install -q uv
    !uv pip install -q --system --reinstall "vllm==0.26.0" --torch-backend=auto
    """
)

# baseline-120-reki-core's real-game validation sample (see
# docs/superpowers/plans/parallel-baseline-120/README.md's "concrete,
# empirical finding" and docs/superpowers/experiments/baseline-120.md for
# the full game-list/wall-clock justification). Keeps the existing
# ls20+vc33 precedent (baseline-100's smoke game plus this plan's own
# measured "before" reference) and adds 6 more games spread across the
# documented 25-game public list, for per-game regression coverage per
# AGENTS.md's "repeated seeds/configurations" language.
GAME_SAMPLE = ["ls20", "vc33", "su15", "tn36", "ka59", "lf52", "tr87", "sc25"]
# Deliberately below play_local.py's 200-step default: 8 games x 200 steps
# risked exceeding a single Colab session at an unmeasured 31B
# per-decision latency -- see the experiment doc's wall-clock trade-off.
MAX_STEPS_PER_GAME = 100


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
        "# baseline-120 — Colab Gemma-4-31B real-game validation\n\n"
        "Extends Day 2's one-game smoke test "
        "(`docs/superpowers/experiments/baseline-100.md`) into a real, "
        "scored `baseline-120-reki-core` validation run across a "
        "documented multi-game sample. Development notebook, not the "
        "Kaggle submission (see `scripts/build_notebook.py` for that). "
        "Attach an A100 or L4 GPU runtime before running (Runtime > Change "
        "runtime type).\n\n"
        "1. Install pinned deps + vLLM\n"
        "2. Clone this repo at the exact commit and check out `zerx/`\n"
        "3. Print the resolved environment (GPU, package versions — no secrets)\n"
        "4. Start a local vLLM server for `google/gemma-4-31B-it`\n"
        "5. Play each game in `GAME_SAMPLE` directly via `MyAgent`, capped at "
        "`MAX_STEPS_PER_GAME` actions each\n"
        "6. Save each game's real outcome (state, levels completed, actions) "
        "plus its RHAE from `arc.get_scorecard()` to Google Drive (outside "
        "ephemeral runtime storage)"
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
            # nvidia-smi's "CUDA Version" is the DRIVER's max-supported ceiling,
            # not what torch/vllm actually linked against -- that mismatch is
            # exactly what caused two rounds of misdiagnosis on 2026-08-04. Print
            # torch's own resolved CUDA build directly so a future libcudart-style
            # failure is diagnosable from this cell alone, not from guessing.
            try:
                import torch
                print("torch.__version__:", torch.__version__)
                print("torch.version.cuda:", torch.version.cuda)
                print("torch.cuda.is_available():", torch.cuda.is_available())
            except Exception as exc:
                print("torch CUDA introspection failed:", exc)
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
            # Precision/quantization: this Colab runtime (A100-SXM4-80GB, confirmed
            # 2026-08-04, see docs/HANDOFF.md) has enough VRAM to load bf16 (~61.4GB
            # weights) directly. We deliberately do NOT do that here. Kaggle's RTX
            # Pro 6000 (48GB) cannot fit bf16 (61.4GB > 48GB) and must run quantized;
            # per the human owner's 2026-08-06 decision (docs/HANDOFF.md), Colab
            # mirrors Kaggle's precision instead of using whatever the Colab card
            # happens to have headroom for, so a Colab validation result actually
            # reflects the model Kaggle will run, not a strictly-more-accurate one
            # (AGENTS.md/docs/TEAM_WORKFLOW.md: Kaggle is the deployment source of
            # truth; Colab results are provisional and must be comparable to it).
            #
            # 8-bit, not 4-bit: vLLM's bitsandbytes in-flight quantization only
            # supports 4-bit (nf4) from an unquantized checkpoint -- confirmed
            # against vLLM's own docs (docs.vllm.ai/en/stable/features/quantization/
            # bnb/, fetched 2026-08-06), which document exactly one in-flight mode
            # ("load as 4bit quantization") and no 8-bit equivalent. The real 8-bit
            # path is vLLM's dynamic FP8 quantization (--quantization fp8): weights
            # quantized to FP8_E4M3 (~1 byte/param, ~31GB total) with a per-tensor
            # scale computed at load time, no calibration data needed (vLLM's FP8
            # W8A8 docs, docs.vllm.ai/en/latest/features/quantization/llm_compressor/
            # fp8/, fetched 2026-08-06). Ampere (A100, compute capability 8.0) is
            # below the >=8.9 threshold for full W8A8, so it runs FP8 as weight-only
            # W8A16 via the FP8 Marlin kernel -- correct weights/memory footprint,
            # but the docs note "latency improvements are limited in this mode" on
            # this card; Kaggle's RTX Pro 6000 (Blackwell, likely >=8.9) may see real
            # W8A8 speedup instead. That inference-speed difference is expected and
            # does not break the precision parity this switch is actually for.
            # bitsandbytes is no longer installed above -- FP8 quantization is
            # native to vLLM and needs no extra package.
            VLLM_LOG_PATH = "/content/vllm_server.log"
            vllm_log = open(VLLM_LOG_PATH, "w")
            vllm_proc = subprocess.Popen(
                [
                    "python3.12", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", "google/gemma-4-31B-it",
                    "--served-model-name", "gemma-4-31b-it",
                    "--port", "8000",
                    "--quantization", "fp8",
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
        f'GAME_SAMPLE = {json.dumps(GAME_SAMPLE)}\n'
        f'MAX_STEPS_PER_GAME = {MAX_STEPS_PER_GAME}\n'
        + dedent(
            """
            import os
            import sys
            import time
            import importlib.util

            os.environ["ZERX_BACKEND"] = "gemma_local"
            os.environ["ZERX_PLATFORM"] = "colab"
            os.environ["ZERX_MODEL_REVISION"] = "gemma-4-31b-it"

            sys.path.insert(0, "")
            sys.path.insert(0, "vendor/ARC-AGI-3-Agents")

            import arc_agi
            from arc_agi import OperationMode

            def _load_my_agent_class():
                spec = importlib.util.spec_from_file_location(
                    "user_agent_module", "agent/my_agent.py"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.MyAgent

            # Loading agent/my_agent.py here constructs its backend via
            # ZERX_BACKEND=gemma_local -- once Track 1's select_backend
            # lands this resolves through it to
            # GemmaModelBackend(self._config.model_revision), pointed at
            # the vLLM server started above via its default base_url
            # (localhost:8000). Before Track 1 lands, agent/my_agent.py's
            # __init__ hardcodes GemmaModelBackend directly regardless of
            # Config.backend -- gemma_local already resolves correctly
            # either way (see docs/superpowers/plans/parallel-baseline-120/
            # README.md's "concrete, empirical finding").
            MyAgentCls = _load_my_agent_class()
            MyAgentCls.MAX_ACTIONS = min(
                getattr(MyAgentCls, "MAX_ACTIONS", MAX_STEPS_PER_GAME),
                MAX_STEPS_PER_GAME,
            )

            arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)

            per_game_play_results = []
            for i, game_id in enumerate(GAME_SAMPLE, 1):
                print(f"=== [{i}/{len(GAME_SAMPLE)}] {game_id} ===")
                start = time.monotonic()
                try:
                    env = arc.make(game_id)
                    agent = MyAgentCls(
                        card_id="colab-baseline-120",
                        game_id=game_id,
                        agent_name=f"MyAgent.colab.{game_id}",
                        ROOT_URL="http://localhost",
                        record=False,
                        arc_env=env,
                        tags=["colab", "baseline-120"],
                    )
                    agent.main()
                    final = agent.frames[-1]
                    per_game_play_results.append({
                        "game_id": game_id,
                        "state": str(final.state),
                        "levels_completed": final.levels_completed,
                        "actions": agent.action_counter,
                        "wall_time_seconds": time.monotonic() - start,
                        "exception": None,
                    })
                    print(
                        f"  -> state={final.state}, "
                        f"levels_completed={final.levels_completed}, "
                        f"actions={agent.action_counter}"
                    )
                except Exception as exc:  # noqa: BLE001 - one bad game must not lose the rest
                    per_game_play_results.append({
                        "game_id": game_id,
                        "state": None,
                        "levels_completed": None,
                        "actions": None,
                        "wall_time_seconds": time.monotonic() - start,
                        "exception": repr(exc),
                    })
                    print(f"  -> EXCEPTION: {exc!r}")
            """
        )
    )

    save_results_cell = code_cell(
        dedent(
            """\
            import json
            import subprocess
            from google.colab import drive

            drive.mount("/content/drive")

            scorecard = arc.get_scorecard()

            def _rhae_for(game_id):
                env_score_list = scorecard.find_environment(game_id)
                if env_score_list is None or not env_score_list.runs:
                    return None, "no EnvironmentScoreList found for this game_id"
                latest_run = env_score_list.runs[-1]
                return latest_run.score, latest_run.message

            per_game_full = []
            for entry in per_game_play_results:
                if entry["exception"] is not None:
                    rhae, rhae_message = None, "game raised an exception before scoring"
                else:
                    rhae, rhae_message = _rhae_for(entry["game_id"])
                per_game_full.append({**entry, "rhae": rhae, "rhae_message": rhae_message})

            result = {
                "experiment_id": "baseline-120",
                "model_revision": "gemma-4-31b-it",
                "base_commit": subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True
                ).stdout.strip(),
                "gpu": subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True,
                ).stdout.strip(),
                "dtype": "bfloat16",
                "quantization": "fp8",
                "game_sample": GAME_SAMPLE,
                "max_steps_per_game": MAX_STEPS_PER_GAME,
                "per_game": per_game_full,
                "aggregate_score": scorecard.score,
            }
            out_path = "/content/drive/MyDrive/zerx-baseline-120-result.json"
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
