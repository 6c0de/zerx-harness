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

    # agent/my_agent.py does `from agents.agent import Agent` -- that `agents`
    # package is the arcprize/ARC-AGI-3-Agents framework, not a pip package.
    # Locally, `make setup` clones it to vendor/ARC-AGI-3-Agents and
    # scripts/slim_framework.py rewrites its __init__.py to skip the
    # upstream's eager langgraph/langsmith/smolagents imports (deps we never
    # install). This notebook cloned OUR repo above but never cloned the
    # framework itself -- real Colab run confirmed this raises
    # "ModuleNotFoundError: No module named 'agents'" in the smoke-game cell
    # below, since vendor/ARC-AGI-3-Agents simply doesn't exist yet.
    framework_clone_cell = code_cell(
        dedent(
            """\
            !git clone --depth 1 https://github.com/arcprize/ARC-AGI-3-Agents.git vendor/ARC-AGI-3-Agents
            !python3.12 scripts/slim_framework.py
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
            # Precision: bf16, no quantization. This reverses the 2026-08-06 fp8
            # decision, whose premise turned out to be false.
            #
            # That decision reasoned: "Kaggle's RTX Pro 6000 has 48GB VRAM -- bf16
            # (~61.4GB weights) does not fit; it needs quantization regardless of
            # what Colab needs", and set Colab to fp8 so the two sides would run
            # comparable precision. The 48GB figure was never measured. The
            # environment probe measured it (2026-08-06,
            # docs/superpowers/experiments/kaggle-env-probe.md): the card is an
            # RTX PRO 6000 Blackwell Server Edition with 97887 MiB -- ~96GB, not
            # 48GB -- and the attached weights are 62.58GB of bf16.
            #
            # So bf16 fits on both sides, with room to spare:
            #   Kaggle  RTX PRO 6000 Blackwell  ~96GB  vs 62.58GB   (~33GB spare)
            #   Colab   A100-SXM4-80GB           80GB  vs 62.58GB   (~17GB spare)
            #
            # Parity is preserved -- which was the real point of the fp8 decision
            # (AGENTS.md/docs/TEAM_WORKFLOW.md: Kaggle is the deployment source of
            # truth, Colab results are provisional and must be comparable) -- and
            # now at the higher precision rather than the lower one. Quantization
            # was a cost paid for a constraint that does not exist.
            #
            # Kaggle does not serve through vLLM at all: the probe found vllm
            # absent from the image, internet disabled, and no vLLM in the
            # competition's offline wheels, so the submission loads bf16 in-process
            # via transformers (zerx/model_backend.py's TransformersModelBackend).
            # Colab keeps vLLM because it can install it, and because a served
            # endpoint is the faster dev loop. The precision matches; the serving
            # mechanism deliberately does not, and that difference is recorded
            # rather than hidden.
            VLLM_LOG_PATH = "/content/vllm_server.log"
            vllm_log = open(VLLM_LOG_PATH, "w")
            vllm_proc = subprocess.Popen(
                [
                    "python3.12", "-m", "vllm.entrypoints.openai.api_server",
                    "--model", "google/gemma-4-31B-it",
                    "--served-model-name", "gemma-4-31b-it",
                    "--port", "8000",
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
            # Per-decision trace (source: model/heuristic/fallback_*, raw
            # model response, model_error) -- MyAgent.__init__ wires this
            # automatically off Config.trace_export_path (zerx/trace.py).
            # A directory, not a file: JsonlTraceWriter auto-names
            # <dir>/<game_id>-<timestamp>.jsonl per game (see its own
            # docstring) since this cell constructs a fresh MyAgent per
            # game in GAME_SAMPLE. Without this, a 0.0 result is
            # unexplainable after the fact -- no visibility into whether
            # the model was ever actually called successfully.
            os.environ["ZERX_TRACE_EXPORT_PATH"] = "/content/traces/"
            # Diagnostic-run-only override (not a default change): zerx/
            # budget.py's should_favor_execution flips once actions_taken
            # crosses 80% of budget_soft_cap and, when true, zerx/policy.py's
            # decide() skips the model call entirely in favor of the top
            # heuristic click candidate. Config.budget_soft_cap now defaults
            # to 400 (tied to Config.max_actions) rather than the old 50,
            # which used to fire at action 40 and silently turn the back
            # half of every MAX_STEPS_PER_GAME=100 game into heuristic-only
            # play. Pinned well above MAX_STEPS_PER_GAME here anyway so this
            # run measures the model's behavior for the whole game
            # regardless of what the default happens to be.
            os.environ["ZERX_BUDGET_SOFT_CAP"] = "1000"

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
            # The per-game action cap is now a Config field
            # (ZERX_MAX_ACTIONS), which MyAgent.__init__ applies to the
            # instance -- so it overrides anything set on the class here.
            # Set the env var instead. This also supersedes the old
            # min()-against-the-base-class form, which could only ever
            # LOWER the cap below the inherited 80 and never raise it to
            # the MAX_STEPS_PER_GAME actually requested (docs/HANDOFF.md
            # "Known failures or risks" item 7 -- confirmed by a real Colab
            # run capping at 81, not the requested 100).
            os.environ["ZERX_MAX_ACTIONS"] = str(MAX_STEPS_PER_GAME)
            MyAgentCls = _load_my_agent_class()

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
            import os
            import shutil
            import subprocess
            from google.colab import drive

            drive.mount("/content/drive")

            # Copy the per-decision trace files (see smoke_game_cell's
            # ZERX_TRACE_EXPORT_PATH) off ephemeral Colab runtime storage
            # before it's gone -- same reasoning as the result JSON below.
            TRACE_SRC = "/content/traces"
            TRACE_DST = "/content/drive/MyDrive/zerx-baseline-120-traces"
            if os.path.isdir(TRACE_SRC):
                shutil.copytree(TRACE_SRC, TRACE_DST, dirs_exist_ok=True)
                print("Saved traces to:", TRACE_DST)
            else:
                print("No traces directory found at", TRACE_SRC)

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
                "quantization": None,  # bf16 fits on both cards; see the
                # serving cell for why the 2026-08-06 fp8 decision was reversed.
                "game_sample": GAME_SAMPLE,
                "max_steps_per_game": MAX_STEPS_PER_GAME,
                "budget_soft_cap": os.environ.get("ZERX_BUDGET_SOFT_CAP"),
                "trace_dir": TRACE_DST if os.path.isdir(TRACE_SRC) else None,
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
            framework_clone_cell,
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
