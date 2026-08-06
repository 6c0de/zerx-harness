"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "rtx6000"

# Internal mapping; don't edit unless Kaggle adds new options.
#
# `name` goes into the notebook's own metadata. Be aware that Kaggle's push
# API **ignores it** — see `push_accelerator_flag()` below and
# docs/superpowers/experiments/kaggle-env-probe.md. The values below are the
# starter's; only the two marked `verified` have actually been observed to
# be honoured by the API.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False, "push": None},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True,  "push": "NvidiaTeslaT4"},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True,  "push": "NvidiaTeslaP100"},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True,  "push": "NvidiaRtxPro6000"},
}

# `machine_shape` values observed to actually be honoured by the push API.
#
# The starter's own accelerator name for the RTX card ("nvidiaRtx6000") is
# NOT one of them: pushing with it yielded a Tesla P100, silently. The real
# string is "NvidiaRtxPro6000", recovered by selecting the accelerator in
# the Kaggle web UI and reading back the server's own metadata with
# `kaggle kernels pull -m` — it appears in neither the starter nor the
# Kaggle SDK's documented list (which knows only NvidiaTeslaT4,
# NvidiaTeslaP100, Tpu1VmV38). See
# docs/superpowers/experiments/kaggle-env-probe.md.
#
# Kept as a warn-list rather than a hard validation: Kaggle can add shapes
# faster than this file learns about them, and refusing to push an unknown
# value would be worse than pushing it with a warning. Silence is the thing
# to avoid — an unrecognised value costs you the default GPU without saying so.
_VERIFIED_ACCELERATORS = {"NvidiaTeslaT4", "NvidiaTeslaP100", "NvidiaRtxPro6000"}


def push_accelerator_flag() -> str:
    """The `--accelerator` argument `kaggle kernels push` needs, or "".

    Selecting an accelerator by writing it into the notebook's
    `metadata.kaggle.accelerator` — which is what this script (and the
    upstream official starter) has always done — **does not work**. Kaggle's
    push API reads exactly one GPU-related key from
    `notebooks/kernel-metadata.json`, the `enable_gpu` bool, and nothing
    reads the notebook's own accelerator field. The accelerator is carried
    only by the CLI flag.

    Measured 2026-08-06 with a controlled pair of runs
    (docs/superpowers/experiments/kaggle-env-probe.md): pushing with
    `--accelerator NvidiaTeslaT4` produced a Tesla T4, while pushing with
    `--accelerator nvidiaRtx6000` produced a Tesla P100 — the default. The
    flag works; that particular value does not, and fails silently.
    """
    return _ACCELERATORS[ACCELERATOR]["push"] or ""

# ─────────────────────────────────────────────────────────────────────────────
# RUN CONFIGURATION
#
# This notebook ships no language model. The scoring rules, measured through a
# real competition-mode gateway (eval/gateway_smoke.py) rather than assumed:
#
#   * `level_score = min(115, (human_baseline / actions_on_that_level)**2 * 100)`,
#     with a level's actions counted from the previous level's completion.
#   * One play per game. `arc_agi/api.py`'s RESET handler refuses to execute a
#     reset when `_action_count == 0`, so exploration cannot be reset away — it
#     is charged to whichever level completes next.
#
# So a level is won or lost in its first ~2-3x the human baseline, and actions
# after that are nearly worthless — which is what `zerx/single_play.py`'s
# careful/reckless split encodes.
#
# The honest score for this agent is **0.1153** on the 25 public games. That is
# far below the leaderboard, and the route being taken for a competitive score
# is the guarded Duck fork (scripts/build_duck_notebook.py, docs/DUCK_FORK.md).
# This notebook remains the deployment path for our own agent.
#
# Every knob is an env var (read by `agent.my_agent._policy_from_env`) so a run
# can be retuned without editing bundled source.
# ─────────────────────────────────────────────────────────────────────────────
ZERX_ENV = {
    # Wall clock, not action count, is the scarce resource: `Swarm` runs one
    # thread per game, so every game is live at once and they share the 9h.
    "ZERX_GAME_SECONDS": "1800",        # per game; games run concurrently
    "ZERX_MAX_ACTIONS": "20000",        # per game, guards against one game hogging
    "ZERX_HARD_ACTION_CAP": "2000000",  # lifts the framework's MAX_ACTIONS=80
    # `careful_budget` is the action count after which the current level is
    # treated as lost and exploration goes unrestrained. Public level-1 human
    # baselines run 17-78 actions; past ~4x the top of that a level scores
    # under 1% however it ends.
    "ZERX_CAREFUL_BUDGET": "220",
    "ZERX_STICKY": "0.7",
    "ZERX_NOISE_FRACTION": "0.35",
    "ZERX_SEED": "0",
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def zerx_bundle_files() -> List[Path]:
    """Every top-level `zerx` module to bundle into the notebook (Task 14
    made `agent/my_agent.py` import from `zerx.*`, but nothing ever bundled
    the package itself — a real Kaggle run would fail with
    `ModuleNotFoundError: No module named 'zerx'`, since internet is
    disabled at eval time).

    Enumerated programmatically — not a hand-typed list — so a future new
    top-level zerx module is picked up automatically. Deliberately
    non-recursive (`glob("*.py")`, not `rglob`): this must NEVER bundle
    `zerx/backends/` (the Cerebras dev-only module) under any
    circumstance, and a flat glob naturally excludes subdirectories.
    """
    return sorted((ROOT / "zerx").glob("*.py"))


def writefile_body(body: str) -> str:
    """Never hand `%%writefile` an empty cell body.

    IPython rejects it outright — "UsageError: %%writefile is a cell magic,
    but the cell body is empty" — and on Kaggle papermill turns that into a
    failed kernel. `zerx/__init__.py` is a legitimately empty file, so the
    submission notebook died on its very first %%writefile cell, long before
    reaching the agent.

    Caught by notebooks/model-smoke (scripts/build_model_smoke_notebook.py) on
    real Kaggle hardware, which is precisely the class of failure that kernel
    exists to find without spending a submission.
    """
    if body.strip():
        return body
    return "# intentionally empty\n"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    zerx_paths = zerx_bundle_files()
    zerx_bodies = [(path, path.read_text()) for path in zerx_paths]

    # Build-time secret scan gate (AGENTS.md: "a packaging test
    # (zerx/secret_scan.py) scans the generated artifact ... and fails the
    # build if found"). Scan the agent body plus every bundled zerx body
    # BEFORE writing anything, so a leak never reaches notebooks/*.ipynb.
    #
    # secret_scan.py's own pattern/description strings literally spell out
    # "CEREBRAS_API_KEY" and "api.cerebras.ai" by construction (it's the
    # detector, not leaked content) — scanning it verbatim would make every
    # build fail permanently on a self-referential false positive. Strip
    # only those two known literals from ITS body before scanning, rather
    # than exempting the whole file, so a real secret accidentally pasted
    # anywhere else in secret_scan.py is still caught.
    sys.path.insert(0, str(ROOT))
    from zerx.secret_scan import scan_for_secrets

    def _scan_text(path: Path, body: str) -> str:
        if path.name == "secret_scan.py":
            return body.replace("CEREBRAS_API_KEY", "").replace("api.cerebras.ai", "")
        return body

    scan_target_bodies = [_scan_text(path, body) for path, body in zerx_bodies]
    combined_source = agent_body + "\n".join(scan_target_bodies)
    findings = scan_for_secrets(combined_source)
    if findings:
        raise SystemExit(f"[build_notebook] secret scan failed: {findings}")

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    # Bundle zerx/*.py (top-level modules only — never zerx/backends/, the
    # Cerebras dev-only module) so `agent/my_agent.py`'s `from zerx.config
    # import Config` etc. resolve on Kaggle, where internet is disabled and
    # there is no pip rescue. Written to /tmp/zerx/, same reasoning as the
    # agent cell below: keep it out of notebook outputs.
    # IPython's `%%writefile` magic calls plain `open(filename, 'w')` with no
    # `os.makedirs` (verified against IPython 9.x `OSMagics.writefile`), so
    # writing to /tmp/zerx/<mod>.py raises FileNotFoundError on a fresh Kaggle
    # kernel, where /tmp/zerx does not exist. Create the directory first.
    mkdir_cell = code_cell(
        "import os\n"
        "os.makedirs('/tmp/zerx', exist_ok=True)"
    )

    zerx_write_cells = [
        code_cell(f"%%writefile /tmp/zerx/{path.name}\n" + writefile_body(body))
        for path, body in zerx_bodies
    ]

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    env_prefix = " ".join(f"{k}={v}" for k, v in ZERX_ENV.items())
    run_cell_source = dedent(
        """        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \
                   /kaggle/working/ARC-AGI-3-Agents

            # Copy the zerx package alongside it so `import zerx` resolves —
            # main.py runs with cwd /kaggle/working/ARC-AGI-3-Agents, which is
            # on sys.path via cwd.
            !mkdir -p /kaggle/working/ARC-AGI-3-Agents && \
                cp -r /tmp/zerx /kaggle/working/ARC-AGI-3-Agents/zerx

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Import gate: a typo or a missing bundled module must fail here,
            # loudly, rather than 9 hours later as a mysteriously empty score.
            # In-process on purpose — IPython swallows a shell command's
            # non-zero exit, so only a real Python exception stops the notebook.
            import sys
            sys.path.insert(0, '/tmp')
            from zerx.single_play import SinglePlayAgent
            print('policy import OK:', SinglePlayAgent.__module__)

            # ZERX_* go on the command line rather than into the .env above:
            # main.py runs in a separate process, and an explicit prefix does
            # not depend on when the framework happens to call load_dotenv().
            !cd /kaggle/working/ARC-AGI-3-Agents && \
                MPLBACKEND=agg \
                ZERX_ENV_PREFIX \
                python main.py --agent myagent
        """
    ).replace("ZERX_ENV_PREFIX", env_prefix)
    run_cell = code_cell(run_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            install_cell,
            mkdir_cell,
            *zerx_write_cells,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU, and so the attached weights can never
    # silently drift away from what the run cell expects.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        changes = []

        wanted_gpu = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted_gpu:
            meta["enable_gpu"] = wanted_gpu
            changes.append(f"enable_gpu={wanted_gpu}")

        # Both are cleared deliberately. The agent loads no model and needs no
        # extra wheels; leaving a stale attachment behind would add minutes of
        # mount time to every run for nothing.
        if meta.get("model_sources") != []:
            meta["model_sources"] = []
            changes.append("model_sources=[]")

        if meta.get("dataset_sources") != []:
            meta["dataset_sources"] = []
            changes.append("dataset_sources=[]")

        if changes:
            METADATA_PATH.write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(f"[build_notebook] Synced {', '.join(changes)} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")

    flag = push_accelerator_flag()
    if flag:
        print(f"[build_notebook] Push with:  kaggle kernels push -p notebooks/ "
              f"--accelerator {flag}")
        if flag not in _VERIFIED_ACCELERATORS:
            print(f"[build_notebook] WARNING: {flag!r} has never been observed to "
                  f"work. Verified values: {sorted(_VERIFIED_ACCELERATORS)}. An "
                  f"unrecognised value is ignored silently and you get the default "
                  f"GPU instead — measured 2026-08-06, see "
                  f"docs/superpowers/experiments/kaggle-env-probe.md.")


if __name__ == "__main__":
    main()
