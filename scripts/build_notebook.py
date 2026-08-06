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
# MODEL WIRING (docs/superpowers/specs/2026-08-06-kaggle-p0-model-attach-design.md)
#
# Until these were set, a submission ran with no language model at all and
# said nothing about it: kernel-metadata.json had an empty `model_sources`,
# nothing here set ZERX_BACKEND, so `Config.backend` fell to its default
# "fake", `select_backend` returned a `FakeModelBackend` with no scripted
# responses, every generate() raised, and the agent played heuristics-only
# through its own fallback chain. No crash, no error, just a meaningless
# score (docs/HANDOFF.md, ARC-HANDOFF-001).
# ─────────────────────────────────────────────────────────────────────────────
ZERX_BACKEND = "gemma_kaggle"
ZERX_PLATFORM = "kaggle"
ZERX_GEMMA_BASE_URL = "http://localhost:8000/v1/chat/completions"

# Kaggle Models handle for the weights, synced into
# notebooks/kernel-metadata.json's `model_sources` by main() so the two can
# never drift apart. Keep in step with scripts/build_probe_notebook.py's
# MODEL_SOURCE — the probe must attach the same weights the submission does,
# or its mount-path answer does not transfer.
MODEL_SOURCE = "google/gemma-4/transformers/gemma-4-31b-it/1"

# Offline wheels for a transformers new enough to know Gemma 4.
#
# The Kaggle image ships transformers 5.0.0. The checkpoint declares
# `model_type: gemma4` / `Gemma4ForConditionalGeneration` and was saved by
# 5.5.0.dev0, and loading it on 5.0.0 fails outright:
#
#   ValueError: The checkpoint you are trying to load has model type `gemma4`
#   but Transformers does not recognize this architecture.
#
# Gemma 4 support landed in transformers 5.5.0. There is no way around the
# version: the model directory carries no modeling code and `auto_map` is
# null, so `trust_remote_code=True` has nothing to load, and internet is
# disabled at evaluation time. Measured on real Kaggle hardware by
# notebooks/model-smoke — see docs/superpowers/experiments/kaggle-env-probe.md.
#
# 5.5.0 exactly, not the newest release: it is the lowest version that
# supports Gemma 4, and its requirements are all satisfied by what the image
# already has (huggingface-hub 1.11.0 against <2.0,>=1.5.0; tokenizers 0.22.2
# against <=0.23.0,>=0.22.0; safetensors 0.7.0; numpy 2.0.2). A newer
# transformers risks demanding a newer tokenizers or hub, which would mean
# replacing compiled packages that torch 2.10 is built against.
WHEELS_DATASET = "enzeceb/zerx-transformers-wheels"

# Filesystem path the attached Gemma weights mount at, under /kaggle/input.
#
# Deliberately None until Phase B. The Kaggle Models UI label
# ("google/gemma-4/Transformers/gemma-4-31b-it") is an organizational path,
# not a filesystem path — and separately is not a valid Hugging Face repo id
# either (docs/HANDOFF.md, Day 2 item 2), so it cannot simply be guessed.
# `notebooks/probe/probe.ipynb` (scripts/build_probe_notebook.py) reports the
# real path; fill it in from that result.
#
# While it is None the generated notebook still builds — but refuses to play
# on Kaggle, loudly and immediately. That refusal is the point: AGENTS.md
# requires model initialization failures to "fail before gameplay rather than
# degrading an entire evaluation silently", and shipping a submission that
# quietly has no model is the exact failure this whole change exists to end.
KAGGLE_MODEL_DIR: str | None = (
    "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1"
)

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


TRANSFORMERS_INSTALL_SOURCE = dedent(
    '''\
    """Install a transformers new enough to load Gemma 4, entirely offline.

    See WHEELS_DATASET in scripts/build_notebook.py for why this is needed at
    all. Nothing is fetched from the network -- --no-index guarantees it.
    """
    import glob, os, subprocess, sys, zipfile

    # Locate the attached wheels wherever Kaggle mounted them, rather than
    # hardcoding a path: dataset mount points have bitten this project before.
    # Find the wheel itself, anywhere under /kaggle/input, rather than
    # guessing the mount path.
    #
    # Kaggle's layout is not stable across kernels: one kernel mounted this
    # dataset flat at /kaggle/input/zerx-transformers-wheels, another nested it
    # under /kaggle/input/datasets/. A glob written for either shape fails on
    # the other, and each failure costs a run. What does not vary is the
    # filename we actually need.
    TARGET = "transformers-5.5.0-py3-none-any.whl"

    def _find_wheel_dir():
        for root, _dirs, files in os.walk("/kaggle/input"):
            if TARGET in files:
                return root
        return None

    wheel_dir = _find_wheel_dir()

    if wheel_dir is None:
        # Kaggle sometimes keeps a zip-uploaded directory as a .zip rather than
        # expanding it, and --find-links needs real .whl files. Expand any
        # archive that plausibly holds them, then look again.
        archives = glob.glob("/kaggle/input/**/*.zip", recursive=True)
        if archives:
            os.makedirs("/tmp/wheels", exist_ok=True)
            for archive in archives:
                try:
                    with zipfile.ZipFile(archive) as handle:
                        handle.extractall("/tmp/wheels")
                except zipfile.BadZipFile:
                    continue
            if os.path.exists(os.path.join("/tmp/wheels", TARGET)):
                wheel_dir = "/tmp/wheels"
            else:
                for root, _dirs, files in os.walk("/tmp/wheels"):
                    if TARGET in files:
                        wheel_dir = root
                        break

    if wheel_dir is None:
        # Report what IS mounted. "Not attached" and "attached somewhere I did
        # not look" need completely different fixes, and a bare failure cannot
        # tell them apart.
        tree = []
        for root, dirs, _files in os.walk("/kaggle/input"):
            if root.count(os.sep) <= 4:
                tree.append(root)
            else:
                dirs[:] = []
        raise SystemExit(
            f"{TARGET} was not found anywhere under /kaggle/input. Directories "
            f"seen: {sorted(tree)[:60]}. Expected it from the dataset "
            "'WHEELS_DATASET_LITERAL'. If kernel-metadata.json already lists "
            "that under dataset_sources, note that a newly created kernel's "
            "first run can start before its data sources finish attaching -- "
            "pushing again is usually enough."
        )

    found = glob.glob(os.path.join(wheel_dir, "*.whl"))
    print(f"{len(found)} wheels under {wheel_dir}")

    # The version pin is load-bearing. An unpinned `transformers` is
    # "already satisfied" by the image's 5.0.0, so pip does nothing at all and
    # the install reports success while changing none of the problem -- which
    # is exactly what the first attempt did.
    #
    # --no-deps on purpose: a plain resolve would pull the newer numpy in the
    # wheel set and replace the one torch 2.10 was built against. Every
    # dependency transformers 5.5.0 declares is already satisfied by the image,
    # confirmed on the same run (regex 2025.11.3, typer 0.24.2, click 8.3.3,
    # rich 13.9.4, and the rest all present), so transformers is the only
    # package that needs to change.
    packages = ["transformers==5.5.0"]
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
         "--find-links", wheel_dir, *packages],
        capture_output=True, text=True,
    )
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise SystemExit("offline transformers install failed")

    # Confirm the version that is now importable, and that it recognises the
    # architecture -- installing the wheel is not the same as fixing the load.
    check = subprocess.run(
        [sys.executable, "-c",
         "import transformers;"
         "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES;"
         "print(transformers.__version__, 'gemma4' in CONFIG_MAPPING_NAMES)"],
        capture_output=True, text=True,
    )
    print("transformers now:", check.stdout.strip() or check.stderr[-1000:])
    if "True" not in check.stdout:
        raise SystemExit(
            "transformers still does not recognise gemma4 after the install"
        )
    '''
).replace("WHEELS_DATASET_LITERAL", WHEELS_DATASET)


READINESS_GATE_SOURCE = dedent(
    '''\
    """Prove the model is really there and really answers, before any game runs.

    Generated by scripts/build_notebook.py; runs as its own process so the
    62.58 GB it loads is released before `main.py` loads its own copy.

    AGENTS.md: "Model initialization failures and out-of-memory conditions
    must fail before gameplay rather than degrading an entire evaluation
    silently." Exits non-zero on any problem; the notebook turns that into a
    hard stop.
    """
    import os
    import subprocess
    import sys

    sys.path.insert(0, os.getcwd())

    model_dir = os.environ["ZERX_MODEL_PATH"]
    if not os.path.isdir(model_dir):
        sys.exit(f"ZERX_MODEL_PATH={model_dir!r} is not a directory")

    # Record the hardware we actually got. Every environment probe so far ran
    # with is_competition_rerun=false, so it is NOT proven that a scored rerun
    # allocates the same shape a --accelerator push does. If it differs, this
    # line is what makes the result diagnosable rather than merely
    # disappointing.
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        # No nvidia-smi at all means no GPU was attached. Say so plainly
        # instead of dying here with a FileNotFoundError traceback that looks
        # like a bug in the gate rather than a misconfigured kernel.
        gpu = "NOT FOUND -- nvidia-smi is absent, so this kernel has no GPU"
    print("GPU:", gpu, flush=True)

    from zerx.config import Config
    from zerx.model_backend import select_backend

    config = Config.from_env()
    backend = select_backend(config)
    name = type(backend).__name__
    print("resolved backend:", name)
    print("config hash:", config.config_hash())
    print("config:", config.to_json(), flush=True)

    if name == "FakeModelBackend":
        sys.exit(
            "select_backend() resolved to FakeModelBackend, whose every "
            "generate() raises -- the agent would play heuristics-only and "
            "report a meaningless score."
        )
    if name == "GemmaModelBackend":
        # An HTTP client for a vLLM server. There is none here and there
        # cannot be: vllm is absent from the Kaggle image, internet is
        # disabled, and the competition's offline wheels do not ship it
        # (docs/superpowers/experiments/kaggle-env-probe.md). Every call would
        # raise ConnectionRefused and the agent would fall through to
        # heuristics silently -- which the FakeModelBackend check above does
        # NOT catch, because the class is different.
        sys.exit(
            "select_backend() resolved to GemmaModelBackend, an HTTP client "
            "for a vLLM server that does not exist on Kaggle. Use "
            "backend='gemma_kaggle' (in-process transformers)."
        )

    warmup_seconds = backend.warmup()
    latency = backend.last_latency_seconds
    print(f"weights loaded + first generation: {warmup_seconds:.1f}s")
    print(f"steady-state per-call latency: {latency:.2f}s")

    # The per-game action cap has never been calibrated against a real Gemma
    # call. ~25 games x max_actions is what has to fit in Kaggle's ~9 hours,
    # and model time is only part of that.
    projected_hours = 25 * config.max_actions * latency / 3600
    print(f"projected: 25 games x {config.max_actions} actions = "
          f"{projected_hours:.1f}h of model time (kernel limit ~9h)")
    if projected_hours > 7.0:
        print(f"WARNING: {projected_hours:.1f}h of model time leaves little room "
              f"under the ~9h kernel limit. Lower ZERX_MAX_ACTIONS if the run is "
              f"cut off mid-game.", flush=True)
    print("readiness gate passed", flush=True)
    '''
)


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

    transformers_install_cell = code_cell(TRANSFORMERS_INSTALL_SOURCE)

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

    write_gate_cell = code_cell(
        "%%writefile /tmp/zerx_readiness_gate.py\n" + READINESS_GATE_SOURCE
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Copy the zerx package alongside it so `import zerx` resolves —
            # main.py runs with cwd /kaggle/working/ARC-AGI-3-Agents, which is
            # on sys.path via cwd.
            !mkdir -p /kaggle/working/ARC-AGI-3-Agents && \\
                cp -r /tmp/zerx /kaggle/working/ARC-AGI-3-Agents/zerx

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
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

            # ---- model readiness gate ----
            # Resolve the backend exactly as MyAgent will, and refuse to play
            # if it comes back as the no-op FakeModelBackend. AGENTS.md:
            # "Model initialization failures and out-of-memory conditions must
            # fail before gameplay rather than degrading an entire evaluation
            # silently." Before this gate existed, a submission with no model
            # attached was indistinguishable from a working one until the
            # leaderboard came back.
            #
            # Run in-process (not via `!`) on purpose: a shell command's
            # non-zero exit is swallowed by IPython, so only a real Python
            # exception actually stops the notebook here.
            import sys

            KAGGLE_MODEL_DIR = MODEL_DIR_LITERAL
            if KAGGLE_MODEL_DIR is None:
                raise SystemExit(
                    "KAGGLE_MODEL_DIR is not set in scripts/build_notebook.py, so "
                    "no model weights have been located under /kaggle/input. "
                    "Run notebooks/probe/probe.ipynb "
                    "(scripts/build_probe_notebook.py) to resolve the real mount "
                    "path, set the constant, rebuild, and push again. Refusing to "
                    "play heuristics-only and report it as a scored run."
                )
            if not os.path.isdir(KAGGLE_MODEL_DIR):
                raise SystemExit(
                    f"KAGGLE_MODEL_DIR={KAGGLE_MODEL_DIR!r} does not exist. The "
                    "model source is either not attached to this kernel or mounts "
                    "elsewhere; check kernel-metadata.json's model_sources against "
                    "the probe notebook's /kaggle/input listing."
                )

            # Run the readiness gate as its own process, NOT in this kernel.
            # It loads all 62.58 GB of weights to prove they load; if that
            # happened here, the notebook kernel would still be holding them
            # when `main.py` starts below and tries to load a second copy, and
            # even a ~96 GB card cannot hold two. Exiting the process is what
            # releases the VRAM.
            import subprocess as _sp

            _gate = _sp.run(
                [sys.executable, '/tmp/zerx_readiness_gate.py'],
                cwd='/kaggle/working/ARC-AGI-3-Agents',
                env={**os.environ,
                     'ZERX_BACKEND': 'ZERX_BACKEND_LITERAL',
                     'ZERX_PLATFORM': 'ZERX_PLATFORM_LITERAL',
                     'ZERX_MODEL_PATH': KAGGLE_MODEL_DIR},
            )
            if _gate.returncode != 0:
                raise SystemExit(
                    f"readiness gate failed (exit {_gate.returncode}) -- see its "
                    "output above. Refusing to start a scored run that would "
                    "silently play without a model."
                )

            # Run it. The gateway records every action and emits submission.parquet.
            # ZERX_* are passed on the command line rather than left to the .env
            # above: main.py runs in a separate process, and an explicit prefix
            # does not depend on when the framework happens to call load_dotenv().
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                ZERX_BACKEND=ZERX_BACKEND_LITERAL \\
                ZERX_PLATFORM=ZERX_PLATFORM_LITERAL \\
                ZERX_GEMMA_BASE_URL=ZERX_GEMMA_BASE_URL_LITERAL \\
                ZERX_MODEL_PATH=MODEL_DIR_PLAIN \\
                python main.py --agent myagent
        """
    )
    # Substituted rather than f-string-interpolated: the cell body above
    # contains literal braces (the AVAILABLE_AGENTS dict it writes out), which
    # an f-string would try to evaluate.
    run_cell_source = (
        run_cell_source
        .replace("MODEL_DIR_LITERAL", repr(KAGGLE_MODEL_DIR))
        .replace("MODEL_DIR_PLAIN", KAGGLE_MODEL_DIR or "")
        .replace("ZERX_BACKEND_LITERAL", ZERX_BACKEND)
        .replace("ZERX_PLATFORM_LITERAL", ZERX_PLATFORM)
        .replace("ZERX_GEMMA_BASE_URL_LITERAL", ZERX_GEMMA_BASE_URL)
    )
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
            transformers_install_cell,
            mkdir_cell,
            *zerx_write_cells,
            write_agent_cell,
            write_gate_cell,
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

        if meta.get("model_sources") != [MODEL_SOURCE]:
            meta["model_sources"] = [MODEL_SOURCE]
            changes.append(f"model_sources=[{MODEL_SOURCE!r}]")

        if meta.get("dataset_sources") != [WHEELS_DATASET]:
            meta["dataset_sources"] = [WHEELS_DATASET]
            changes.append(f"dataset_sources=[{WHEELS_DATASET!r}]")

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

    if KAGGLE_MODEL_DIR is None:
        print("[build_notebook] WARNING: KAGGLE_MODEL_DIR is None — this notebook "
              "will refuse to play on Kaggle. Run scripts/build_probe_notebook.py "
              "first and set the constant from its /kaggle/input listing.")


if __name__ == "__main__":
    main()
