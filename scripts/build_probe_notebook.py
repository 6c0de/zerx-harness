"""Generates notebooks/probe/probe.ipynb — a Kaggle *environment probe*.

This is neither the submission notebook (scripts/build_notebook.py) nor the
Colab dev notebook (scripts/build_colab_notebook.py). It plays no game,
loads no model, and consumes no submission slot. Its only job is to report
what the real Kaggle ARC-AGI-3 runtime actually contains, so the
model-serving path can be designed from measurements instead of
assumptions.

Why it exists: docs/HANDOFF.md's ARC-HANDOFF-001 recommends "add a cell
that installs vLLM offline and launches the server", but nothing has ever
verified that vLLM can be installed or run inside a Kaggle notebook with
internet disabled. vLLM is not part of the Kaggle Python image, its own
tracker carries an open "vLLM will NOT run in a Kaggle Notebook" install
issue for versions above 0.10, and the Colab bring-up of the same model
already cost four separately-diagnosed install failures. AGENTS.md is
explicit that we do not build on planned-but-unverified paths.

The probe runs in an environment configured identically to the real
submission — same accelerator, internet disabled, same competition and
model attached — so its answers transfer directly.

Push it with (from the repo root, after .kaggle/access_token exists):

    python scripts/build_probe_notebook.py
    KAGGLE_API_TOKEN=$(cat .kaggle/access_token) kaggle kernels push -p notebooks/probe

Then read /kaggle/working/probe.json from the finished kernel's output.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
PROBE_DIR = ROOT / "notebooks" / "probe"
NOTEBOOK_PATH = PROBE_DIR / "probe.ipynb"
METADATA_PATH = PROBE_DIR / "kernel-metadata.json"

# ─────────────────────────────────────────────────────────────────────────────
# The probe must mirror the real submission environment, not a convenient one.
# These three constants are the mirror; keep them in step with
# scripts/build_notebook.py.
# ─────────────────────────────────────────────────────────────────────────────
KAGGLE_USERNAME = "enzeceb"
KERNEL_SLUG = "zerx-kaggle-env-probe"

# Same accelerator the scored run will use (AGENTS.md: Kaggle RTX Pro 6000,
# 48GB, g4-standard-48, ARC-AGI-3-exclusive). Probing a T4 would answer
# questions about a card we will never run on.
ACCELERATOR_NAME = "nvidiaRtx6000"

# Kaggle Models handle for the target weights. docs/HANDOFF.md records the
# Kaggle Models UI displaying this as "google/gemma-4/Transformers/gemma-4-31b-it";
# Kaggle's API handles are lowercase in the framework segment. If a push
# fails with "model not found", correct this one line — that failure costs
# nothing (it is rejected before any GPU is allocated) and the probe's
# /kaggle/input listing is what confirms the real mount path afterwards.
MODEL_SOURCE = "google/gemma-4/transformers/gemma-4-31b-it"

COMPETITION_SLUG = "arc-prize-2026-arc-agi-3"


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


# Every probe section is wrapped in its own try/except and appends to a
# single PROBE dict. Kaggle aborts a notebook at the first uncaught
# exception, and a probe that dies on question 3 of 8 is nearly worthless —
# the whole point is to come back with every answer in one run.
_INIT_CELL = dedent(
    """\
    import json, os, platform, shutil, subprocess, sys

    PROBE = {"schema": 1}

    def section(name):
        \"\"\"Record one probe section, capturing failures instead of raising.\"\"\"
        def decorator(fn):
            try:
                PROBE[name] = fn()
                print(f"[ok] {name}")
            except Exception as exc:
                PROBE[name] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"[FAILED] {name}: {type(exc).__name__}: {exc}")
            return fn
        return decorator
    """
)

_HOST_CELL = dedent(
    """\
    @section("host")
    def _host():
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "is_competition_rerun": bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN")),
            "disk_free_gb": {
                path: round(shutil.disk_usage(path).free / 1e9, 1)
                for path in ("/kaggle/working", "/tmp", "/kaggle/input")
                if os.path.isdir(path)
            },
            "ram_total_gb": round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1
            ),
        }


    @section("gpu")
    def _gpu():
        smi = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        return {
            "nvidia_smi": smi.stdout.strip() or smi.stderr.strip(),
            "returncode": smi.returncode,
        }
    """
)

_TORCH_CELL = dedent(
    """\
    @section("torch")
    def _torch():
        import torch
        out = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            props = torch.cuda.get_device_properties(0)
            out.update({
                "device_name": props.name,
                "vram_total_gb": round(props.total_memory / 1e9, 1),
                "compute_capability": f"{major}.{minor}",
                # >= 8.9 is the threshold for true FP8 W8A8; below it, FP8
                # runs weight-only W8A16 (see docs/HANDOFF.md's 2026-08-06
                # Colab/Kaggle quantization decision).
                "fp8_w8a8_capable": (major, minor) >= (8, 9),
            })
        # float8_e4m3fn's presence is what makes a transformers-native FP8
        # path viable at all, independent of vLLM.
        out["has_float8_e4m3fn"] = hasattr(torch, "float8_e4m3fn")
        return out


    @section("packages")
    def _packages():
        import importlib.metadata as md
        import importlib.util

        names = [
            "vllm", "transformers", "accelerate", "bitsandbytes",
            "safetensors", "tokenizers", "numpy", "pandas", "pyarrow",
            "huggingface_hub", "flash_attn", "xformers",
        ]
        found = {}
        for name in names:
            spec = None
            try:
                spec = importlib.util.find_spec(name)
            except Exception:
                pass  # a broken/partial install raises here; record as absent
            try:
                version = md.version(name)
            except Exception:
                version = None
            found[name] = {"importable": spec is not None, "version": version}
        return found


    @section("transformers_quantization")
    def _transformers_quantization():
        \"\"\"Which quantization configs this transformers actually exposes.

        Decides whether an in-process transformers backend can hit the same
        FP8 precision the Colab run was standardized on, or whether it would
        have to fall back to 4-bit bitsandbytes (which would break that
        parity and needs recording if chosen).
        \"\"\"
        import transformers
        candidates = [
            "FineGrainedFP8Config", "BitsAndBytesConfig", "QuantoConfig",
            "TorchAoConfig", "CompressedTensorsConfig",
        ]
        return {
            "version": transformers.__version__,
            "configs": {name: hasattr(transformers, name) for name in candidates},
        }
    """
)

_INPUT_CELL = dedent(
    """\
    @section("kaggle_input")
    def _kaggle_input():
        \"\"\"Resolve where the attached model and competition actually mount.

        The Kaggle Models UI label ("google/gemma-4/Transformers/gemma-4-31b-it")
        is an organizational path, not a filesystem path -- and separately is
        not a valid Hugging Face repo id either (docs/HANDOFF.md, Day 2 item
        2). Only a real listing settles it.
        \"\"\"
        root = "/kaggle/input"
        tree = {}
        if not os.path.isdir(root):
            return {"error": f"{root} does not exist"}
        for dirpath, dirnames, filenames in os.walk(root):
            depth = dirpath[len(root):].count(os.sep)
            if depth >= 6:
                dirnames[:] = []
                continue
            # Weight files are what matter; list them with sizes, and count
            # the rest rather than dumping thousands of paths.
            weights = sorted(
                f for f in filenames
                if f.endswith((".safetensors", ".bin", ".gguf", ".json"))
            )[:40]
            if weights:
                tree[dirpath] = {
                    "files": [
                        {"name": f,
                         "size_gb": round(
                             os.path.getsize(os.path.join(dirpath, f)) / 1e9, 3)}
                        for f in weights
                    ],
                    "total_files_in_dir": len(filenames),
                }
        return {"weight_bearing_dirs": tree, "top_level": sorted(os.listdir(root))}


    @section("competition_dataset")
    def _competition_dataset():
        base = "/kaggle/input/competitions/COMPETITION_SLUG"
        if not os.path.isdir(base):
            return {"error": f"{base} does not exist"}
        out = {"top_level": sorted(os.listdir(base))}
        wheels = os.path.join(base, "arc_agi_3_wheels")
        if os.path.isdir(wheels):
            # Whether vLLM could ever be installed offline from what the
            # competition already ships is a yes/no this listing answers.
            out["arc_agi_3_wheels"] = sorted(os.listdir(wheels))
        return out


    @section("network")
    def _network():
        \"\"\"Confirm internet really is disabled, rather than trusting metadata.\"\"\"
        import socket
        socket.setdefaulttimeout(5)
        try:
            socket.create_connection(("pypi.org", 443), timeout=5).close()
            return {"internet_reachable": True}
        except Exception as exc:
            return {"internet_reachable": False, "error": f"{type(exc).__name__}: {exc}"}
    """
).replace("COMPETITION_SLUG", COMPETITION_SLUG)

_SAVE_CELL = dedent(
    """\
    out_path = "/kaggle/working/probe.json"
    with open(out_path, "w") as handle:
        json.dump(PROBE, handle, indent=2, default=str)

    print("=" * 72)
    print(json.dumps(PROBE, indent=2, default=str))
    print("=" * 72)
    print("Saved:", out_path)

    failed = [name for name, value in PROBE.items()
              if isinstance(value, dict) and "error" in value]
    if failed:
        # Not raised: a partial probe is still worth downloading, and Kaggle
        # would discard the saved output if this cell raised.
        print("Sections that failed:", failed)
    """
)


def build() -> dict:
    intro = markdown_cell(
        "# zerx — Kaggle environment probe\n\n"
        "Not a submission. Plays no game, loads no model, consumes no "
        "submission slot. Reports what the real ARC-AGI-3 Kaggle runtime "
        "contains so the model-serving path can be designed from measured "
        "facts (see `docs/superpowers/specs/2026-08-06-kaggle-p0-model-attach-design.md`).\n\n"
        "Generated by `scripts/build_probe_notebook.py` — do not edit cells "
        "directly.\n\n"
        "Every section is independently error-trapped, so one failure does "
        "not cost the other answers. Results land in "
        "`/kaggle/working/probe.json`."
    )

    return {
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
                "accelerator": ACCELERATOR_NAME,
                "isInternetEnabled": False,
                "isGpuEnabled": True,
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            intro,
            code_cell(_INIT_CELL),
            code_cell(_HOST_CELL),
            code_cell(_TORCH_CELL),
            code_cell(_INPUT_CELL),
            code_cell(_SAVE_CELL),
        ],
    }


def build_metadata() -> dict:
    return {
        "id": f"{KAGGLE_USERNAME}/{KERNEL_SLUG}",
        "title": "zerx — Kaggle env probe",
        "code_file": "probe.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": [],
        "dataset_sources": [],
        "kernel_sources": [],
        "competition_sources": [COMPETITION_SLUG],
        "model_sources": [MODEL_SOURCE],
    }


def main() -> None:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    METADATA_PATH.write_text(json.dumps(build_metadata(), indent=2) + "\n", encoding="utf-8")
    print(f"[build_probe_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}")
    print(f"[build_probe_notebook] Wrote {METADATA_PATH.relative_to(ROOT)}")
    print(f"[build_probe_notebook] accelerator={ACCELERATOR_NAME}  "
          f"model_source={MODEL_SOURCE}")


if __name__ == "__main__":
    main()
