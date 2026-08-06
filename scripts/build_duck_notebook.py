"""Patch Tufa Labs' published Duck notebook into our submission notebook.

Upstream: https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner
by Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal Tesnar
and Stefano Viel (Tufa Labs), published alongside their ARC-AGI-3 Milestone 1
write-up. The solver, the harness and every dataset this notebook attaches are
theirs. See docs/DUCK_FORK.md for what we changed and why.

Why a patch script instead of editing the .ipynb by hand
--------------------------------------------------------
Every change we make has to be reviewable, reproducible and re-appliable if
upstream publishes a new version. A script that locates cells by content and
applies named edits gives all three; a hand-edited 25 KB JSON blob gives none.
It also fails loudly if upstream changes shape, instead of silently producing a
notebook missing our safety nets.

Run:  python scripts/build_duck_notebook.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "reference" / "duck" / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb"
OUT_DIR = ROOT / "notebooks" / "duck"
OUT_NB = OUT_DIR / "duck.ipynb"
OUT_META = OUT_DIR / "kernel-metadata.json"

KERNEL_ID = "sixcode/arc-agi-3-duck-guarded"
KERNEL_TITLE = "ARC-AGI-3 duck guarded"

# Leave this much of the 9-hour kernel limit for the benchmark to wind down,
# write its artifacts, and let the gateway emit submission.parquet. A run that
# is killed at the limit produces no submission at all, which is the worst
# outcome available and the one this exists to prevent.
SUBMISSION_SOFT_END_SECONDS = 8 * 3600

# Wall clock for the *unscored* Save & Run All pass. Upstream lets that pass run
# to the notebook's full budget; we only need it to prove the pipeline works
# end to end, and a short one can be repeated cheaply.
VERIFY_RUNTIME_SECONDS = 2400


def cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def find_cell(cells: list, needle: str) -> int:
    matches = [i for i, c in enumerate(cells) if needle in cell_source(c)]
    if len(matches) != 1:
        raise SystemExit(
            f"[build_duck] expected exactly one cell containing {needle!r}, "
            f"found {len(matches)}. Upstream notebook changed shape — re-read it "
            "before trusting any patch below."
        )
    return matches[0]


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True},
            "outputs": [], "execution_count": None, "source": source}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


# ── our additions ───────────────────────────────────────────────────────────

PREFLIGHT = dedent(
    '''\
    """Preflight gate — fail now, loudly, rather than in eight hours, silently.

    Added by this fork. Upstream already raises if the source bundle is
    missing, but three other ways to lose a whole run go undetected:

    1. **The wrong GPU.** Kaggle's push API reads the accelerator from the CLI
       flag, not from notebook metadata, and an unrecognised value is ignored
       *silently* — measured in this project: pushing with `nvidiaRtx6000`
       yielded a Tesla P100. The solver needs the RTX Pro 6000's VRAM to hold a
       27B FP8 model; on a P100 it would crawl or OOM, and the first symptom
       would be a bad score.
    2. **A dataset that did not mount.** Kaggle attaches inputs asynchronously
       and a freshly created kernel's first run can start before they finish.
    3. **No GPU at all**, which reads as a missing `nvidia-smi` rather than an
       error.

    None of these is recoverable mid-run, so all three stop the notebook here.
    """
    import subprocess
    from pathlib import Path

    def _gpu_description() -> str:
        try:
            return subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
        except FileNotFoundError:
            return ""

    gpu = _gpu_description()
    print(f"preflight: gpu = {gpu or 'NONE'}")
    if not gpu:
        raise SystemExit(
            "No GPU: nvidia-smi is absent. This notebook serves a 27B model and "
            "cannot run on CPU. Push with --accelerator NvidiaRtxPro6000."
        )

    total_mib = 0
    for part in gpu.split(","):
        digits = "".join(ch for ch in part if ch.isdigit())
        if "MiB" in part and digits:
            total_mib = int(digits)
    print(f"preflight: vram = {total_mib} MiB")
    if total_mib and total_mib < 40000:
        raise SystemExit(
            f"GPU has only {total_mib} MiB. The RTX Pro 6000 this notebook "
            "targets reports ~97000 MiB; Kaggle silently substitutes a smaller "
            "card when the --accelerator value is not recognised. Re-push with "
            "--accelerator NvidiaRtxPro6000 rather than spending a submission "
            "on a card that cannot hold the model."
        )

    missing = [ref for ref, path in kaggle_input_paths.items() if not Path(path).exists()]
    if missing:
        raise SystemExit(
            f"Attached inputs did not mount: {missing}. Kaggle attaches datasets "
            "asynchronously and a new kernel's first run can outrun them — "
            "pushing again is usually enough."
        )
    print(f"preflight: {len(kaggle_input_paths)} input(s) mounted")
    print("preflight: OK")
    '''
)

CUSTOMISE = dedent(
    f'''\
    """Fork customisation — bounds only; the solver is left exactly as published.

    Deliberately nothing here touches prompts, sampling, tools or the model.
    We have no way to measure such a change before submitting — the harness
    needs a 96 GB card and a 27B model — and an unmeasurable change to a
    solution that is already leaderboard-verified is a gamble, not an
    improvement. What we do change is the run's *safety envelope*, where the
    failure modes are known and the downside is bounded.

    Upstream reassigns `bm.games`, `bm.n_passes` and `bm.game_weights` in the
    run cell below, so this cell only sets things that cell does not touch.
    """
    # Bound the unscored Save & Run All pass. Upstream derives its soft end from
    # `target.max_runtime_s`, so shrinking that attribute shortens the
    # verification pass without touching upstream code. The scored rerun ignores
    # this: `TRUE_SUBMISSION` takes the other branch entirely.
    if not TRUE_SUBMISSION:
        target.max_runtime_s = {VERIFY_RUNTIME_SECONDS}
        print(f"fork: verification pass bounded to {{target.max_runtime_s}}s")

    # A run manifest. Upstream prints its preamble but not the resolved shape of
    # the run, which is what you need to tell "scored badly" apart from "played
    # four games and got killed".
    print("fork: run manifest")
    for key in ("max_runtime_s", "actual_run_as_submission", "is_competition_rerun"):
        print(f"  target.{{key}} = {{getattr(target, key, '<unset>')}}")
    for key in ("label", "n_passes", "max_runtime_minutes", "max_steps"):
        print(f"  bm.{{key}} = {{getattr(bm, key, '<unset>')}}")
    print(f"  bm.solver = {{type(getattr(bm, 'solver', None)).__name__}}")
    '''
)


def patch_run_cell(source: str) -> str:
    """Give the scored run a soft deadline. Upstream only gives one to the
    unscored pass, so a hidden game set larger or slower than the public one can
    run past Kaggle's 9-hour limit and be killed — which produces no
    submission.parquet at all, scoring nothing regardless of how well the games
    went. A soft end lets the benchmark stop itself and write its results.

    Set late on purpose: if the run would have finished anyway, this changes
    nothing. It only fires in the case that would otherwise have lost
    everything. The `soft_end_time` path is the same one upstream exercises on
    every interactive run, so this changes the value, not the mechanism.
    """
    old = dedent(
        """\
        soft_end = None
        if not TRUE_SUBMISSION:
            budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
            if budget > 0:
                soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))
        """
    )
    if old not in source:
        raise SystemExit(
            "[build_duck] the soft-deadline block is not where this patch "
            "expects it. Re-read the upstream run cell before rebuilding."
        )
    new = dedent(
        f"""\
        soft_end = None
        if not TRUE_SUBMISSION:
            budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
            if budget > 0:
                soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))
        else:
            # FORK CHANGE: upstream leaves the scored run with no deadline at
            # all. Kaggle kills a notebook at 9 hours and a killed notebook
            # emits no submission.parquet, so a hidden game set that is larger
            # or slower than the public one loses the entire run rather than
            # part of it. Stop early enough to wind down and write results.
            soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(
                seconds={SUBMISSION_SOFT_END_SECONDS}
            )
        print(f"taaf.kaggle: soft_end = {{soft_end}}")
        """
    )
    return source.replace(old, new)


def build() -> dict:
    if not UPSTREAM.exists():
        raise SystemExit(
            f"Upstream notebook not found at {UPSTREAM}. Fetch it first:\n"
            "  kaggle kernels pull jeroencottaar/"
            "tufa-labs-duck-harness-june-30-milestone-winner -p reference/duck -m"
        )
    notebook = json.loads(UPSTREAM.read_text(encoding="utf-8"))
    cells = notebook["cells"]

    # 1. Soft deadline for the scored run.
    run_idx = find_cell(cells, "soft_end = None")
    cells[run_idx]["source"] = patch_run_cell(cell_source(cells[run_idx]))

    # 2. Replace upstream's empty customisation hook with ours. Using the hook
    #    upstream provides, rather than editing their logic, keeps our changes
    #    in one reviewable place.
    hook_idx = find_cell(cells, "Make one-off changes to `bm`")
    cells[hook_idx]["source"] = CUSTOMISE

    # 3. Preflight, immediately after the cell that resolves the input mounts
    #    (it needs `kaggle_input_paths`) and before the multi-GB setup commands.
    mounts_idx = find_cell(cells, "TAAF_KAGGLE_INPUT_PATHS")
    cells.insert(mounts_idx + 1, markdown_cell(
        "## 3b. Preflight (fork addition)\n\nRefuse to start the run when the "
        "GPU, the VRAM or an attached input is not what this notebook needs — "
        "each of those otherwise costs a whole submission to discover."
    ))
    cells.insert(mounts_idx + 2, code_cell(PREFLIGHT))

    # 4. Attribution, at the top, where a reader sees it first.
    cells.insert(0, markdown_cell(dedent(
        """\
        # ARC-AGI-3 — Duck harness (guarded fork)

        **This notebook is a fork of Tufa Labs' published Duck harness.** The
        solver, the harness (TAAF) and every attached dataset are their work:
        Harold Bessis, Jeroen Cottaar, Isaiah Pressman, Andries Smit, Michal
        Tesnar and Stefano Viel. Original notebook:
        [tufa-labs-duck-harness-june-30-milestone-winner](https://www.kaggle.com/code/jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner),
        write-up:
        [competition discussion 717133](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133),
        source: [github.com/Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness).

        This fork changes **no solver behaviour**. It adds a preflight gate, a
        soft deadline for the scored run, a bounded verification pass and a run
        manifest — all of them guards against losing a run outright. Every
        change is generated by `scripts/build_duck_notebook.py` and documented
        in `docs/DUCK_FORK.md`.
        """
    )))

    notebook["metadata"].setdefault("kaggle", {})
    return notebook


def main() -> None:
    notebook = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    upstream_meta = json.loads(
        (UPSTREAM.parent / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    meta = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": OUT_NB.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        # Carried over verbatim: these are the wheelhouse, the harness source
        # and the model snapshot the notebook is written against. Substituting
        # any of them is not a small change.
        "dataset_sources": upstream_meta["dataset_sources"],
        "kernel_sources": [],
        "competition_sources": upstream_meta["competition_sources"],
        "model_sources": [],
        # Pinned upstream. The harness is built against this image's CUDA and
        # Python; letting it float would make the run unreproducible.
        "docker_image": upstream_meta["docker_image"],
    }
    OUT_META.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"[build_duck] wrote {OUT_NB.relative_to(ROOT)} "
          f"({len(notebook['cells'])} cells)")
    print(f"[build_duck] wrote {OUT_META.relative_to(ROOT)}")
    print("[build_duck] push with:  kaggle kernels push -p notebooks/duck "
          "--accelerator NvidiaRtxPro6000")


if __name__ == "__main__":
    sys.exit(main())
