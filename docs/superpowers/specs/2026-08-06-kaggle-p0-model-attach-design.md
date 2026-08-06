# Design — Kaggle P0: attach a real model to the submission

- Date: 2026-08-06
- Branch: `feat/kaggle-p0-model-attach` (off `master` @ `72d0426`)
- Owner: local session (Claude Code), with the human owner approving every
  Kaggle-side action
- Closes (intended): `docs/HANDOFF.md` ARC-HANDOFF-001 (P0), plus the four
  smaller P0 blockers recorded in the same file's "Kaggle state" and
  "Exact next action" sections
- Does **not** touch: `master`, `STRATEGY.md`, or any P1/P3 audit finding

## Problem

A submission built from `master` today runs the agent with **no language
model at all**, and does so silently. `notebooks/kernel-metadata.json` has
`model_sources: []`; `scripts/build_notebook.py` never sets `ZERX_BACKEND`
and never serves a model; so `Config.backend` falls to its default
`"fake"`, `select_backend` returns `FakeModelBackend()` with an empty
response list, and every `generate()` raises. The agent then plays
heuristics-only through its own fallback chain — no crash, no error, just
a meaningless score. The entire Gemma thesis is absent from the scored
artifact.

Four smaller blockers sit alongside it:

1. `scripts/build_notebook.py`'s `ACCELERATOR = "t4"` (2x16 GB)
   contradicts the RTX Pro 6000 (48 GB) target in `AGENTS.md`.
2. `notebooks/kernel-metadata.json`'s `id` is still the literal
   `REPLACE_WITH_YOUR_USERNAME/...` placeholder, which `make submit`
   refuses to push.
3. This checkout has no `.venv` and no `.kaggle/access_token`, so nothing
   can be built, tested, or pushed from this machine.
4. No submission has ever been made, so the whole build → push → run →
   `submission.parquet` → scoring chain is unverified.

## Constraint that shapes the design

The obvious fix — "add a vLLM cell to the Kaggle notebook" — rests on an
assumption nobody has verified: that vLLM can be installed and run inside
a Kaggle notebook with internet disabled. Evidence against it is easy to
find. vLLM is not part of the Kaggle Python image, vLLM's own tracker
carries an open "vLLM will NOT run in a Kaggle Notebook" installation
issue for versions above 0.10, and the community workaround is a
multi-gigabyte prebuilt-wheels dataset. The Colab bring-up of the same
model already cost four distinct, separately-diagnosed install failures
(`docs/HANDOFF.md`, Day 2 items 1–4).

`AGENTS.md` is explicit that we do not design on assumptions: *"Do not
assume that a design-document path or command exists merely because it was
planned."* So this design deliberately splits the work in two, and spends
one cheap Kaggle kernel run on measurement before committing to a serving
architecture.

## Approach

Two phases on one branch.

### Phase A — measure, and land every probe-independent fix

**A1. A probe notebook.** A new `scripts/build_probe_notebook.py`
generates `notebooks/probe/probe.ipynb` with its own
`notebooks/probe/kernel-metadata.json`, in its own directory so it can
never collide with the submission kernel (Kaggle treats one directory as
one kernel). It is not a submission and consumes no submission slot.

It runs in an environment configured *identically* to the real submission
— `rtx6000` accelerator, internet disabled, the competition attached, and
the Gemma model attached as a model source — but plays no game. It prints
and persists:

- GPU name, total VRAM, and CUDA compute capability (the >= 8.9 threshold
  decides whether FP8 runs as true W8A8 or falls back to weight-only
  W8A16, per the quantization decision already recorded in
  `docs/HANDOFF.md`);
- `torch.__version__`, `torch.version.cuda`, and the `transformers`
  version;
- whether `vllm`, `bitsandbytes`, and `accelerate` are importable;
- the `/kaggle/input` tree, which resolves the Gemma model's real mount
  path — the Kaggle Models UI label
  (`google/gemma-4/Transformers/gemma-4-31b-it`) is an organizational
  path, not a filesystem path, and is separately known not to be a valid
  Hugging Face repo id either;
- the contents of the competition's bundled wheels directory;
- a JSON summary written to `/kaggle/working/probe.json` so the result is
  downloadable rather than only readable in the log.

**A2. Probe-independent fixes to `scripts/build_notebook.py`.**

- `ACCELERATOR` becomes `"rtx6000"`.
- The run cell exports `ZERX_BACKEND`, `ZERX_PLATFORM`, and
  `ZERX_GEMMA_BASE_URL` from constants declared at the top of the build
  script, following the existing "change this one line" convention that
  `ACCELERATOR` already uses. Phase B then only flips constants.
- A fail-fast readiness gate runs before `main.py`: resolve
  `Config.from_env()`, call `select_backend`, and abort loudly if the
  result is a `FakeModelBackend`. This is the direct countermeasure to
  the failure mode that motivated this whole design — a silent
  heuristics-only run must become an early, loud failure. `AGENTS.md`
  requires exactly this: *"Model initialization failures and
  out-of-memory conditions must fail before gameplay rather than
  degrading an entire evaluation silently."*

**A3. `notebooks/kernel-metadata.json`.** Real kernel id
(`enzeceb/...`) and a non-empty `model_sources`.

**A4. Local environment on this Windows machine.** `make` is not
installed here and the only interpreter is Python 3.14, so the underlying
commands are run directly, exactly as `docs/superpowers/experiments/baseline-000.md`
already records for this platform. The vendored `Makefile` is not
modified — it stays byte-for-byte what a macOS/Linux teammate uses, per
`AGENTS.md`'s team contract. The Kaggle API token is placed by the human
owner into `.kaggle/access_token` (gitignored); no agent writes it.

**A5. Tests.** A new `tests/test_build_probe_notebook.py` covering the
probe generator, and additions to `tests/test_kaggle_bundle_importable.py`
asserting that the built submission notebook sets a real `ZERX_BACKEND`
and that `kernel-metadata.json` carries a non-empty `model_sources`.

### Phase B — serve the model, using what Phase A measured

Written only after the probe result exists. It adds the model-serving
path (vLLM server, or an in-process `transformers` backend, or a
prequantized checkpoint — whichever the probe shows is actually
available) plus whatever backend code that choice requires. Deferring
this is the entire point: the choice is made from measured facts rather
than from a plan written against an unverified assumption.

## What this design does not close

Stated plainly, because the phase split makes it easy to misread progress:

- **ARC-HANDOFF-001 is not closed by Phase A.** After Phase A,
  `select_backend` returns the right class and the gate fires correctly,
  but nothing serves a model yet. Phase B closes it.
- **"No submission has ever been made" is not closed by Phase A.** A
  probe kernel is not a submission. That blocker closes only when Phase B
  ships, the human owner approves the push, the kernel reaches a terminal
  status, and Kaggle returns a score — which is also `AGENTS.md`'s own
  definition of a complete Kaggle experiment.

## Known interaction with an unfixed P1 bug

ARC-HANDOFF-002 (concurrent game threads sharing mutable `GameAction`
singletons) is still open and is deliberately out of scope here. It
matters for interpreting Phase B's result: the Kaggle run cell invokes
`main.py --agent myagent` with no `--game`, so `Swarm` plays every game
concurrently in threads, and one game can submit another game's ACTION6
coordinates. A disappointing score would then be ambiguous between "the
model reasons poorly" and "the actions were corrupted in flight".

ARC-HANDOFF-002's own option (a) — iterate games sequentially in the run
cell — is a one-line change to the run command rather than a code change,
at a wall-clock cost. Whether to take it is a Phase B decision for the
human owner, recorded here so the trade-off is visible before the run
rather than after it.

## Testing

Model-free and local, per `AGENTS.md`'s testing gates:

- the probe generator produces valid notebook JSON, attaches the model
  and competition sources, disables internet, and requests the intended
  accelerator;
- the submission notebook's run cell sets a non-`fake` `ZERX_BACKEND` and
  `ZERX_PLATFORM=kaggle`;
- the readiness gate is present and precedes the `main.py` invocation;
- `kernel-metadata.json` has a non-placeholder id and a non-empty
  `model_sources`;
- the existing bundle-importability and secret-hygiene tests still pass,
  and `zerx/backends/` is still never bundled.

## Approval gate

Nothing is pushed to Kaggle without explicit human-owner approval. Before
any push, this session presents the commit, experiment id, kernel slug,
accelerator, attached sources, resolved model path, offline status, and
expected output — the exact list `AGENTS.md`'s Kaggle gate requires.
