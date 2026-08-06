# Experiment — `kaggle-env-probe`

- Experiment ID: `kaggle-env-probe`
- Date: 2026-08-06
- Owner: local session (Claude Code), human owner approved every push
- Branch: `feat/kaggle-p0-model-attach`
- Base commit: `c7e7d3d` (probe v1/v2), plus the depth fix below for v3/v4
- Kernel: `enzeceb/zerx-kaggle-env-probe` (private), 4 versions
- Cost: 4 kernel runs, ~0.05h of a 30h weekly GPU quota. No submission slot
  consumed — this is a notebook, not a submission.

## Hypothesis

`docs/HANDOFF.md`'s ARC-HANDOFF-001 recommends serving Gemma-4-31B on
Kaggle by installing vLLM offline and launching its server on an RTX Pro
6000 (48GB). Every element of that — vLLM's availability, the accelerator,
the model's mount path — was planned but never measured. `AGENTS.md`
forbids building on unverified paths, so this probe measures the real
runtime before any serving code is written.

## Method

`scripts/build_probe_notebook.py` generates a notebook that mirrors the
submission environment (internet disabled, competition attached, the same
versioned Gemma model source attached) but plays no game and loads no
model. Every section is independently error-trapped so one failure cannot
cost the other answers. Results are written to `/kaggle/working/probe.json`.

Four versions were run:

| v | What changed | Accelerator requested | GPU actually assigned |
|---|---|---|---|
| 1 | first run | notebook metadata `nvidiaRtx6000`, no CLI flag | **Tesla P100-PCIE-16GB** |
| 2 | none (re-push) | `--accelerator nvidiaRtx6000` | **Tesla P100-PCIE-16GB** |
| 3 | `/kaggle/input` walk depth fix, new `attached_model` section | `--accelerator` omitted | **Tesla P100-PCIE-16GB** |
| 4 | none (re-push) | `--accelerator NvidiaTeslaT4` | **Tesla T4** |

## Results

### 1. The accelerator selection in the starter has never worked

This is the headline finding, and it is not specific to this repo — the
**upstream official starter has the same gap**.

`scripts/build_notebook.py` (ours and upstream's) writes the chosen
accelerator into the *notebook's* `metadata.kaggle.accelerator`, and syncs
only `enable_gpu` into `kernel-metadata.json`. But `kaggle kernels push`
never reads either accelerator field: the CLI's own
`kaggle/api/kaggle_api_extended.py` reads exactly one GPU-related key from
the metadata file, `enable_gpu` (a bool). The accelerator is carried only
by the `--accelerator` CLI flag, which the starter's `make submit` does not
pass.

So `ACCELERATOR = "rtx6000"` requested nothing. Kaggle assigned its default
GPU. The same was true of the previous `ACCELERATOR = "t4"` default.

**Controlled experiment separating "flag broken" from "value unavailable":**

- v2, `--accelerator nvidiaRtx6000` → P100 (the default)
- v4, `--accelerator NvidiaTeslaT4` → **Tesla T4**

The flag works. `nvidiaRtx6000` is silently ignored and falls back to the
default rather than erroring. Consistent with the Kaggle SDK's own
documented `machine_shape` options
(`kagglesdk/kernels/types/kernels_api_service.py`), which list only
`NvidiaTeslaT4`, `NvidiaTeslaP100`, and `Tpu1VmV38` — **no RTX 6000**.

**Unresolved:** whether the scored *competition rerun* allocates different
hardware than a commit run, and whether the RTX 6000 can be selected from
the Kaggle web UI (which may expose options the CLI enum does not). Both
`AGENTS.md` and the starter's README assert an ARC-AGI-3-exclusive RTX Pro
6000; neither has been confirmed against a live run. Every probe run here
reported `is_competition_rerun: false`.

### 2. On the assigned P100, PyTorch cannot run anything at all

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.
```

`torch.cuda.is_available()` still returns `True`, so a naive readiness check
would pass and then fail at the first kernel launch. The T4 (sm_75) is
supported.

### 3. vLLM is not available, and cannot be made available offline

| Package | Importable | Version |
|---|---|---|
| `vllm` | **no** | — |
| `bitsandbytes` | **no** | — |
| `flash_attn` | no | — |
| `xformers` | no | — |
| `transformers` | yes | 5.0.0 |
| `accelerate` | yes | 1.13.0 |
| `safetensors` | yes | 0.7.0 |
| `tokenizers` | yes | 0.22.2 |
| `huggingface_hub` | yes | 1.11.0 |
| `numpy` / `pandas` / `pyarrow` | yes | 2.0.2 / 2.3.3 / 24.0.0 |

Internet is genuinely disabled (`gaierror: Temporary failure in name
resolution`), and the competition's own offline wheels directory ships
`arc_agi`, `arcengine`, flask, matplotlib and similar — **no vLLM, no
bitsandbytes**. Installing either would require uploading a wheels dataset.

`transformers` 5.0.0 exposes `FineGrainedFP8Config`, `BitsAndBytesConfig`,
`QuantoConfig`, `TorchAoConfig`, and `CompressedTensorsConfig`. Note
`BitsAndBytesConfig` is useless without the absent `bitsandbytes` package.

### 4. The attached model mounts correctly, at a path worth recording

```
/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1
```

2 safetensors shards, **62.58 GB total** (bf16, consistent with the ~61.4GB
estimate in `docs/HANDOFF.md`), alongside `config.json`,
`generation_config.json`, `model.safetensors.index.json`,
`processor_config.json`, `tokenizer.json`, `tokenizer_config.json`. The
`processor_config.json` reflects the model's multimodal image input.

Getting there required a versioned handle. `model_sources` entries are
rejected without a trailing version number, and the framework segment is
lowercase in the API (`transformers`, not the UI's `Transformers`):

```
google/gemma-4/transformers/gemma-4-31b-it/1
```

### 5. Host

Python 3.12.13, Linux 6.12.90, 4 CPUs, 33.7 GB RAM. Disk free:
`/kaggle/working` 20.9 GB, `/tmp` 1102.5 GB, `/kaggle/input` 20.9 GB.

## Conclusion: `investigate`

Not `keep` — no serving path is established. Not `revert` — the probe
itself is sound and its findings are the reason to keep it.

The blocking question is now hardware, not software. 62.58 GB of bf16
weights need roughly 31 GB at 8-bit or 16 GB at 4-bit, before KV cache and
activations. Neither obtainable card (P100 16GB, T4 16GB) can hold this
model, and the P100 cannot run PyTorch at all. Phase B's serving design
cannot be finalized until it is known which GPU the scored run gets.

### Cheapest next tests, in order

1. **Set the accelerator from the Kaggle web UI** on
   `enzeceb/zerx-kaggle-env-probe` (Settings → Accelerator) and re-run.
   The UI may expose options the CLI enum does not. Costs one probe run,
   no submission slot. Requires the human owner — it is a UI action.
2. **If the UI has no RTX 6000 either**, the remaining way to learn what
   the scored rerun allocates is a real submission whose rerun path reports
   its GPU. That costs a submission slot, and argues for sending a
   deliberately cheap heuristics-only submission first rather than a
   one-shot model submission into unknown hardware.
3. Only once the card is known does the serving choice (`transformers`
   in-process at some quantization, versus a vLLM wheels dataset, versus a
   smaller Gemma variant) become answerable rather than speculative.

### Note on the mission target

`AGENTS.md` scopes this project to "Gemma-4-31B only". If the scored run
turns out to provide a 16 GB card, that constraint is unsatisfiable and the
scope needs an explicit human-owner decision, not a silent substitution.
Gemma 4 also ships E2B, E4B, 12B, and 26B-A4B variants; the 26B-A4B MoE
activates only ~3.8B parameters per token and the 12B is dense — either
could be viable where the 31B is not. Recording this as a decision the
owner must make, not one this session took.

## Probe defect found and fixed

v1/v2's `/kaggle/input` walk pruned at `depth >= 6`. An attached Kaggle
model sits at `models/<owner>/<model>/<framework>/<instance>/<version>/` —
depth 6 before a single weight file — so the probe reported no weights at
all, which read exactly like "the model did not mount". Fixed in v3 by
raising the limit and adding a dedicated `attached_model` section that
locates weight shards wherever they land and sums their size. The finding
in section 4 above comes from the fixed version.
