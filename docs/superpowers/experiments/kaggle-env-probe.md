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
| 6 | none (re-push) | `--accelerator NvidiaRtxPro6000` | **RTX PRO 6000 Blackwell Server Edition** |

(v5 was a human-owner save from the Kaggle web UI, used to recover the
correct accelerator string; its run was cancelled and produced no output.)

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

The flag works. `nvidiaRtx6000` — the starter's own name for the card — is
silently ignored and falls back to the default rather than erroring.

**The correct string is `NvidiaRtxPro6000`.** It appears in neither the
starter nor the Kaggle SDK's documented `machine_shape` list
(`kagglesdk/kernels/types/kernels_api_service.py` knows only
`NvidiaTeslaT4`, `NvidiaTeslaP100`, `Tpu1VmV38`). It was recovered by
having the human owner select the accelerator in the Kaggle web UI, then
reading the server's own metadata back:

```bash
kaggle kernels pull enzeceb/zerx-kaggle-env-probe -m
# -> "machine_shape": "NvidiaRtxPro6000"
```

Pushing with that value produced the RTX card on the first try (v6). This
is the general technique worth remembering: when the CLI's documented enum
is incomplete, set the option in the UI and read the server's metadata back
rather than guessing strings.

### 2. On the assigned P100, PyTorch cannot run anything at all

```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.
```

`torch.cuda.is_available()` still returns `True`, so a naive readiness check
would pass and then fail at the first kernel launch. The T4 (sm_75) is
supported, and the RTX PRO 6000 produced no such warning at all.

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

### 5. The RTX PRO 6000 has ~96 GB, not the 48 GB this project assumed

The card, once actually obtained (v6):

| | |
|---|---|
| Name | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| VRAM | 97887 MiB per `nvidia-smi`; `torch` reports 102.0 GB total |
| Compute capability | **12.0** |
| FP8 W8A8 capable | **yes** |
| CPUs | 48 |
| RAM | 189.9 GB |
| PyTorch sm warnings | **none** |

`AGENTS.md` and `docs/HANDOFF.md` both state 48 GB, and the 2026-08-06
Colab/Kaggle quantization decision rests entirely on that figure: *"Kaggle's
RTX Pro 6000 has 48GB VRAM — bf16 (~61.4GB weights) does not fit; it needs
quantization regardless of what Colab needs."* **The premise is wrong.**
62.58 GB of bf16 weights fit in ~96 GB with roughly 33 GB left for KV cache
and activations, so no quantization is required on Kaggle at all.

### 6. Host

Python 3.12.13, Linux 6.12.90. Disk free: `/kaggle/working` 20.9 GB,
`/tmp` 1102.5 GB, `/kaggle/input` 20.9 GB. Note `/kaggle/working` is far
too small to copy the weights into — they must be loaded in place from
`/kaggle/input`, which is a read-only mount.

The RTX runs on a different image than the default GPU shape:
`gcr.io/kaggle-private-byod/python@sha256:37c64f7d...`. Package versions
reported above were identical across both, but this is worth re-checking if
a future run behaves differently.

## Conclusion: `keep` the probe; Phase B is now unblocked

The hardware question that blocked the serving design is answered, and the
answer is better than the project assumed.

**Implications for Phase B, in order of how much they simplify it:**

1. **No quantization is needed.** bf16 fits natively. This removes the
   entire FP8/bitsandbytes problem — which matters, because `bitsandbytes`
   is not installed and cannot be installed offline, making
   `BitsAndBytesConfig` unusable despite `transformers` exposing it.
2. **vLLM is not needed either, and cannot be had.** With bf16 fitting
   directly and `transformers` 5.0.0 + `accelerate` 1.13.0 present, an
   in-process backend loading straight from the read-only `/kaggle/input`
   mount is both simpler and the only option that does not require
   uploading a multi-gigabyte wheels dataset. vLLM's advantage is
   throughput under concurrent load; whether that matters here depends on
   ARC-HANDOFF-002 (the agent swarm plays games concurrently in threads),
   which is a separate open decision.
3. **The Colab fp8 parity decision should be revisited**, since its stated
   reason no longer holds. Not changed here — it is recorded in
   `docs/HANDOFF.md` as a human-owner decision and should be re-decided by
   the owner, not silently reversed by this session.

### What was built from these findings (Phase B, `feat/kaggle-phase-b-transformers-backend`)

- `TransformersModelBackend` in `zerx/model_backend.py`: loads bf16 in-process
  straight off the read-only `/kaggle/input` mount, no server, no
  quantization. torch/transformers are imported inside its loader function,
  never at module scope, so the module still imports on a GPU-free machine —
  now asserted by walking the AST rather than grepping the source, since the
  literal text `import torch` legitimately appears.
- `select_backend` splits the two Gemma backends: `gemma_local` keeps the
  HTTP client (Colab, where we start vLLM ourselves), `gemma_kaggle` gets the
  in-process one. Before this, `gemma_kaggle` returned the HTTP backend and
  would have failed silently into heuristics-only play — and the existing
  readiness gate would **not** have caught it, because it only checked for
  `FakeModelBackend`.
- One copy of the weights per process, behind a lock. `Swarm` builds a fresh
  agent per game and runs them concurrently; a per-instance load would have
  pulled 62.58 GB off disk once per game.
- The notebook's readiness gate moved into its own script, run as a
  subprocess. It loads the weights to prove they load — which means the
  kernel would still be holding 62.58 GB when `main.py` started its own copy
  had it stayed in-process. It also logs the GPU it actually got, performs
  one real generation, and projects `25 games x max_actions x latency`
  against the ~9 h limit, warning above 7 h. The per-game cap has never been
  calibrated against a real Gemma call.

### Still unverified

Every run here reported `is_competition_rerun: false`. Whether the scored
rerun allocates the same shape as a commit run pushed with
`--accelerator NvidiaRtxPro6000` has not been proven, and the only way to
prove it is a real submission. A submission's run cell should therefore log
its own GPU, so that if the hardware differs the result is diagnosable
rather than merely disappointing.

## Probe defect found and fixed

v1/v2's `/kaggle/input` walk pruned at `depth >= 6`. An attached Kaggle
model sits at `models/<owner>/<model>/<framework>/<instance>/<version>/` —
depth 6 before a single weight file — so the probe reported no weights at
all, which read exactly like "the model did not mount". Fixed in v3 by
raising the limit and adding a dedicated `attached_model` section that
locates weight shards wherever they land and sums their size. The finding
in section 4 above comes from the fixed version.
