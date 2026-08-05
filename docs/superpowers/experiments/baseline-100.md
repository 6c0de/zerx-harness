# baseline-100 — first Gemma-4-31B model-in-loop smoke game

- Date: 2026-08-04
- Base commit: `89126ecf3ea40e203567d5203669cc47ac35874c` (branch `day2-colab-gemma-baseline-100`)
- Model: `google/gemma-4-31B-it` (Apache 2.0; NOT the Kaggle Models UI slug
  `google/gemma-4/Transformers/gemma-4-31b-it`, which is not a valid HF Hub
  repo id — see `docs/superpowers/experiments/baseline-000.md` addendum and
  `scripts/build_colab_notebook.py` commit `8d742c5` for the full story)
- GPU: `NVIDIA A100-SXM4-80GB` (Colab Pro, High-RAM runtime shape)
- Precision/dtype: `bfloat16` (no quantization — 80GB VRAM comfortably fits
  ~61.4GB of bf16 weights + KV cache at the smoke test's capped
  `--max-model-len 8192`; the notebook's default before this run was 4-bit
  bitsandbytes quantization, sized for a 40GB A100 — swap back if a future
  run lands on the 40GB SKU instead)
- Backend: `gemma_local` (`zerx/model_backend.py`'s `GemmaModelBackend`,
  vLLM `0.26.0` OpenAI-compatible server on `localhost:8000`, installed via
  `uv pip install --reinstall --torch-backend=auto`)
- Game: `ls20`
- Environment setup: **succeeded**. `nvidia-smi` confirmed the real
  80GB A100; `torch.__version__`/`torch.version.cuda` confirmed a
  matching CUDA 13.0 build (no `libcudart` linkage error); the vLLM
  server came up and reported ready; the notebook reached its
  Drive-save cell without a Python exception.
- **Known gap — per-game play outcome not captured.** The notebook's
  results cell (`scripts/build_colab_notebook.py`'s `save_results_cell`)
  only recorded environment/setup metadata (model, GPU, dtype, commit) —
  it did not capture `play_local.py`'s actual per-game output
  (`state=...`, `levels_completed=...`, `actions=...`). This is a real
  gap in the notebook's own design, not a play failure — the setup
  reaching the save-results cell at all means `choose_action` ran against
  the live model without an unhandled exception (the whole point of the
  smoke test), but the exact action count / terminal state / whether any
  `NotImplementedError`-style fallback still fired mid-run is not
  recorded here. Fix scoped as a follow-up to
  `scripts/build_colab_notebook.py` (capture `agent.frames[-1]` /
  `agent.action_counter` into the saved JSON, matching
  `scripts/play_local.py`'s own per-game summary line) — not done in this
  session, noted here rather than guessed at.
- Conclusion: **environment/packaging validated, pipeline result
  unmeasured**. This satisfies `AGENTS.md`'s Colab gate for "load the
  exact Gemma revision" and confirms `GemmaModelBackend` talks to a real
  model without exception — but does NOT yet constitute a scored
  `baseline-100` result in the STRATEGY.md §7 sense ("End-to-end stable
  and reproducible"). Re-run with the results-capture fix before treating
  any RHAE/action-efficiency number as real. `investigate` per
  STRATEGY.md §7.1 ("a measurement/logging defect exists"), not `keep` or
  `revert`.
