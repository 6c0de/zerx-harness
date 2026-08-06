"""The only module allowed to load/call the Gemma model. Defines a narrow
Protocol so every other module (and all local tests) can depend on
`ModelBackend` without ever importing a real model. `GemmaModelBackend`
talks to a local vLLM OpenAI-compatible chat-completions server via an
injected `http_post` callable — the same pattern
`zerx/backends/cerebras_dev.py` uses for Cerebras — so this module itself
never imports vllm/torch/transformers and every local test runs without a
GPU or a running server. The real server is started only in
`notebooks/colab_gemma_smoke.ipynb`, on Colab/Kaggle.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

from zerx.config import Config

logger = logging.getLogger(__name__)


class ModelBackend(Protocol):
    def generate(self, prompt: str) -> str:
        ...


@dataclass
class FakeModelBackend:
    """Test double: returns scripted responses in order."""

    responses: List[str] = field(default_factory=list)
    _calls: List[str] = field(default_factory=list, init=False)

    def generate(self, prompt: str) -> str:
        self._calls.append(prompt)
        if not self.responses:
            raise RuntimeError("FakeModelBackend: no scripted responses left")
        return self.responses.pop(0)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    @property
    def last_prompt(self) -> str:
        if not self._calls:
            raise RuntimeError("FakeModelBackend: generate() was never called")
        return self._calls[-1]


HttpPost = Callable[[str, dict, dict, float], dict]

_DEFAULT_BASE_URL = "http://localhost:8000/v1/chat/completions"


def _default_http_post(url: str, headers: dict, json_body: dict, timeout: float) -> dict:
    import json
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(json_body).encode("utf-8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class GemmaModelBackend:
    """Real backend — talks to a local vLLM OpenAI-compatible server
    serving Gemma-4-31B (Kaggle model handle
    `google/gemma-4/Transformers/gemma-4-31b-it`, Apache 2.0). Constructed
    and exercised with a fake `http_post` in local unit tests; the real
    vLLM server is started only in `notebooks/colab_gemma_smoke.ipynb`.
    """

    def __init__(
        self,
        model_revision: str,
        base_url: str = _DEFAULT_BASE_URL,
        request_timeout_seconds: float = 60.0,
        max_retries: int = 2,
        http_post: Optional[HttpPost] = None,
    ) -> None:
        self.model_revision = model_revision
        self.base_url = base_url
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self._http_post = http_post if http_post is not None else _default_http_post
        self.last_latency_seconds: Optional[float] = None

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        json_body = {
            "model": self.model_revision,
            "messages": [{"role": "user", "content": prompt}],
        }
        last_error: Optional[Exception] = None
        for _ in range(self.max_retries):
            start = time.monotonic()
            try:
                response = self._http_post(
                    self.base_url, headers, json_body, self.request_timeout_seconds
                )
                self.last_latency_seconds = time.monotonic() - start
                return response["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised if exhausted
                last_error = exc
        assert last_error is not None
        raise last_error


ModelLoader = Callable[[str, str], "_LoadedModel"]


@dataclass
class _LoadedModel:
    """What a loader hands back: something callable plus what it cost."""

    generate: Callable[[str, int], str]
    description: str


# One process, one copy of the weights.
#
# The framework's Swarm builds a *fresh agent per game* and runs them
# concurrently, so a backend that loaded in __init__ would pull 62.58 GB off
# disk once per game and blow up VRAM on the second one. The cache is keyed
# by (path, dtype) and guarded by its own lock so two game threads racing
# into the first call still load exactly once.
_MODEL_CACHE: dict = {}
_MODEL_CACHE_LOCK = threading.Lock()


def clear_model_cache() -> None:
    """Drop every cached model. For tests that inject different loaders."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _default_transformers_loader(model_path: str, dtype: str) -> _LoadedModel:
    """Load Gemma with transformers + accelerate, straight off the read-only
    /kaggle/input mount.

    torch/transformers are imported *here*, not at module scope, for the same
    reason the Cerebras import is lazy: this module is bundled into the Kaggle
    notebook and imported by `agent/my_agent.py`, and every local test must
    keep running on a machine with no GPU and no transformers installed.

    Measured Kaggle facts this relies on
    (docs/superpowers/experiments/kaggle-env-probe.md): the card is an RTX PRO
    6000 Blackwell with ~96 GB, the weights are 62.58 GB of bf16, and
    `transformers` 5.0.0 + `accelerate` 1.13.0 are present while `vllm` and
    `bitsandbytes` are not and cannot be installed offline. So: bf16, no
    quantization, no server.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    torch_dtype = getattr(torch, dtype)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        device_map="auto",
        # /kaggle/working has ~21 GB free against 62.58 GB of weights, so
        # nothing may be copied off the read-only input mount.
        local_files_only=True,
    )
    model.eval()

    tokenizer = getattr(processor, "tokenizer", processor)

    def _generate(prompt: str, max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        # No <|think|> control token: Gemma 4 enables its thinking mode only
        # when that token opens the system prompt. We want one short JSON
        # object per call, and decide() allows exactly one model call per
        # action, so reasoning tokens are latency we cannot spend. The model
        # still emits an empty thought block, which policy's
        # _extract_json_object already sees through.
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # one legal action, not a creative sample
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )
        # Return only what the model added, never the echoed prompt.
        new_tokens = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    device = str(getattr(model, "device", "unknown"))
    return _LoadedModel(generate=_generate, description=f"{model_path} dtype={dtype} device={device}")


class TransformersModelBackend:
    """In-process Gemma, loaded with transformers — no HTTP, no server.

    `GemmaModelBackend` talks to a vLLM OpenAI-compatible server. That works
    on Colab, where vLLM can be installed. It cannot work on Kaggle: the
    probe established that `vllm` is absent from the image, internet is
    disabled, and the competition's offline wheels ship `arc_agi`/`arcengine`
    and friends but no vLLM. Pointing `gemma_kaggle` at an HTTP backend there
    means every call raises ConnectionRefused and the agent quietly plays
    heuristics-only — the exact silent-degradation failure this project keeps
    having to hunt down.

    Calls are serialized. The framework runs games concurrently in threads
    and a single HF model object is not safe to `generate()` from several at
    once; serializing costs throughput but a corrupted decode costs the run.
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        max_new_tokens: int = 96,
        loader: Optional[ModelLoader] = None,
    ) -> None:
        if not model_path:
            raise ValueError(
                "TransformersModelBackend needs a model_path. Set ZERX_MODEL_PATH "
                "(or Config.model_path) to the directory the weights mount at."
            )
        self.model_path = model_path
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._loader = loader if loader is not None else _default_transformers_loader
        self._call_lock = threading.Lock()
        self.last_latency_seconds: Optional[float] = None
        self.call_count = 0

    def _loaded(self) -> _LoadedModel:
        # Keyed on the weights alone, deliberately: "one copy of these weights
        # per process" is the property that matters, and a key including the
        # loader's id() could collide after garbage collection recycles it.
        # Production has exactly one loader; tests that swap it call
        # clear_model_cache().
        key = (self.model_path, self.dtype)
        with _MODEL_CACHE_LOCK:
            if key not in _MODEL_CACHE:
                start = time.monotonic()
                _MODEL_CACHE[key] = self._loader(self.model_path, self.dtype)
                logger.info(
                    "loaded model in %.1fs: %s",
                    time.monotonic() - start,
                    _MODEL_CACHE[key].description,
                )
            return _MODEL_CACHE[key]

    def warmup(self) -> float:
        """Load the weights and run one real generation, returning its latency.

        Called by the Kaggle notebook's readiness gate so an OOM or a missing
        checkpoint fails *before* gameplay rather than degrading the whole
        evaluation silently (AGENTS.md), and so the measured per-call latency
        is on the record before anyone picks a per-game action cap.
        """
        start = time.monotonic()
        self.generate("Reply with exactly this JSON and nothing else: {\"ok\": true}")
        return time.monotonic() - start

    def generate(self, prompt: str) -> str:
        loaded = self._loaded()
        with self._call_lock:
            start = time.monotonic()
            try:
                return loaded.generate(prompt, self.max_new_tokens)
            finally:
                self.last_latency_seconds = time.monotonic() - start
                self.call_count += 1


def select_backend(config: Config) -> ModelBackend:
    """Construct the ModelBackend named by config.backend
    ('fake' | 'cerebras_dev' | 'gemma_local' | 'gemma_kaggle'),
    forwarding config.platform to CerebrasDevBackend so its existing
    platform=='kaggle' lockout applies. Raises ValueError for any other
    backend string. 'fake' returns FakeModelBackend() with an empty
    responses list (deliberate: every call raises, exercising the
    fallback chain) -- not a general-purpose scripted-response
    constructor; callers who need scripted responses still construct
    FakeModelBackend(responses=[...]) directly.
    """
    if config.backend == "fake":
        if config.platform != "local":
            # AGENTS.md: model initialization problems "must fail before
            # gameplay rather than degrading an entire evaluation silently."
            # We deliberately do NOT raise here (that would turn a
            # misconfiguration into a zero-score run), but a fake backend off
            # `local` means every generate() raises and the agent plays with
            # no model at all — that must never be discovered only afterwards
            # by reading the leaderboard.
            logger.error(
                "backend='fake' selected on platform=%r: every model call will "
                "fail and the agent will run heuristics-only. Set ZERX_BACKEND "
                "to a real backend for a scored run.",
                config.platform,
            )
        return FakeModelBackend()
    if config.backend == "gemma_local":
        # Colab: a vLLM OpenAI-compatible server we started ourselves.
        return GemmaModelBackend(config.model_revision, base_url=config.gemma_base_url)
    if config.backend == "gemma_kaggle":
        # Kaggle: in-process, because there is nothing to talk to over HTTP.
        # vLLM is absent from the image, internet is disabled, and the
        # competition's offline wheels do not include it
        # (docs/superpowers/experiments/kaggle-env-probe.md). Routing this to
        # GemmaModelBackend would make every call raise ConnectionRefused and
        # drop the agent into heuristics-only without a word.
        return TransformersModelBackend(
            config.model_path, dtype=config.model_dtype, max_new_tokens=config.max_new_tokens
        )
    if config.backend == "cerebras_dev":
        # Imported lazily, INSIDE the one branch that can use it. The Kaggle
        # bundle (scripts/build_notebook.py) deliberately ships only
        # `zerx/*.py` and never `zerx/backends/`, so a module-level import of
        # this would make `import zerx.model_backend` — and therefore
        # `agent/my_agent.py` itself — raise ModuleNotFoundError on Kaggle,
        # where there is no internet and no pip rescue. Keeping it lazy
        # preserves the secret-hygiene exclusion AND keeps the bundle
        # importable; the `cerebras_dev` branch is unreachable on Kaggle
        # anyway (Config rejects it when platform=="kaggle").
        from zerx.backends.cerebras_dev import CerebrasDevBackend

        return CerebrasDevBackend(model_id=config.model_revision, platform=config.platform)
    raise ValueError(f"Unknown backend: {config.backend!r}")
