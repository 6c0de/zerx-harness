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

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Protocol

from zerx.backends.cerebras_dev import CerebrasDevBackend
from zerx.config import Config


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
        return FakeModelBackend()
    if config.backend in ("gemma_local", "gemma_kaggle"):
        return GemmaModelBackend(config.model_revision, base_url=config.gemma_base_url)
    if config.backend == "cerebras_dev":
        return CerebrasDevBackend(model_id=config.model_revision, platform=config.platform)
    raise ValueError(f"Unknown backend: {config.backend!r}")
