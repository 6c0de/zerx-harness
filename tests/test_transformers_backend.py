"""Tests for `TransformersModelBackend` — the in-process Kaggle backend.

Why this backend exists at all: `GemmaModelBackend` is an HTTP client for a
vLLM OpenAI-compatible server. That works on Colab, where we start such a
server ourselves. It cannot work on Kaggle — the environment probe
(docs/superpowers/experiments/kaggle-env-probe.md) established that `vllm` is
absent from the image, internet is disabled, and the competition's offline
wheels ship `arc_agi`/`arcengine` but no vLLM. Routing `gemma_kaggle` at HTTP
meant every call raised ConnectionRefused and the agent played heuristics-only
without a word.

Every test here injects a fake loader, so none of them needs torch,
transformers, or a GPU.
"""
from __future__ import annotations

import threading
import time

import pytest

from zerx.model_backend import (
    TransformersModelBackend,
    _LoadedModel,
    clear_model_cache,
)


@pytest.fixture(autouse=True)
def _clean_model_cache():
    """The model cache is process-wide by design; keep tests independent."""
    clear_model_cache()
    yield
    clear_model_cache()


def make_loader(response: str = '{"action": "ACTION1"}', load_calls: list | None = None):
    def loader(model_path: str, dtype: str) -> _LoadedModel:
        if load_calls is not None:
            load_calls.append((model_path, dtype))
        return _LoadedModel(
            generate=lambda prompt, max_new_tokens: response,
            description=f"fake({model_path})",
        )

    return loader


def test_generate_returns_the_models_text_and_records_latency():
    backend = TransformersModelBackend("/weights", loader=make_loader())

    assert backend.generate("prompt") == '{"action": "ACTION1"}'
    assert backend.call_count == 1
    assert backend.last_latency_seconds is not None


def test_weights_load_once_per_process_not_once_per_game():
    """The framework's Swarm builds a fresh agent per game and runs them
    concurrently. Loading in __init__ (or per instance) would pull 62.58 GB
    off disk once per game and exhaust VRAM on the second.
    """
    load_calls: list = []
    loader = make_loader(load_calls=load_calls)

    for _ in range(4):
        TransformersModelBackend("/weights", loader=loader).generate("prompt")

    assert load_calls == [("/weights", "bfloat16")]


def test_different_weights_are_cached_separately():
    load_calls: list = []
    loader = make_loader(load_calls=load_calls)

    TransformersModelBackend("/weights-a", loader=loader).generate("p")
    TransformersModelBackend("/weights-b", loader=loader).generate("p")

    assert len(load_calls) == 2


def test_nothing_is_loaded_until_the_first_call():
    """Construction happens per agent; loading 62.58 GB must not."""
    load_calls: list = []
    TransformersModelBackend("/weights", loader=make_loader(load_calls=load_calls))
    assert load_calls == []


def test_missing_model_path_fails_at_construction():
    """Loudly, and before gameplay — not at the first action of a scored run."""
    with pytest.raises(ValueError, match="model_path"):
        TransformersModelBackend("", loader=make_loader())
    with pytest.raises(ValueError, match="model_path"):
        TransformersModelBackend(None, loader=make_loader())  # type: ignore[arg-type]


def test_warmup_performs_a_real_generation_and_returns_its_cost():
    """The Kaggle readiness gate calls this so an OOM or a missing checkpoint
    fails before any game runs, and so per-call latency is on the record
    before anyone picks a per-game action cap.
    """
    backend = TransformersModelBackend("/weights", loader=make_loader())

    elapsed = backend.warmup()

    assert elapsed >= 0
    assert backend.call_count == 1, "warmup must actually call the model"


def test_a_loader_failure_propagates_rather_than_degrading_silently():
    def broken_loader(model_path: str, dtype: str) -> _LoadedModel:
        raise RuntimeError("CUDA out of memory")

    backend = TransformersModelBackend("/weights", loader=broken_loader)

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        backend.generate("prompt")


def test_max_new_tokens_reaches_the_model():
    seen: list = []

    def loader(model_path: str, dtype: str) -> _LoadedModel:
        def generate(prompt: str, max_new_tokens: int) -> str:
            seen.append(max_new_tokens)
            return "{}"

        return _LoadedModel(generate=generate, description="fake")

    TransformersModelBackend("/weights", max_new_tokens=17, loader=loader).generate("p")

    assert seen == [17]


def test_concurrent_calls_are_serialised():
    """Games run concurrently in threads and one HF model object is not safe
    to generate() from several at once. Serialising costs throughput; a
    corrupted decode costs the run.
    """
    overlaps = []
    active = []
    active_lock = threading.Lock()

    def loader(model_path: str, dtype: str) -> _LoadedModel:
        def generate(prompt: str, max_new_tokens: int) -> str:
            with active_lock:
                active.append(1)
                overlaps.append(len(active))
            time.sleep(0.02)
            with active_lock:
                active.pop()
            return "{}"

        return _LoadedModel(generate=generate, description="fake")

    backend = TransformersModelBackend("/weights", loader=loader)
    threads = [threading.Thread(target=backend.generate, args=("p",)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert overlaps, "the model was never called"
    assert max(overlaps) == 1, f"calls overlapped: {overlaps}"
    assert backend.call_count == 6


def test_satisfies_the_same_model_backend_protocol_as_the_others():
    """AGENTS.md requires every backend to satisfy one narrow protocol, so
    policy code never branches on which one it got.
    """
    from zerx.model_backend import FakeModelBackend, GemmaModelBackend

    backend = TransformersModelBackend("/weights", loader=make_loader())
    for other in (FakeModelBackend(responses=["{}"]), GemmaModelBackend("rev")):
        assert hasattr(other, "generate")
    assert isinstance(backend.generate("p"), str)
