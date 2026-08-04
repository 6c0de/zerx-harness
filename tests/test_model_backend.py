import pytest

from zerx.model_backend import FakeModelBackend, GemmaModelBackend


def test_fake_backend_returns_scripted_responses_in_order():
    backend = FakeModelBackend(responses=["first", "second"])
    assert backend.generate("prompt-a") == "first"
    assert backend.generate("prompt-b") == "second"


def test_fake_backend_raises_when_exhausted():
    backend = FakeModelBackend(responses=[])
    with pytest.raises(RuntimeError):
        backend.generate("prompt")


def test_fake_backend_tracks_call_count_and_last_prompt():
    backend = FakeModelBackend(responses=["a", "b"])
    backend.generate("first-prompt")
    backend.generate("second-prompt")
    assert backend.call_count == 2
    assert backend.last_prompt == "second-prompt"


def test_gemma_backend_constructs_without_loading_model():
    backend = GemmaModelBackend(model_revision="gemma-4-31b-it")
    assert backend.model_revision == "gemma-4-31b-it"


def test_gemma_backend_generate_not_yet_implemented():
    backend = GemmaModelBackend(model_revision="gemma-4-31b-it")
    with pytest.raises(NotImplementedError):
        backend.generate("prompt")
