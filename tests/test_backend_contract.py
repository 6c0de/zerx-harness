from zerx.backends.cerebras_dev import CerebrasDevBackend
from zerx.model_backend import FakeModelBackend, GemmaModelBackend


def test_all_backends_expose_generate_method():
    fake = FakeModelBackend(responses=["x"])
    cerebras = CerebrasDevBackend(model_id="gemma-4-31b", api_key="sk-test", http_post=lambda *a, **k: {"choices": [{"message": {"content": "x"}}]})
    gemma = GemmaModelBackend(model_revision="gemma-4-31b-it")

    for backend in (fake, cerebras, gemma):
        assert callable(getattr(backend, "generate", None))


def test_fake_and_cerebras_return_str_from_generate():
    fake = FakeModelBackend(responses=["hello"])
    cerebras = CerebrasDevBackend(model_id="gemma-4-31b", api_key="sk-test", http_post=lambda *a, **k: {"choices": [{"message": {"content": "hello"}}]})
    assert isinstance(fake.generate("p"), str)
    assert isinstance(cerebras.generate("p"), str)
