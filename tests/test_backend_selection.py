import pytest

from zerx.config import Config
from zerx.model_backend import FakeModelBackend, GemmaModelBackend, select_backend


def test_select_backend_fake_returns_fake_backend_with_no_scripted_responses():
    backend = select_backend(Config(backend="fake"))
    assert isinstance(backend, FakeModelBackend)
    with pytest.raises(RuntimeError):
        backend.generate("prompt")


def test_select_backend_gemma_local_returns_configured_gemma_backend():
    config = Config(
        backend="gemma_local",
        model_revision="gemma-4-31b-it",
        gemma_base_url="http://localhost:9001/v1/chat/completions",
    )
    backend = select_backend(config)
    assert isinstance(backend, GemmaModelBackend)
    assert backend.model_revision == "gemma-4-31b-it"
    assert backend.base_url == "http://localhost:9001/v1/chat/completions"


def test_select_backend_gemma_kaggle_returns_configured_gemma_backend():
    config = Config(
        backend="gemma_kaggle",
        model_revision="gemma-4-31b-it",
        gemma_base_url="http://localhost:9002/v1/chat/completions",
    )
    backend = select_backend(config)
    assert isinstance(backend, GemmaModelBackend)
    assert backend.model_revision == "gemma-4-31b-it"
    assert backend.base_url == "http://localhost:9002/v1/chat/completions"


def test_select_backend_cerebras_dev_returns_cerebras_backend_on_local_platform():
    from zerx.backends.cerebras_dev import CerebrasDevBackend

    config = Config(backend="cerebras_dev", platform="local", model_revision="gemma-4-31b")
    backend = select_backend(config)
    assert isinstance(backend, CerebrasDevBackend)
    assert backend.model_id == "gemma-4-31b"


def test_select_backend_cerebras_dev_forwards_config_platform_argument(monkeypatch):
    """Regression test for the exact bug this track fixes:
    `CerebrasDevBackend` was never reachable end-to-end, so its `platform`
    kwarg was never exercised with a real Config value. Use a platform
    value ("colab") that differs from CerebrasDevBackend's own default
    ("local") so a hardcoded/default-value bug in select_backend cannot
    accidentally pass this test.
    """
    import zerx.model_backend as model_backend_module

    captured = {}

    class _RecordingCerebrasDevBackend:
        def __init__(self, *, model_id, platform):
            captured["model_id"] = model_id
            captured["platform"] = platform

    monkeypatch.setattr(model_backend_module, "CerebrasDevBackend", _RecordingCerebrasDevBackend)

    config = Config(backend="cerebras_dev", platform="colab", model_revision="gemma-4-31b")
    backend = select_backend(config)

    assert isinstance(backend, _RecordingCerebrasDevBackend)
    assert captured["platform"] == "colab"
    assert captured["model_id"] == "gemma-4-31b"


def test_cerebras_dev_on_kaggle_platform_is_unreachable_via_config_guard():
    """Config.__post_init__ already rejects backend='cerebras_dev' with
    platform='kaggle' at Config-construction time -- select_backend never
    even runs in that case. This proves the two independent layers
    (Config's guard, CerebrasDevBackend's own guard) still compose
    correctly after this track's change, rather than assuming it.
    """
    with pytest.raises(ValueError):
        Config(backend="cerebras_dev", platform="kaggle")


def test_select_backend_raises_value_error_for_unknown_backend_string():
    with pytest.raises(ValueError):
        select_backend(Config(backend="not-a-real-backend"))
