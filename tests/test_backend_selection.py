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


def test_select_backend_gemma_kaggle_returns_an_in_process_backend():
    """`gemma_kaggle` used to return `GemmaModelBackend` — an HTTP client for
    a vLLM server. That can never work on Kaggle: the environment probe
    (docs/superpowers/experiments/kaggle-env-probe.md) established that `vllm`
    is absent from the image, internet is disabled, and the competition's
    offline wheels ship `arc_agi`/`arcengine` but no vLLM. Every call would
    have raised ConnectionRefused and dropped the agent into heuristics-only
    silently — indistinguishable from having no model at all, which is the
    failure ARC-HANDOFF-001 is about.

    `gemma_local` still returns the HTTP backend: on Colab we start a real
    vLLM server ourselves.
    """
    from zerx.model_backend import TransformersModelBackend

    config = Config(
        backend="gemma_kaggle",
        model_revision="gemma-4-31b-it",
        model_path="/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1",
    )
    backend = select_backend(config)
    assert isinstance(backend, TransformersModelBackend)
    assert not isinstance(backend, GemmaModelBackend)
    assert backend.model_path == (
        "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1"
    )
    assert backend.dtype == "bfloat16"


def test_select_backend_gemma_kaggle_refuses_without_a_model_path():
    """A missing path must fail loudly at construction, not at the first
    action of a scored run.
    """
    import pytest

    with pytest.raises(ValueError, match="model_path"):
        select_backend(Config(backend="gemma_kaggle", platform="kaggle"))


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

    `select_backend` imports CerebrasDevBackend lazily, inside the
    cerebras_dev branch, so that the Kaggle bundle (which ships no
    `zerx/backends/`) stays importable — see
    tests/test_kaggle_bundle_importable.py. The patch target is therefore
    the defining module, which the lazy import resolves at call time, not an
    attribute of zerx.model_backend.
    """
    import zerx.backends.cerebras_dev as cerebras_module

    captured = {}

    class _RecordingCerebrasDevBackend:
        def __init__(self, *, model_id, platform):
            captured["model_id"] = model_id
            captured["platform"] = platform

    monkeypatch.setattr(cerebras_module, "CerebrasDevBackend", _RecordingCerebrasDevBackend)

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
