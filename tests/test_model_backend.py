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


def _fake_http_post(response_json, captured=None):
    def _post(url, headers, json_body, timeout):
        if captured is not None:
            captured.append({"url": url, "headers": headers, "json_body": json_body, "timeout": timeout})
        return response_json
    return _post


def _ok_response(text='{"action": "ACTION1"}'):
    return {"choices": [{"message": {"content": text}}]}


def test_gemma_backend_generate_returns_message_content():
    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it",
        http_post=_fake_http_post(_ok_response()),
    )
    assert backend.generate("prompt text") == '{"action": "ACTION1"}'


def test_gemma_backend_generate_records_latency():
    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it",
        http_post=_fake_http_post(_ok_response()),
    )
    backend.generate("prompt text")
    assert backend.last_latency_seconds is not None
    assert backend.last_latency_seconds >= 0.0


def test_gemma_backend_sends_model_revision_and_prompt_in_body():
    captured = []
    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it",
        http_post=_fake_http_post(_ok_response(), captured=captured),
    )
    backend.generate("prompt text")
    assert captured[0]["json_body"]["model"] == "gemma-4-31b-it"
    assert captured[0]["json_body"]["messages"] == [{"role": "user", "content": "prompt text"}]


def test_gemma_backend_uses_configured_base_url():
    captured = []
    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it",
        base_url="http://localhost:9000/v1/chat/completions",
        http_post=_fake_http_post(_ok_response(), captured=captured),
    )
    backend.generate("prompt text")
    assert captured[0]["url"] == "http://localhost:9000/v1/chat/completions"


def test_gemma_backend_retries_on_transient_failure_then_succeeds():
    calls = {"count": 0}

    def flaky_post(url, headers, json_body, timeout):
        calls["count"] += 1
        if calls["count"] < 2:
            raise TimeoutError("simulated transient failure")
        return _ok_response()

    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it", http_post=flaky_post, max_retries=2
    )
    assert backend.generate("prompt text") == '{"action": "ACTION1"}'
    assert calls["count"] == 2


def test_gemma_backend_raises_after_exhausting_retries():
    def always_fails(url, headers, json_body, timeout):
        raise TimeoutError("simulated permanent failure")

    backend = GemmaModelBackend(
        model_revision="gemma-4-31b-it", http_post=always_fails, max_retries=1
    )
    with pytest.raises(TimeoutError):
        backend.generate("prompt text")


def test_gemma_backend_module_never_imports_vllm_torch_or_transformers():
    """The whole point of the injected-http_post pattern is that
    zerx/model_backend.py itself stays GPU/model-library-free, exactly
    like zerx/backends/cerebras_dev.py — verify by reading the module's
    own source, not by asserting something that's always true.
    """
    import zerx.model_backend as mb

    source = open(mb.__file__, encoding="utf-8").read()
    assert "import vllm" not in source
    assert "import torch" not in source
    assert "import transformers" not in source
