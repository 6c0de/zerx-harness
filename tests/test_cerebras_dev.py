import pytest

from zerx.backends.cerebras_dev import CerebrasDevBackend


def _fake_http_post(response_json, captured=None):
    def _post(url, headers, json_body, timeout):
        if captured is not None:
            captured.append({"url": url, "headers": headers, "json_body": json_body, "timeout": timeout})
        return response_json
    return _post


def _ok_response(text='{"action": "ACTION1"}'):
    return {"choices": [{"message": {"content": text}}]}


def test_generate_returns_message_content():
    backend = CerebrasDevBackend(
        model_id="gemma-4-31b",
        api_key="sk-test-not-real",
        http_post=_fake_http_post(_ok_response()),
    )
    assert backend.generate("prompt text") == '{"action": "ACTION1"}'


def test_generate_records_latency_not_credentials():
    backend = CerebrasDevBackend(
        model_id="gemma-4-31b",
        api_key="sk-test-not-real",
        http_post=_fake_http_post(_ok_response()),
    )
    backend.generate("prompt text")
    assert backend.last_latency_seconds is not None
    assert backend.last_latency_seconds >= 0.0


def test_credential_present_true_when_key_given():
    backend = CerebrasDevBackend(model_id="gemma-4-31b", api_key="sk-test-not-real")
    assert backend.credential_present is True


def test_credential_present_false_when_no_key_anywhere(monkeypatch):
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    backend = CerebrasDevBackend(model_id="gemma-4-31b")
    assert backend.credential_present is False


def test_request_never_contains_raw_key_in_body():
    captured = []
    backend = CerebrasDevBackend(
        model_id="gemma-4-31b",
        api_key="sk-test-not-real",
        http_post=_fake_http_post(_ok_response(), captured=captured),
    )
    backend.generate("prompt text")
    assert "sk-test-not-real" not in str(captured[0]["json_body"])
    assert captured[0]["headers"]["Authorization"] == "Bearer sk-test-not-real"


def test_retries_on_transient_failure_then_succeeds():
    calls = {"count": 0}

    def flaky_post(url, headers, json_body, timeout):
        calls["count"] += 1
        if calls["count"] < 2:
            raise TimeoutError("simulated transient failure")
        return _ok_response()

    backend = CerebrasDevBackend(
        model_id="gemma-4-31b", api_key="sk-test-not-real", http_post=flaky_post, max_retries=2
    )
    assert backend.generate("prompt text") == '{"action": "ACTION1"}'
    assert calls["count"] == 2


def test_raises_after_exhausting_retries():
    def always_fails(url, headers, json_body, timeout):
        raise TimeoutError("simulated permanent failure")

    backend = CerebrasDevBackend(
        model_id="gemma-4-31b", api_key="sk-test-not-real", http_post=always_fails, max_retries=1
    )
    with pytest.raises(TimeoutError):
        backend.generate("prompt text")


def test_never_constructs_when_platform_kaggle():
    with pytest.raises(ValueError):
        CerebrasDevBackend(model_id="gemma-4-31b", api_key="sk-test-not-real", platform="kaggle")
