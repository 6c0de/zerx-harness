import io
import urllib.error

import pytest

from zerx.backends.cerebras_dev import CerebrasDevBackend, _default_http_post


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


def test_default_http_post_includes_the_response_body_on_http_error(monkeypatch):
    """Cerebras's (and most OpenAI-compatible APIs') error responses carry
    a JSON body explaining WHY a request was rejected, e.g. "Incorrect
    API key provided" vs. "model access denied" -- very different root
    causes. The bare HTTPError str() (e.g. "HTTP Error 401: Unauthorized")
    discards that body entirely, which is exactly the gap that made a real
    live 401 undiagnosable beyond "the server said no" (see
    docs/HANDOFF.md's Decision.model_error work). Confirmed by reading
    _default_http_post directly: it never calls exc.read() before this
    fix.
    """

    def _fake_urlopen(request, timeout=None):
        body = b'{"error": {"message": "Incorrect API key provided"}}'
        raise urllib.error.HTTPError(
            url="https://api.cerebras.ai/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(body),
        )

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError, match="Incorrect API key provided"):
        _default_http_post("https://api.cerebras.ai/v1/chat/completions", {}, {}, 10.0)


def test_request_sends_a_non_default_user_agent():
    """Confirmed live (2026-08-05): Cerebras's Cloudflare WAF returns HTTP
    403 (Cloudflare error 1010, "browser signature" block) for requests
    carrying urllib's default User-Agent ("Python-urllib/3.x"), even with
    a valid credential and a valid model id -- verified by retrying the
    identical request with an explicit User-Agent header, which succeeded
    (HTTP 200) with the same key. Must send a real, non-default
    User-Agent on every request or every cerebras_dev call fails
    regardless of credential/model correctness.
    """
    captured = []
    backend = CerebrasDevBackend(
        model_id="gemma-4-31b",
        api_key="sk-test-not-real",
        http_post=_fake_http_post(_ok_response(), captured=captured),
    )
    backend.generate("prompt text")
    user_agent = captured[0]["headers"].get("User-Agent", "")
    assert user_agent
    assert "python-urllib" not in user_agent.lower()
