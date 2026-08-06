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


def test_gemma_backend_module_imports_no_model_library_at_module_scope():
    """`import zerx.model_backend` must never require torch/transformers/vllm.

    This used to be checked by grepping the module source for "import torch".
    That proxy broke once `TransformersModelBackend` landed: it genuinely does
    import torch and transformers, but *inside* its loader function, which
    only runs on Kaggle where they exist. The text is present; the invariant
    is not violated.

    So check the invariant itself — no module-scope import of those libraries
    — by walking the AST rather than the characters. `test_bundled_zerx_
    package_imports_without_the_backends_subpackage` covers the behavioural
    half: the bundle really does import on a machine without them.
    """
    import ast

    import zerx.model_backend as mb

    tree = ast.parse(open(mb.__file__, encoding="utf-8").read())
    forbidden = {"vllm", "torch", "transformers", "accelerate"}

    module_level_imports = set()
    for node in tree.body:  # top level only — nested ones are the whole point
        if isinstance(node, ast.Import):
            module_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_level_imports.add(node.module.split(".")[0])

    leaked = forbidden & module_level_imports
    assert not leaked, f"module-scope import of {sorted(leaked)} breaks GPU-free import"
