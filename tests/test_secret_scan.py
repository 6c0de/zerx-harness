from zerx.secret_scan import scan_for_secrets


def test_clean_text_has_no_findings():
    assert scan_for_secrets("this notebook loads gemma from /kaggle/input") == []


def test_flags_cerebras_endpoint_reference():
    findings = scan_for_secrets("client = Client(base_url='https://api.cerebras.ai/v1')")
    assert any("api.cerebras.ai" in f for f in findings)


def test_flags_cerebras_api_key_env_var_name():
    findings = scan_for_secrets('CEREBRAS_API_KEY = "sk-something"')
    assert any("CEREBRAS_API_KEY" in f for f in findings)


def test_flags_extra_secret_value_if_present():
    findings = scan_for_secrets("some text sk-my-actual-key-123 more text", extra_patterns=["sk-my-actual-key-123"])
    assert len(findings) == 1


def test_does_not_flag_extra_secret_value_if_absent():
    findings = scan_for_secrets("clean text here", extra_patterns=["sk-my-actual-key-123"])
    assert findings == []
