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


def test_catches_a_bare_key_value_with_no_variable_name():
    """ARC-HANDOFF-006: the scanner matched only `CEREBRAS_API_KEY` and
    `api.cerebras.ai`, so a key literal pasted without its variable name --
    the likelier accident -- shipped clean.
    """
    findings = scan_for_secrets('client = Client("csk-abcdef0123456789abcdef0123456789")')
    assert findings


def test_catches_an_openai_style_literal_and_a_bearer_header():
    assert scan_for_secrets("KEY = 'sk-abcdefghijklmnopqrstuvwxyz0123'")
    assert scan_for_secrets('headers = {"Authorization": "Bearer abcdefghijklmnop1234"}')


def test_catches_a_credential_assigned_to_a_generic_api_key_variable():
    assert scan_for_secrets('api_key = "aaaaaaaaaaaaaaaaaaaaaaaaaaaa"')


def test_does_not_fire_on_ordinary_source_code():
    clean = (
        "def generate(self, prompt: str) -> str:\n"
        "    headers = {'Content-Type': 'application/json'}\n"
        "    return self._http_post(self.base_url, headers, body, 60.0)\n"
    )
    assert scan_for_secrets(clean) == []


def test_the_scanner_does_not_flag_its_own_pattern_definitions():
    """The build gate scans every bundled zerx module. If secret_scan.py's
    own regex sources matched, every build would fail permanently on a
    self-referential false positive.
    """
    import pathlib

    import zerx.secret_scan as module

    body = pathlib.Path(module.__file__).read_text()
    body = body.replace("CEREBRAS_API_KEY", "").replace("api.cerebras.ai", "")
    assert scan_for_secrets(body) == []
