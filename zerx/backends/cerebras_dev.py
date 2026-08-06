"""Development-only Cerebras Inference Cloud backend. Never selected when
Config.platform == "kaggle" (enforced both here and in zerx/config.py, as
defense in depth). Reads CEREBRAS_API_KEY directly from the environment —
the one deliberate exception to "only config.py reads env vars", because a
credential is not a config value and must never be serialized, hashed, or
logged (see AGENTS.md's "Cerebras development boundary").

As of August 2026, Cerebras serves `gemma-4-31b` in preview with both text
and image input support — verify this still holds (model catalog and
capabilities can change) before assuming either mode works.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

HttpPost = Callable[[str, dict, dict, float], dict]

_CEREBRAS_CHAT_URL = "https://api.cerebras.ai/v1/chat/completions"
# Confirmed live (2026-08-05): Cerebras's Cloudflare front returns HTTP 403
# (Cloudflare error 1010, a WAF "browser signature" block) for requests
# carrying Python urllib's default User-Agent, independent of credential or
# model-id correctness -- the identical request succeeded once a real
# User-Agent was set. Every request must send a non-default one.
_USER_AGENT = "zerx-harness-cerebras-dev/1.0"


def _default_http_post(url: str, headers: dict, json_body: dict, timeout: float) -> dict:
    import json
    import urllib.error
    import urllib.request

    request = urllib.request.Request(
        url, data=json.dumps(json_body).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # The bare HTTPError str() ("HTTP Error 401: Unauthorized") discards
        # the response body, which is where Cerebras (and most
        # OpenAI-compatible APIs) actually explain *why* -- e.g. "Incorrect
        # API key provided" vs. "model access denied" are very different
        # root causes that were previously indistinguishable from a single
        # generic 401.
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {body}") from exc


class CerebrasDevBackend:
    def __init__(
        self,
        model_id: str,
        api_version: str = "v1",
        request_timeout_seconds: float = 10.0,
        max_retries: int = 2,
        api_key: Optional[str] = None,
        http_post: Optional[HttpPost] = None,
        platform: str = "local",
    ) -> None:
        if platform == "kaggle":
            raise ValueError("cerebras_dev must never be constructed when platform=kaggle")
        self.model_id = model_id
        self.api_version = api_version
        self.request_timeout_seconds = request_timeout_seconds
        self.max_retries = max_retries
        self._api_key = api_key if api_key is not None else os.environ.get("CEREBRAS_API_KEY")
        self._http_post = http_post if http_post is not None else _default_http_post
        self.last_latency_seconds: Optional[float] = None

    @property
    def credential_present(self) -> bool:
        return self._api_key is not None

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        }
        json_body = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
        }

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            start = time.monotonic()
            try:
                response = self._http_post(
                    _CEREBRAS_CHAT_URL, headers, json_body, self.request_timeout_seconds
                )
                self.last_latency_seconds = time.monotonic() - start
                return response["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised if exhausted
                last_error = exc
        assert last_error is not None
        raise last_error
