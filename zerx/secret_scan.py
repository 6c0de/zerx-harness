"""Scans generated-artifact text (a notebook's source, a built package's
files) for leaked Cerebras credentials/endpoints before it's allowed to
ship anywhere near Kaggle. See AGENTS.md's hard safeguards.
"""
from __future__ import annotations

import re
from typing import Iterable, List

_STATIC_PATTERNS = (
    (re.compile(r"api\.cerebras\.ai"), "reference to api.cerebras.ai"),
    (re.compile(r"CEREBRAS_API_KEY"), "reference to CEREBRAS_API_KEY"),
)


def scan_for_secrets(text: str, extra_patterns: Iterable[str] = ()) -> List[str]:
    findings: List[str] = []
    for pattern, description in _STATIC_PATTERNS:
        if pattern.search(text):
            findings.append(description)
    for secret in extra_patterns:
        if secret and secret in text:
            findings.append("literal secret value found in artifact")
    return findings
