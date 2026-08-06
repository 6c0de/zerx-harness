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
    # The three above/below are name-based. A leaked key *value* pasted
    # without its variable name passed the scan entirely (ARC-HANDOFF-006),
    # which is the more likely accident: someone hardcodes the literal
    # while debugging and never types the variable name at all.
    (
        re.compile(r"\bcsk-[A-Za-z0-9_-]{16,}"),
        "Cerebras-style secret key literal (csk-...)",
    ),
    (
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
        "OpenAI-style secret key literal (sk-...)",
    ),
    (
        re.compile(
            r"(?i)\bauthorization[\"']?\s*[:=]\s*[\"']?\s*bearer\s+[A-Za-z0-9._-]{16,}"
        ),
        "hardcoded bearer token in an Authorization header",
    ),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token)\b\s*[:=]\s*"
            r"[\"'][A-Za-z0-9_\-]{20,}[\"']"
        ),
        "hardcoded credential assigned to an api-key/token variable",
    ),
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
