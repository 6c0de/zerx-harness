"""JSON action parsing, bounded deterministic repair, and legal-action
validation. No model calls live here — see `decide()` (added in Task 12)
for the orchestrator that calls the model backend.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import FrozenSet, Optional

from zerx.types import Action, ActionName


@dataclass(frozen=True)
class ParsedAction:
    action: Action
    repaired: bool


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw: str) -> Optional[str]:
    """Deterministic repair: strip markdown code fences and pull out the
    first {...} substring. No model call, no retried reasoning.
    """
    stripped = raw.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped)
    stripped = re.sub(r"```$", "", stripped).strip()
    match = _JSON_OBJECT_RE.search(stripped)
    return match.group(0) if match else None


def parse_action(raw: str, legal_actions: FrozenSet[ActionName]) -> Optional[ParsedAction]:
    """Try a direct `json.loads` first; on failure, attempt exactly one
    deterministic extraction/repair and retry. Returns None (never raises)
    if both attempts fail or the result doesn't validate — callers fall
    back per the documented fallback chain.
    """
    for attempt, candidate_text in enumerate((raw, _extract_json_object(raw))):
        if candidate_text is None:
            continue
        try:
            payload = json.loads(candidate_text)
        except (json.JSONDecodeError, TypeError):
            continue
        action = _validate_payload(payload, legal_actions)
        if action is not None:
            return ParsedAction(action=action, repaired=(attempt == 1))
    return None


def _validate_payload(payload: object, legal_actions: FrozenSet[ActionName]) -> Optional[Action]:
    if not isinstance(payload, dict) or "action" not in payload:
        return None
    name_raw = payload["action"]
    if not isinstance(name_raw, str) or name_raw not in ActionName.__members__:
        return None
    name = ActionName[name_raw]
    if name not in legal_actions:
        return None
    data = payload.get("data") or {}
    try:
        if name == ActionName.ACTION6:
            return Action(name=name, x=int(data["x"]), y=int(data["y"]))
        return Action(name=name)
    except (KeyError, ValueError, TypeError):
        return None
