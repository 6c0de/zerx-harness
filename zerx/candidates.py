"""Multi-candidate generation and deterministic scoring/selection --
exp-140-vlm-refinement infrastructure (STRATEGY.md 3.2). Off by default:
nothing here is called unless Config.candidate_count > 1 (see
zerx/policy.py's decide()). Never calls decide() itself and never touches
the environment -- pure candidate generation/scoring/selection over
backend.generate() responses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, List, Optional

from zerx.config import Config
from zerx.model_backend import ModelBackend
from zerx.policy import ParsedAction, parse_action
from zerx.types import ActionName


@dataclass(frozen=True)
class Candidate:
    raw_response: str
    parsed: Optional[ParsedAction]
    static_score: float


def static_candidate_score(candidate_raw: str, parsed: Optional[ParsedAction]) -> float:
    """Deterministic scoring, STRATEGY.md 3.2's factors adapted to a
    single-action ParsedAction -- see
    docs/superpowers/plans/2026-08-05-exp-140-vlm-refinement.md's "Design
    notes" for the full rationale of which factors were kept, merged, or
    dropped.
    """
    if parsed is None:
        return 0.0
    score = 1.0
    if parsed.repaired:
        score -= 0.2
    if parsed.action.name == ActionName.RESET:
        score -= 0.5
    return max(0.0, score)
