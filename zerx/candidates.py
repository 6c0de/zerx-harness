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


def generate_candidates(
    backend: ModelBackend,
    prompt: str,
    legal_actions: FrozenSet[ActionName],
    count: int,
) -> List[Candidate]:
    """Calls backend.generate(prompt) `count` times, parses each response,
    scores each deterministically. Never calls an arbiter -- that's a
    separate, explicitly optional step (see select_candidate()). A
    response that fails to parse gets parsed=None and a 0.0 score but does
    not stop generation of the remaining candidates.
    """
    candidates: List[Candidate] = []
    for _ in range(count):
        raw = backend.generate(prompt)
        try:
            parsed = parse_action(raw, legal_actions)
        except Exception:
            parsed = None
        candidates.append(
            Candidate(
                raw_response=raw,
                parsed=parsed,
                static_score=static_candidate_score(raw, parsed),
            )
        )
    return candidates


def select_best_candidate(candidates: List[Candidate]) -> Optional[Candidate]:
    """Deterministic selection -- no LLM arbiter. This is what actually
    gets used when arbiter_on is False (the default, and the only
    supported mode for this track). Ties break toward the earliest
    candidate in generation order: Python's max() only replaces its
    running best on a strictly greater key, so the first item with the
    maximum score wins.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda c: c.static_score)


def _build_arbiter_prompt(candidates: List[Candidate]) -> str:
    lines = "\n".join(
        f"{i}: {c.raw_response!r} (static_score={c.static_score:.2f})"
        for i, c in enumerate(candidates)
    )
    return (
        "Multiple candidate actions were generated for the same game state. "
        "Pick the single best one.\n"
        f"{lines}\n\n"
        "Respond with exactly one integer: the index of the best candidate."
    )


def _select_with_arbiter(candidates: List[Candidate], arbiter: ModelBackend) -> Optional[Candidate]:
    valid = [c for c in candidates if c.parsed is not None]
    if not valid:
        return None
    try:
        raw = arbiter.generate(_build_arbiter_prompt(valid))
        index = int(raw.strip())
    except Exception:
        return None
    if 0 <= index < len(valid):
        return valid[index]
    return None


def select_candidate(
    candidates: List[Candidate],
    config: Config,
    arbiter: Optional[ModelBackend] = None,
) -> Optional[Candidate]:
    """Entry point decide() calls (Task 6). Deterministic by default
    (select_best_candidate). Only consults `arbiter` when
    config.arbiter_on is True AND an arbiter backend is actually supplied
    -- both conditions required, so passing an arbiter with the flag off
    is still a true no-op. STRATEGY.md 3.2: the arbiter is the
    lowest-priority, most speculative part of this track's scope, so on
    any arbiter failure (bad output, exception, no valid candidates) this
    falls back to the deterministic pick rather than raising.
    """
    if not candidates:
        return None
    if config.arbiter_on and arbiter is not None:
        picked = _select_with_arbiter(candidates, arbiter)
        if picked is not None:
            return picked
    return select_best_candidate(candidates)
