"""JSON action parsing, bounded deterministic repair, and legal-action
validation. No model calls live here — see `decide()` (added in Task 12)
for the orchestrator that calls the model backend.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from typing import FrozenSet, Optional, Sequence, Tuple

from zerx.budget import BudgetSignal, evaluate_budget
from zerx.config import Config
from zerx.heuristics import ClickCandidate, DeadSignatureTracker, rank_click_candidates
from zerx.memory import MemoryState, maybe_refresh
from zerx.model_backend import ModelBackend
from zerx.perception import PerceptionResult, perceive
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


@dataclass(frozen=True)
class Decision:
    action: Action
    source: str  # "model" | "heuristic" | "fallback_heuristic" | "fallback_deterministic" | "fallback_random" | "reset"
    repaired: bool = False
    budget: Optional[BudgetSignal] = None
    target_object_label: Optional[str] = None
    raw_response: Optional[str] = None  # the model's raw text, when a
    # model call happened this step -- populated even on a failed parse,
    # so tooling (zerx/trace.py) can show what the model actually said.
    model_error: Optional[str] = None  # str(exc) when a model call was
    # attempted and raised (auth/network/backend failure, distinct from a
    # successful-but-unparseable response, which populates raw_response
    # instead) -- otherwise every model-call failure was silently
    # discarded with zero visibility into why a step fell back.


_FALLBACK_PREFERENCE = (
    ActionName.ACTION5,
    ActionName.ACTION1,
    ActionName.ACTION2,
    ActionName.ACTION3,
    ActionName.ACTION4,
    ActionName.ACTION6,
)


def _deterministic_fallback(
    legal_actions: FrozenSet[ActionName], grid_size: int = 64
) -> Optional[Action]:
    for name in _FALLBACK_PREFERENCE:
        if name in legal_actions:
            if name == ActionName.ACTION6:
                return Action(name=name, x=grid_size // 2, y=grid_size // 2)
            return Action(name=name)
    if ActionName.RESET in legal_actions:
        return Action(name=ActionName.RESET)
    return None


def _random_fallback(legal_actions: FrozenSet[ActionName], grid_size: int = 64) -> Action:
    name = random.choice(tuple(legal_actions))
    if name == ActionName.ACTION6:
        return Action(
            name=name, x=random.randint(0, grid_size - 1), y=random.randint(0, grid_size - 1)
        )
    return Action(name=name)


# A 64x64 frame is legally allowed to segment into thousands of
# single-cell objects (a two-colour checkerboard yields 4096), which
# rendered ~49k tokens of object table into a single prompt — enough to
# overflow the context window and fail or truncate the call outright.
# The ranked click-candidate list, not this table, is what the model acts
# on; the table is supporting detail, so bounding it costs little.
_MAX_PROMPT_OBJECTS = 60


def build_prompt(
    perception: PerceptionResult,
    memory: MemoryState,
    candidates: Sequence[ClickCandidate] = (),
    legal_actions: FrozenSet[ActionName] = frozenset(),
    budget: Optional[BudgetSignal] = None,
) -> str:
    """STRATEGY.md §8: show the model its top ranked click candidates, not
    just the raw object table, so it can select ACTION6 by label instead of
    guessing coordinates — directly reduces the coordinate-hallucination
    failure mode.

    Also surfaces `legal_actions` (AGENTS.md's required control flow step 2:
    "do not assume all games support the same actions") and `budget` as a
    strategy signal (AGENTS.md step 7: never a forced/invented move, just a
    hint). Deliberately does not describe what ACTION1-ACTION5 *do* —
    AGENTS.md forbids hard-coding their semantics since they vary by game.
    """
    shown_objects = perception.objects[:_MAX_PROMPT_OBJECTS]
    object_lines = (
        "\n".join(
            f"- {obj.label}: color={obj.color} size={obj.size} bbox={obj.bbox}"
            for obj in shown_objects
        )
        or "(no non-background objects)"
    )
    omitted = len(perception.objects) - len(shown_objects)
    if omitted > 0:
        # Tell the model the table is truncated rather than letting it assume
        # it is exhaustive — a silently-cut list invites false "there is
        # nothing else on the board" reasoning.
        object_lines += (
            f"\n(+{omitted} more objects not listed; "
            f"showing the first {_MAX_PROMPT_OBJECTS} of {len(perception.objects)})"
        )
    # Confirmed root cause of a real 0.0 Colab run (docs/HANDOFF.md item 6's
    # live reproduction, and item 10's real gemma-4-31b-it trace): showing
    # this section unconditionally -- even on a turn where ACTION6 isn't
    # legal -- contradicts the "Legal actions this turn" line right below
    # it. The model's own raw output confirmed this: it repeatedly proposed
    # {"action": "ACTION6", ...} on turns where ACTION6 was never legal,
    # got correctly rejected by parse_action(), and fell through to the
    # fallback chain 76 of 101 steps in that trace. Gate the whole section
    # on ACTION6 actually being legal this turn, same as decide()'s own
    # heuristic path already does (`if candidates and ActionName.ACTION6 in
    # legal_actions`) -- there is nothing to click toward otherwise.
    candidates_section = ""
    if ActionName.ACTION6 in legal_actions:
        candidate_lines = (
            "\n".join(
                f"- {c.object_label}: click (x={c.x}, y={c.y}), score={c.score:.2f}"
                for c in candidates[:5]
            )
            or "(no click candidates)"
        )
        candidates_section = (
            "Ranked click candidates (if you choose ACTION6, prefer one of "
            f"these exact coordinates over guessing):\n{candidate_lines}\n\n"
        )
    legal_action_names = ", ".join(sorted(a.value for a in legal_actions)) or "(none)"
    budget_line = (
        f"Actions taken so far: {budget.actions_taken} (soft cap {budget.soft_cap}). "
        "This is a strategy signal only, not a required move."
        if budget is not None
        else "(no budget signal)"
    )
    return (
        "You are playing a grid-based puzzle game.\n"
        f"Grid:\n{perception.ascii_grid}\n\n"
        f"Objects:\n{object_lines}\n\n"
        f"{candidates_section}"
        f"Legal actions this turn: {legal_action_names}\n\n"
        f"Action budget: {budget_line}\n\n"
        f"What you've learned so far: {memory.summary or '(nothing yet)'}\n\n"
        'Respond with exactly one JSON object: {"action": "<ACTION_NAME>", '
        '"data": {"x": <int>, "y": <int>}} (data only required for ACTION6), '
        "using one of the legal actions listed above."
    )


def decide(
    frame: GameFrame,
    history: Tuple[GameFrame, ...],
    memory: MemoryState,
    dead_signatures: DeadSignatureTracker,
    config: Config,
    backend: ModelBackend,
    actions_taken: int,
) -> Tuple[Decision, MemoryState]:
    """Implements the required control flow: terminal check, perception,
    heuristics, one bounded model call with deterministic repair, budget as
    a strategy signal only (never a forced/invented move), and a strict
    validated fallback chain. Never raises.
    """
    if frame.is_game_over:
        return Decision(action=Action(name=ActionName.RESET), source="reset"), memory

    legal_actions = frame.legal_actions
    perception = perceive(frame, history)
    budget = evaluate_budget(actions_taken, config.budget_soft_cap)

    candidates = rank_click_candidates(perception, dead_signatures)
    heuristic_action: Optional[Action] = None
    if candidates and ActionName.ACTION6 in legal_actions:
        top = candidates[0]
        heuristic_confident = (
            config.heuristic_first and top.score >= config.heuristic_confidence_threshold
        )
        # AGENTS.md step 7: apply the action-budget policy as a strategy
        # signal. Purely additive alongside `heuristic_confident` above —
        # when the budget is running low relative to `budget_soft_cap`, and
        # there's an actual candidate worth executing (score > 0.0), prefer
        # acting on it over spending a model call. Never invents a move:
        # still gated on a real candidate that passed rank_click_candidates.
        budget_favors_execution = budget.should_favor_execution and top.score > 0.0
        if heuristic_confident or budget_favors_execution:
            heuristic_action = Action(name=ActionName.ACTION6, x=top.x, y=top.y)

    new_memory = memory
    if config.memory_on:
        new_memory = maybe_refresh(
            memory,
            recent_context=perception.ascii_grid,
            summarizer=lambda prev, ctx: prev,  # deterministic no-op for the local skeleton
            refresh_interval=config.memory_refresh_interval,
        )

    if heuristic_action is not None:
        return (
            Decision(
                action=heuristic_action,
                source="heuristic",
                budget=budget,
                target_object_label=top.object_label,
            ),
            new_memory,
        )

    raw_response: Optional[str] = None
    model_error: Optional[str] = None
    if config.candidate_count > 1:
        try:
            from zerx.candidates import generate_candidates, select_candidate

            prompt = build_prompt(perception, new_memory, candidates, legal_actions, budget)
            model_candidates = generate_candidates(
                backend, prompt, legal_actions, config.candidate_count
            )
            best = select_candidate(model_candidates, config)
            parsed = best.parsed if best is not None else None
        except Exception as exc:
            parsed = None
            model_error = f"{type(exc).__name__}: {exc}"
    else:
        try:
            raw_response = backend.generate(
                build_prompt(perception, new_memory, candidates, legal_actions, budget)
            )
            parsed = parse_action(raw_response, legal_actions)
        except Exception as exc:
            parsed = None
            model_error = f"{type(exc).__name__}: {exc}"

    if parsed is not None:
        return (
            Decision(
                action=parsed.action,
                source="model",
                repaired=parsed.repaired,
                budget=budget,
                raw_response=raw_response,
                model_error=model_error,
            ),
            new_memory,
        )

    if candidates and ActionName.ACTION6 in legal_actions:
        top = candidates[0]
        return (
            Decision(
                action=Action(name=ActionName.ACTION6, x=top.x, y=top.y),
                source="fallback_heuristic",
                budget=budget,
                target_object_label=top.object_label,
                raw_response=raw_response,
                model_error=model_error,
            ),
            new_memory,
        )

    deterministic = _deterministic_fallback(legal_actions)
    if deterministic is not None:
        return (
            Decision(
                action=deterministic,
                source="fallback_deterministic",
                budget=budget,
                raw_response=raw_response,
                model_error=model_error,
            ),
            new_memory,
        )

    try:
        random_action = _random_fallback(legal_actions)
        return (
            Decision(
                action=random_action,
                source="fallback_random",
                budget=budget,
                raw_response=raw_response,
                model_error=model_error,
            ),
            new_memory,
        )
    except IndexError:
        return (
            Decision(
                action=Action(name=ActionName.RESET),
                source="fallback_random",
                budget=budget,
                raw_response=raw_response,
                model_error=model_error,
            ),
            new_memory,
        )
