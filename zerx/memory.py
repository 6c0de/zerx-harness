"""Reflection memory: a periodically-refreshed free-text summary of what the
agent has learned this game. Refresh cadence is config-driven; the actual
summarization is injected as a callable so this module has zero model
coupling. Any latency the injected summarizer costs is the caller's
responsibility to measure — this module never touches the action budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

Summarizer = Callable[[str, str], str]  # (previous_summary, recent_context) -> new_summary


@dataclass
class MemoryState:
    summary: str = ""
    step_count: int = 0
    last_refreshed_step: int = 0

    def reset(self) -> None:
        """Clear memory between games — reflection from one game must never
        leak into the next.
        """
        self.summary = ""
        self.step_count = 0
        self.last_refreshed_step = 0


def maybe_refresh(
    state: MemoryState,
    recent_context: str,
    summarizer: Summarizer,
    refresh_interval: int,
) -> MemoryState:
    """Advance the step count by one and, if `refresh_interval` has
    elapsed since the last refresh, produce a new MemoryState with an
    updated summary. Returns a new MemoryState; never mutates `state`.
    """
    new_step_count = state.step_count + 1
    due = (new_step_count - state.last_refreshed_step) >= refresh_interval
    if not due:
        return MemoryState(
            summary=state.summary,
            step_count=new_step_count,
            last_refreshed_step=state.last_refreshed_step,
        )
    new_summary = summarizer(state.summary, recent_context)
    return MemoryState(
        summary=new_summary,
        step_count=new_step_count,
        last_refreshed_step=new_step_count,
    )


@dataclass(frozen=True)
class ConfirmedRule:
    statement: str
    evidence_count: int = 1


@dataclass(frozen=True)
class Hypothesis:
    statement: str
    supporting_evidence: int = 1
    contradicting_evidence: int = 0


@dataclass
class StructuredMemoryState:
    """STRATEGY.md §2.4/§3.1's structured memory schema: distinguishes
    confirmed rules, working hypotheses, and rejected hypotheses instead of
    storing every model statement as one undifferentiated fact, to reduce
    self-reinforcing hallucination in reflection memory. Off by default
    (`Config.structured_memory_on`); `zerx/memory.py`'s existing `MemoryState`
    is untouched and remains the baseline free-text memory.
    """

    confirmed_rules: list = field(default_factory=list)
    working_hypotheses: list = field(default_factory=list)
    rejected_hypotheses: list = field(default_factory=list)
    open_questions: list = field(default_factory=list)
    current_goal: str = ""
    current_plan: list = field(default_factory=list)
    notable_failures: list = field(default_factory=list)
    step_count: int = 0
    last_refreshed_step: int = 0

    def reset(self) -> None:
        """Clear memory between games -- same guarantee as MemoryState.reset(),
        at every field, not just the top-level object.
        """
        self.confirmed_rules = []
        self.working_hypotheses = []
        self.rejected_hypotheses = []
        self.open_questions = []
        self.current_goal = ""
        self.current_plan = []
        self.notable_failures = []
        self.step_count = 0
        self.last_refreshed_step = 0


def record_hypothesis(state: StructuredMemoryState, statement: str) -> StructuredMemoryState:
    """Add a new working hypothesis, or -- if this exact statement is
    already tracked -- increment its supporting evidence instead of
    duplicating it. Never mutates `state`.
    """
    existing = [h for h in state.working_hypotheses if h.statement == statement]
    if existing:
        updated = Hypothesis(
            statement=statement,
            supporting_evidence=existing[0].supporting_evidence + 1,
            contradicting_evidence=existing[0].contradicting_evidence,
        )
        new_working = [updated if h.statement == statement else h for h in state.working_hypotheses]
    else:
        new_working = list(state.working_hypotheses) + [Hypothesis(statement=statement)]
    return replace(state, working_hypotheses=new_working)


def confirm_hypothesis(state: StructuredMemoryState, statement: str) -> StructuredMemoryState:
    """Move a working hypothesis to confirmed_rules (carrying its evidence
    count forward), or -- if it was never tracked as a hypothesis -- confirm
    it directly with evidence_count=1. Deduplicates against an already
    confirmed rule with the same statement by bumping its evidence_count.
    Never mutates `state`.
    """
    matching = [h for h in state.working_hypotheses if h.statement == statement]
    evidence_count = matching[0].supporting_evidence if matching else 1
    new_working = [h for h in state.working_hypotheses if h.statement != statement]

    already_confirmed = [r for r in state.confirmed_rules if r.statement == statement]
    if already_confirmed:
        bumped = ConfirmedRule(statement=statement, evidence_count=already_confirmed[0].evidence_count + evidence_count)
        new_confirmed = [bumped if r.statement == statement else r for r in state.confirmed_rules]
    else:
        new_confirmed = list(state.confirmed_rules) + [ConfirmedRule(statement=statement, evidence_count=evidence_count)]

    return replace(state, working_hypotheses=new_working, confirmed_rules=new_confirmed)


def contradict_hypothesis(state: StructuredMemoryState, statement: str) -> StructuredMemoryState:
    """Increment a working hypothesis's contradicting evidence; the moment
    contradicting_evidence exceeds supporting_evidence, this is a belief
    reversal -- move it from working_hypotheses to rejected_hypotheses
    (STRATEGY.md §7's promotion metric). A statement not currently tracked
    as a working hypothesis is a no-op. Never mutates `state`.
    """
    matching = [h for h in state.working_hypotheses if h.statement == statement]
    if not matching:
        return state

    current = matching[0]
    updated = Hypothesis(
        statement=statement,
        supporting_evidence=current.supporting_evidence,
        contradicting_evidence=current.contradicting_evidence + 1,
    )
    new_working = [h for h in state.working_hypotheses if h.statement != statement]

    if updated.contradicting_evidence >= updated.supporting_evidence:
        new_rejected = list(state.rejected_hypotheses) + [updated]
        return replace(state, working_hypotheses=new_working, rejected_hypotheses=new_rejected)

    new_working.append(updated)
    return replace(state, working_hypotheses=new_working)
