"""Reflection memory: a periodically-refreshed free-text summary of what the
agent has learned this game. Refresh cadence is config-driven; the actual
summarization is injected as a callable so this module has zero model
coupling. Any latency the injected summarizer costs is the caller's
responsibility to measure — this module never touches the action budget.
"""
from __future__ import annotations

from dataclasses import dataclass
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
