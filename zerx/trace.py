"""Per-step trace capture: a pure, pygame-free data model and recorders,
consumed either live (scripts/visualize_play.py's --live mode) or from a
saved JSONL file (--replay mode) -- see
docs/superpowers/specs/2026-08-06-baseline-120-followups-design.md.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Protocol, Sequence, Tuple, Union

from zerx.policy import Decision
from zerx.types import GameFrame

_SOURCE_DESCRIPTIONS = {
    "reset": "terminal frame detected; RESET is the only legal action",
    "heuristic": "high-confidence heuristic candidate used; no model call needed",
    "fallback_heuristic": "model call failed or produced no valid action; used the top-ranked click candidate",
    "fallback_deterministic": "no model or heuristic action available; used the static fallback preference order",
    "fallback_random": "no legal action matched any fallback rule; chose randomly among legal actions",
    "fallback_exact_state_suppressed": "the model/heuristic action was a known no-op for this exact state; substituted a legal alternative",
}


def describe_reasoning(decision: Decision) -> str:
    """Human-readable reasoning text for the visualizer's panel: the raw
    model response when one exists, else a synthesized description of why
    the fallback/heuristic path fired -- prefixed with the model backend's
    actual error message when a model call was attempted and raised, since
    otherwise a fallback step gives no visibility into *why* the model
    call failed (auth, network, ...), only that it did.
    """
    if decision.raw_response:
        return decision.raw_response
    fallback_text = _SOURCE_DESCRIPTIONS.get(decision.source, decision.source)
    if decision.model_error:
        return f"model call failed ({decision.model_error}); {fallback_text}"
    return fallback_text


@dataclass(frozen=True)
class TraceMeta:
    game_id: str
    seed: int
    backend: str
    config_hash: str
    started_at: str  # ISO 8601


@dataclass(frozen=True)
class TraceStep:
    step_index: int
    game_id: str
    grid: Tuple[Tuple[int, ...], ...]
    action_name: str
    action_x: Optional[int]
    action_y: Optional[int]
    source: str
    repaired: bool
    target_object_label: Optional[str]
    reasoning: str
    levels_completed: int
    game_state: str


def build_trace_step(
    *,
    step_index: int,
    game_id: str,
    frame: GameFrame,
    decision: Decision,
    levels_completed: int,
    game_state: str,
) -> TraceStep:
    return TraceStep(
        step_index=step_index,
        game_id=game_id,
        grid=frame.grid,
        action_name=decision.action.name.value,
        action_x=decision.action.x,
        action_y=decision.action.y,
        source=decision.source,
        repaired=decision.repaired,
        target_object_label=decision.target_object_label,
        reasoning=describe_reasoning(decision),
        levels_completed=levels_completed,
        game_state=game_state,
    )


class TraceRecorder(Protocol):
    def record(self, step: TraceStep) -> None:
        ...


class JsonlTraceWriter:
    """Appends one JSON line per record to a target file. `path` may name
    an exact file -- refuses to construct if it already exists, since a
    second run appending to it would silently merge two runs into one
    corrupt file with duplicate step_index ordering -- or an existing
    directory (or a path ending in a path separator), in which case the
    actual filename is resolved lazily on the first `write_meta` or
    `record` call, as `<dir>/<game_id>-<timestamp>.jsonl` (design spec's
    default naming), using whichever call arrives first (both TraceMeta
    and TraceStep carry a game_id). `write_meta` is optional and, if
    called, must happen before any `record` call.
    """

    def __init__(self, path: Union[str, Path]) -> None:
        candidate = Path(path)
        self._dir: Optional[Path] = None
        self._path: Optional[Path] = None
        if candidate.is_dir() or str(path).endswith(("/", "\\")):
            self._dir = candidate
        else:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            if candidate.exists():
                raise FileExistsError(
                    f"{candidate}: trace file already exists -- JsonlTraceWriter "
                    "never appends to a pre-existing file (a second run would "
                    "silently merge into the first and corrupt step_index "
                    "ordering); pass a new path, or a directory to get an "
                    "auto-named <game_id>-<timestamp>.jsonl file instead"
                )
            self._path = candidate

    def write_meta(self, meta: TraceMeta) -> None:
        self._resolve_path(meta.game_id)
        self._append({"type": "meta", **asdict(meta)})

    def record(self, step: TraceStep) -> None:
        self._resolve_path(step.game_id)
        self._append({"type": "step", **asdict(step)})

    def _resolve_path(self, game_id: str) -> None:
        if self._path is not None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        candidate = self._dir / f"{game_id}-{timestamp}.jsonl"
        if candidate.exists():
            raise FileExistsError(
                f"{candidate}: auto-named trace file already exists (two runs "
                f"of {game_id!r} started in the same second) -- retry"
            )
        self._path = candidate

    def _append(self, payload: dict) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")


class CompositeTraceRecorder:
    """Fans one `record` call out to every child recorder, in order."""

    def __init__(self, recorders: Sequence[TraceRecorder]) -> None:
        self._recorders: List[TraceRecorder] = list(recorders)

    def record(self, step: TraceStep) -> None:
        for recorder in self._recorders:
            recorder.record(step)
