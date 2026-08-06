"""Thin harness adapter — translates the real upstream Frame/GameAction API
(see docs/superpowers/experiments/baseline-000.md for the exact verified
names) into zerx's internal types, delegates to zerx.policy.decide(),
finalizes the previous transition and feeds soft-affordance outcomes back,
and translates the result back. Keep this file free of policy logic; it is
glue only.

Contract (enforced by the vendored `agents.agent.Agent` ABC):
  - Subclass `agents.agent.Agent`.
  - Class must be named `MyAgent` (the notebook's __init__.py registers it).
  - Implement `is_done(frames, latest_frame) -> bool`.
  - Implement `choose_action(frames, latest_frame) -> GameAction`.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from typing import List, Optional, Tuple

# Verified against the vendored arcengine/agents packages
# (docs/superpowers/experiments/baseline-000.md, Task 1, Step 5).
from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent

from zerx.config import Config
from zerx.exact_state_memory import ExactStateMemory, action_signature
from zerx.heuristics import DeadSignatureTracker
from zerx.memory import MemoryState, StructuredMemoryState, maybe_refresh_structured
from zerx.model_backend import select_backend
from zerx.perception import perceive
from zerx.policy import Decision, _FALLBACK_PREFERENCE, decide
from zerx.trace import TraceRecorder, JsonlTraceWriter, TraceMeta, build_trace_step
from zerx.transitions import TransitionLedger, grid_hash
from zerx.types import Action, ActionName, GameFrame

logger = logging.getLogger(__name__)


def _to_game_frame(frame: FrameData) -> GameFrame:
    """Translate the real upstream FrameData into our internal GameFrame.

    - `frame.frame` is a list of animation sub-frames; the current 64x64
      grid is the LAST one (`frame.frame[-1]`), and the list is `[]` when
      nothing has rendered yet (e.g. NOT_PLAYED).
    - `frame.available_actions` lists only the currently-legal NON-RESET
      action ids; RESET is always implicitly legal and is unioned in here
      unconditionally. Ids are mapped via `GameAction.from_id(...)`, NOT
      `GameAction(...)` — the latter raises `ValueError` for every non-RESET
      id. `GameAction`'s custom `__init__` reassigns `self._value_` to the
      plain int action id, but Python's Enum machinery already built
      `_value2member_map_` from the original `(action_id, action_type)`
      tuple before `__init__` ran, so value-based lookup (`GameAction(1)`)
      never matches. `.from_id()` avoids this by linear-scanning `.value`
      instead of using that stale map (verified directly against the
      vendored `arcengine` package, not assumed).
    - `frame.state` is a GameState enum; both NOT_PLAYED and GAME_OVER map
      to `is_game_over=True` so decide()'s terminal short-circuit fires on
      a game's very first call as well as after a death.
    - FrameData has no `.score` field (only level-completion counters, a
      different concept) — default to 0 rather than mis-mapping one.
    """
    sub_frames = frame.frame
    grid_rows = sub_frames[-1] if sub_frames else []
    grid = tuple(tuple(row) for row in grid_rows)
    legal = frozenset(
        {
            ActionName[GameAction.from_id(action_id).name]
            for action_id in frame.available_actions
        }
        | {ActionName.RESET}
    )
    is_game_over = frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER)
    return GameFrame(grid=grid, legal_actions=legal, is_game_over=is_game_over, score=0)


def _to_game_action(action: Action) -> GameAction:
    """`GameAction` members are singletons; `.set_data(...)` mutates the
    shared member's `.action_data` in place and returns it. The real
    starter's own random-baseline agent relies on this exact pattern
    (calling `.set_data` on the enum member itself), so we replicate it
    rather than trying to construct a new, independent GameAction.
    """
    upstream = GameAction[action.name.value]
    if action.name == ActionName.ACTION6:
        upstream.set_data({"x": action.x, "y": action.y})
    return upstream


def _find_object_by_label(frame: GameFrame, label: str):
    """Recompute perception just far enough to look up the LabeledObject a
    past Decision targeted, so its outcome can be recorded. Cheap (no GPU,
    no model call) — perception is already deterministic and pure.
    """
    for obj in perceive(frame).objects:
        if obj.label == label:
            return obj
    return None


def _shapes_match(a: GameFrame, b: GameFrame) -> bool:
    """True iff `a.grid` and `b.grid` have identical dimensions. Guards the
    exact-state outcome-feedback block (baseline-115-exact-state-memory)
    against `zerx.transitions._diff`'s shape assumption — `_diff` only
    iterates over `before`'s dimensions, so a before/after grid-shape
    mismatch (e.g. the very first frame's empty NOT_PLAYED grid transitioning
    to a real grid) can silently misreport "no change" instead of a real
    diff. `zerx/transitions.py` is shared infrastructure and is not modified
    for this fix; the guard lives here instead.
    """
    return len(a.grid) == len(b.grid) and all(
        len(row_a) == len(row_b) for row_a, row_b in zip(a.grid, b.grid)
    )


def _safe_fallback_action(latest_frame: FrameData) -> GameAction:
    """Last-resort action used when the normal `choose_action` path raises.

    Deliberately independent of everything that might have just failed: no
    `decide()`, no `_to_game_frame`, no other zerx module. Works only off
    the raw upstream `latest_frame.available_actions` (a `list[int]`),
    mapped id-by-id via `GameAction.from_id` (skipping any id that itself
    raises), returning the first that succeeds. Falls back to
    `GameAction.RESET` (always constructible) if the list is empty or
    `latest_frame` itself is unusable. Wrapped in its own try/except so
    this helper itself can never raise.

    `GameAction` members are process-wide singletons, so whichever member
    this returns may still carry `.reasoning` set by an earlier, unrelated
    normal decision (see `_choose_action_inner`) — always overwrite it here
    so a crash-recovery action never gets recorded as if it came from the
    normal decide() pipeline.
    """
    try:
        available = getattr(latest_frame, "available_actions", None) or []
        for action_id in available:
            try:
                action = GameAction.from_id(action_id)
                action.reasoning = {"source": "exception_fallback"}
                return action
            except Exception:
                continue
    except Exception:
        pass
    GameAction.RESET.reasoning = {"source": "exception_fallback"}
    return GameAction.RESET


class MyAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._config = Config.from_env()
        self._memory = MemoryState()
        self._dead_signatures = DeadSignatureTracker()
        self._backend = select_backend(self._config)
        self._transitions = TransitionLedger()
        self._exact_state_memory = ExactStateMemory()
        self._actions_taken = 0
        self._pending_decision: Optional[Decision] = None
        self._pending_before_frame: Optional[GameFrame] = None
        # --- baseline-130-hypothesis (feat/baseline-130-hypothesis-memory) ---
        self._structured_memory = StructuredMemoryState()
        # --- end baseline-130-hypothesis ---
        trace_recorder: Optional[TraceRecorder] = None
        if self._config.trace_export_path:
            writer = JsonlTraceWriter(self._config.trace_export_path)
            writer.write_meta(
                TraceMeta(
                    game_id=self.game_id,
                    seed=0,
                    backend=self._config.backend,
                    config_hash=self._config.config_hash(),
                    started_at=datetime.now(timezone.utc).isoformat(),
                )
            )
            trace_recorder = writer
        self.trace_recorder: Optional[TraceRecorder] = trace_recorder

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we want to RESET and
        # retry (matches the real starter's own default behavior).
        return latest_frame.state is GameState.WIN

    def _choose_action_inner(
        self, frames: List[FrameData], latest_frame: FrameData
    ) -> GameAction:
        frame = _to_game_frame(latest_frame)

        # Finalize the PREVIOUS action's transition now that its result
        # (this frame) exists. Never do this before the frame arrives.
        record = self._transitions.finalize(frame)
        if (
            record is not None
            and self._pending_decision is not None
            and self._pending_decision.target_object_label is not None
            and self._pending_before_frame is not None
        ):
            target = _find_object_by_label(
                self._pending_before_frame, self._pending_decision.target_object_label
            )
            if target is not None:
                self._dead_signatures.record_outcome(target, effective=record.effective)

        # --- baseline-115-exact-state-memory (feat/baseline-115-exact-state-memory) ---
        if (
            self._config.exact_state_suppression_on
            and record is not None
            and self._pending_decision is not None
            and self._pending_before_frame is not None
            and _shapes_match(self._pending_before_frame, frame)
        ):
            # level_delta is currently always 0 -- _to_game_frame hardcodes
            # score=0; this channel is inert until a real score/level source
            # is wired (see review notes).
            self._exact_state_memory.record_outcome(
                state_signature=grid_hash(self._pending_before_frame),
                action_signature=action_signature(self._pending_decision.action),
                visible_change=record.changed_pixels > 0,
                level_delta=record.score_delta,
            )
        # --- end baseline-115-exact-state-memory ---

        history: Tuple[GameFrame, ...] = tuple(_to_game_frame(f) for f in frames[-4:])
        decision, self._memory = decide(
            frame=frame,
            history=history,
            memory=self._memory,
            dead_signatures=self._dead_signatures,
            config=self._config,
            backend=self._backend,
            actions_taken=self._actions_taken,
        )

        if decision.model_error is not None:
            # Previously silent: decide() swallows a model-backend failure
            # with no logging anywhere, so a real run could fall back every
            # single step with zero visibility into why (confirmed against
            # real cerebras_dev traces where 100% of calls raised and
            # nothing surfaced it). Logged here, not inside decide(), so
            # decide() stays a pure, side-effect-free function.
            logger.warning(
                "model call failed this step: %s (source=%s)",
                decision.model_error,
                decision.source,
            )

        # --- baseline-115-exact-state-memory (feat/baseline-115-exact-state-memory) ---
        # Guarded by `not frame.is_game_over`: decide()'s terminal
        # short-circuit (zerx/policy.py) always returns RESET on a
        # GAME_OVER/NOT_PLAYED frame, and that must never be swapped out —
        # doing so would permanently break the reset-and-retry loop the
        # instant (state, RESET) itself becomes a recorded no-op.
        if not frame.is_game_over and self._config.exact_state_suppression_on:
            state_sig = grid_hash(frame)
            if self._exact_state_memory.is_suppressed(state_sig, action_signature(decision.action)):
                replacement = None
                for name in _FALLBACK_PREFERENCE:
                    if name not in frame.legal_actions or name == decision.action.name:
                        continue
                    candidate = (
                        Action(name=name, x=32, y=32)
                        if name == ActionName.ACTION6
                        else Action(name=name)
                    )
                    # The candidate itself may ALSO be a known no-op for this
                    # exact state — skip past it too, instead of swapping in
                    # another suppressed action (which would defeat the
                    # feature's whole purpose of escaping a repeated no-op).
                    if not self._exact_state_memory.is_suppressed(
                        state_sig, action_signature(candidate)
                    ):
                        replacement = candidate
                        break
                if replacement is not None:
                    decision = replace(
                        decision,
                        action=replacement,
                        source="fallback_exact_state_suppressed",
                        target_object_label=None,
                    )
                # else: every legal alternative is also a known no-op for
                # this exact state -- leave `decision` unchanged rather than
                # invent an unvalidated move (project fallback philosophy).
        # --- end baseline-115-exact-state-memory ---

        if self.trace_recorder is not None:
            try:
                self.trace_recorder.record(
                    build_trace_step(
                        step_index=self._actions_taken,
                        game_id=self.game_id,
                        frame=frame,
                        decision=decision,
                        levels_completed=latest_frame.levels_completed,
                        game_state=latest_frame.state.name,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - a dev-only observability
                # sink (disk full, permissions, etc.) must never be able to
                # alter real agent behavior or desync the transition ledger
                # / action counter below -- log and continue as if untraced.
                logger.warning(
                    "trace_recorder.record failed: %s: %s; continuing without "
                    "tracing this step",
                    type(exc).__name__,
                    exc,
                )

        self._actions_taken += 1

        # --- baseline-130-hypothesis (feat/baseline-130-hypothesis-memory) ---
        if self._config.structured_memory_on:
            self._structured_memory = maybe_refresh_structured(
                self._structured_memory,
                recent_context=perceive(frame).ascii_grid,
                summarizer=lambda prev, ctx: prev,
                refresh_interval=self._config.memory_refresh_interval,
            )
        # --- end baseline-130-hypothesis ---

        self._transitions.begin(frame, decision.action)
        self._pending_decision = decision
        self._pending_before_frame = frame

        upstream = _to_game_action(decision.action)
        # AGENTS.md step 8: record the decision path, fallback/repair status,
        # and configuration ID so it reaches the framework's recorder. The
        # vendored `Agent.do_action_request` reads `.reasoning` straight off
        # the GameAction enum member (see vendor/ARC-AGI-3-Agents/agents/agent.py).
        upstream.reasoning = {
            "source": decision.source,
            "repaired": decision.repaired,
            "config_hash": self._config.config_hash(),
        }
        return upstream

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Guarantee (plan Global Constraint; AGENTS.md's control-flow and
        testing-gate sections): this must never raise an unhandled
        exception. `_choose_action_inner` already implements a validated
        fallback chain internally (see zerx.policy.decide), but this outer
        boundary catches anything unexpected — e.g. a future game handing
        back an out-of-range action id, or an internal zerx bug — and
        degrades to `_safe_fallback_action` instead of crashing the run.
        """
        try:
            return self._choose_action_inner(frames, latest_frame)
        except Exception as exc:  # noqa: BLE001 - intentional catch-all boundary
            logger.error(
                "choose_action raised %s: %s; falling back to a safe action",
                type(exc).__name__,
                exc,
            )
            return _safe_fallback_action(latest_frame)
