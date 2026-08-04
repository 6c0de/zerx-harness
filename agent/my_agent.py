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

from typing import List, Optional, Tuple

# Verified against the vendored arcengine/agents packages
# (docs/superpowers/experiments/baseline-000.md, Task 1, Step 5).
from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent

from zerx.config import Config
from zerx.heuristics import DeadSignatureTracker
from zerx.memory import MemoryState
from zerx.model_backend import GemmaModelBackend
from zerx.perception import perceive
from zerx.policy import Decision, decide
from zerx.transitions import TransitionLedger
from zerx.types import Action, ActionName, GameFrame


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


class MyAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._config = Config.from_env()
        self._memory = MemoryState()
        self._dead_signatures = DeadSignatureTracker()
        self._backend = GemmaModelBackend(self._config.model_revision)
        self._transitions = TransitionLedger()
        self._actions_taken = 0
        self._pending_decision: Optional[Decision] = None
        self._pending_before_frame: Optional[GameFrame] = None

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        # Stop once we win. Don't stop on GAME_OVER — we want to RESET and
        # retry (matches the real starter's own default behavior).
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
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
        self._actions_taken += 1

        self._transitions.begin(frame, decision.action)
        self._pending_decision = decision
        self._pending_before_frame = frame

        return _to_game_action(decision.action)
