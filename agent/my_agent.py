"""Thin harness adapter: translates the framework's Frame/GameAction API into
`zerx.single_play.SinglePlayAgent` calls and back.

Keep this file free of policy. All strategy lives in `zerx/single_play.py`,
which is unit-testable without the framework and drivable from
`eval/local_rhae.py` against the competition's own scorer — through a real
competition-mode gateway, which is the only topology whose numbers transfer
(see `docs/HANDOFF.md`, 2026-08-07).

Contract (enforced by the vendored `agents.agent.Agent` ABC):
  - subclass `agents.agent.Agent`
  - the class must be named `MyAgent` (the notebook's `__init__.py` registers it)
  - implement `is_done(frames, latest_frame) -> bool`
  - implement `choose_action(frames, latest_frame) -> GameAction`

Two framework details this adapter has to work around, both verified against
the vendored source rather than assumed:

* `Agent.MAX_ACTIONS` defaults to 80 and `Agent.main()` reads it off the
  instance. Left alone it caps every game at 81 actions — far below what the
  strategist needs, and invisible in the logs as anything but a low score.

* `GameAction` members are process-wide singletons and `.set_data(...)` mutates
  the shared member in place, while the framework reads `action.action_data`
  later, inside `do_action_request`. `Swarm` runs one thread per game, so
  between our write and that read another game's thread can overwrite the
  coordinates — one game submitting another game's click. `take_action`
  re-applies this agent's own payload under a process-wide lock, in the same
  critical section as the read.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent

from zerx.single_play import ActionKey, SinglePlayAgent, as_grid

logger = logging.getLogger(__name__)

_ACTION_SUBMIT_LOCK = threading.Lock()


def _policy_from_env(game_id: str = "") -> SinglePlayAgent:
    """Build the policy from ZERX_* environment variables.

    Every knob is an env var so a Kaggle run can be retuned without editing
    bundled source, and so the configuration is recorded by the run rather than
    implied by whichever code happened to ship.
    """
    def _f(name: str, default: float) -> float:
        try:
            return float(os.environ[name])
        except (KeyError, ValueError):
            return default

    return SinglePlayAgent(
        max_seconds=_f("ZERX_GAME_SECONDS", 600.0),
        max_actions=int(_f("ZERX_MAX_ACTIONS", 20_000)),
        sticky=_f("ZERX_STICKY", 0.7),
        careful_budget=int(_f("ZERX_CAREFUL_BUDGET", 220)),
        noise_fraction=_f("ZERX_NOISE_FRACTION", 0.35),
        # Per game, so two games never draw the same sequence. Seeded rather
        # than left to the clock so a run is reproducible.
        seed=int(_f("ZERX_SEED", 0)) + (abs(hash(game_id)) % 997 if game_id else 0),
    )


def _config_signature() -> str:
    """A short digest of the ZERX_* settings actually in force.

    Attached to every action's reasoning so a recorded run says which
    configuration produced it, instead of leaving it to be guessed from the
    code that happened to be deployed.
    """
    import hashlib

    relevant = sorted(
        f"{k}={v}" for k, v in os.environ.items() if k.startswith("ZERX_")
    )
    return hashlib.sha256("|".join(relevant).encode()).hexdigest()[:12]


class MyAgent(Agent):
    MAX_ACTIONS = 1_000_000  # the strategist owns its own budget; see below

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.MAX_ACTIONS = int(os.getenv("ZERX_HARD_ACTION_CAP", "1000000"))
        self._strategist = _policy_from_env(str(self.game_id))
        self._config_hash = _config_signature()
        self._pending: Optional[ActionKey] = None
        self._finished = False

    # ---- framework contract -------------------------------------------

    def is_done(self, frames: List[FrameData], latest_frame: FrameData) -> bool:
        """Stop only when the policy says so.

        Deliberately not stopping on GAME_OVER: a game over is recoverable with
        a RESET, which the gateway serves as a level reset, so the play — and
        every level it has already banked — survives it. Stopping there would
        throw away the rest of the game for nothing.
        """
        return self._finished

    def choose_action(self, frames: List[FrameData], latest_frame: FrameData) -> GameAction:
        """Never raises. A crash here would end a game with whatever score it
        had; degrading to RESET keeps the run alive instead.
        """
        try:
            action = self._choose(latest_frame)
        except Exception as exc:  # noqa: BLE001 - intentional outer boundary
            logger.error(
                "%s: choose_action raised %s: %s; falling back to RESET",
                self.game_id, type(exc).__name__, exc,
            )
            self._pending = None
            return GameAction.RESET

        if action is None:
            self._finished = True
            self._pending = None
            return GameAction.RESET

        name, x, y = action
        self._pending = action
        upstream = GameAction[name]
        if name == "ACTION6":
            upstream.set_data({"x": x, "y": y})
        upstream.reasoning = {
            "source": "single_play",
            "phase": "careful" if self._strategist.careful else "reckless",
            "actions": self._strategist.actions_used,
            "levels": self._strategist.levels,
            "config_hash": self._config_hash,
        }
        return upstream

    def take_action(self, action: GameAction) -> Optional[FrameData]:
        with _ACTION_SUBMIT_LOCK:
            pending = self._pending
            if pending is not None and pending[0] == "ACTION6":
                action.set_data({"x": pending[1], "y": pending[2]})
            return super().take_action(action)

    # ---- translation ---------------------------------------------------

    def _choose(self, latest_frame: FrameData) -> Optional[ActionKey]:
        sub_frames = latest_frame.frame
        grid = as_grid(sub_frames[-1]) if len(sub_frames) else ()

        # `available_actions` lists only currently-legal NON-RESET action ids;
        # RESET is always implicitly legal. Ids map via `GameAction.from_id`,
        # never `GameAction(id)`: the enum's custom __init__ rewrites _value_
        # after Python already built _value2member_map_ from the original
        # tuple, so value lookup never matches for anything but RESET.
        legal = []
        for action_id in latest_frame.available_actions or []:
            try:
                legal.append(GameAction.from_id(action_id).name)
            except Exception:  # noqa: BLE001 - an unknown id is not fatal
                continue
        legal.append("RESET")

        state = latest_frame.state
        return self._strategist.step(
            grid=grid,
            legal=legal,
            levels_completed=latest_frame.levels_completed,
            game_over=state in (GameState.GAME_OVER, GameState.NOT_PLAYED),
            won=state is GameState.WIN,
        )
