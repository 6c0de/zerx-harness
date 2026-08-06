"""ARC-HANDOFF-002: on Kaggle every game runs in its own thread of one
process and they all mutate the same `GameAction` enum singletons, so one
game could submit another game's click coordinates.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from arcengine import FrameData, GameAction, GameState  # noqa: E402

from agent.my_agent import MyAgent  # noqa: E402
from zerx.types import Action, ActionName  # noqa: E402


def _make_agent(game_id: str) -> MyAgent:
    return MyAgent(
        card_id="test-card",
        game_id=game_id,
        agent_name=f"test-agent-{game_id}",
        ROOT_URL="http://example.invalid",
        record=False,
        arc_env=None,
    )


class _RecordingEnv:
    """Stands in for the real environment, recording the coordinates the
    framework actually read off the shared enum at submit time -- which is
    the value that matters, not the one choose_action set earlier.
    """

    def __init__(self) -> None:
        self.submitted = []
        self._lock = threading.Lock()

    def step(self, action, data=None, reasoning=None):
        with self._lock:
            self.submitted.append(
                ((reasoning or {}).get("game"), data.get("x"), data.get("y"))
            )
        return SimpleNamespace(
            game_id="g",
            frame=[np.zeros((2, 2), dtype=int)],
            state=GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=1,
            guid="",
            full_reset=False,
            available_actions=[6],
        )


def test_submitted_coordinates_belong_to_the_agent_that_chose_them():
    """Each thread sets its own distinct coordinates, then hammers
    take_action while the others do the same. Without the submit lock the
    payload read inside do_action_request can be another thread's.
    """
    env = _RecordingEnv()
    results = {}
    errors = []

    def run(agent_index: int) -> None:
        try:
            agent = _make_agent(f"game{agent_index}")
            agent.arc_env = env
            x = y = agent_index
            mine = []
            for _ in range(60):
                agent._pending_submit = Action(name=ActionName.ACTION6, x=x, y=y)
                agent._last_reasoning = {"game": agent_index}
                # Deliberately corrupt the shared singleton first, the way a
                # concurrent game's choose_action would.
                GameAction.ACTION6.set_data({"x": 63, "y": 63})
                agent.take_action(GameAction.ACTION6)
                mine.append((x, y))
            results[agent_index] = mine
        except Exception as exc:  # pragma: no cover - surfaced via `errors`
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(1, 5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(env.submitted) == 4 * 60
    for game_index, x, y in env.submitted:
        assert (x, y) == (game_index, game_index), (
            f"game{game_index} submitted ({x},{y}) -- another thread's payload "
            "leaked through the shared GameAction singleton"
        )


def test_a_crashed_step_does_not_resubmit_the_previous_decisions_payload():
    """`_safe_fallback_action` is used when choose_action raises. The stale
    `_pending_submit` from the last good step must not be re-applied to it.
    """
    env = _RecordingEnv()
    agent = _make_agent("game-crash")
    agent.arc_env = env
    agent._pending_submit = Action(name=ActionName.ACTION6, x=11, y=22)

    def boom(frames, latest_frame):
        raise RuntimeError("simulated internal failure")

    agent._choose_action_inner = boom
    frame = FrameData(
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        available_actions=[6],
    )
    action = agent.choose_action([frame], frame)

    assert agent._pending_submit is None
    assert action.reasoning == {"source": "exception_fallback"}


def test_normal_single_threaded_reasoning_is_still_attached():
    frame = FrameData(
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 5],
    )
    agent = _make_agent("game-solo")
    action = agent.choose_action([frame], frame)
    assert action.reasoning["source"]
    assert "config_hash" in action.reasoning


def test_the_payload_is_reapplied_inside_the_submit_lock_not_merely_before_it():
    """The re-apply alone would pass single-threaded; what makes it correct
    under Swarm's threads is that the write and the framework's read of
    `action.action_data` happen in the same critical section. Assert the
    lock is actually held at the moment the environment is stepped.
    """
    from agent import my_agent as my_agent_module

    held = []

    class _LockObservingEnv(_RecordingEnv):
        def step(self, action, data=None, reasoning=None):
            held.append(my_agent_module._ACTION_SUBMIT_LOCK.locked())
            return super().step(action, data=data, reasoning=reasoning)

    agent = _make_agent("game-lock")
    agent.arc_env = _LockObservingEnv()
    agent._pending_submit = Action(name=ActionName.ACTION6, x=5, y=6)
    agent._last_reasoning = {"game": 1}

    assert my_agent_module._ACTION_SUBMIT_LOCK.locked() is False
    agent.take_action(GameAction.ACTION6)
    assert held == [True]
    assert my_agent_module._ACTION_SUBMIT_LOCK.locked() is False  # released
    assert agent.arc_env.submitted == [(1, 5, 6)]
