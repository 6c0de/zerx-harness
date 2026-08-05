"""Real-game regression tests against the live local arc_agi engine -- no
GPU, no model backend required. Drives agent/my_agent.py's real MyAgent
exactly as it exists on this branch's base commit (including the known,
not-yet-fixed hardcoded-GemmaModelBackend construction -- see
docs/superpowers/plans/2026-08-05-baseline-120-local-regression.md's
"Investigation findings" for the full root-cause writeup this file's
tests encode).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

import arc_agi  # noqa: E402
from arc_agi import OperationMode  # noqa: E402

from agent.my_agent import MyAgent  # noqa: E402

ALL_PUBLIC_GAME_IDS = [
    "su15", "sb26", "ft09", "cd82", "sk48", "tr87", "sc25", "ls20", "g50t",
    "bp35", "lf52", "m0r0", "vc33", "tn36", "r11l", "dc22", "sp80", "ka59",
    "cn04", "s5i5", "re86", "ar25", "tu93", "lp85", "wa30",
]

SWEEP_STEP_CAP = 5  # see plan doc: measured 8.15s/action on this platform


@pytest.fixture(scope="module")
def arcade():
    return arc_agi.Arcade(operation_mode=OperationMode.NORMAL)


@pytest.mark.slow_local_engine
@pytest.mark.parametrize("game_id", ALL_PUBLIC_GAME_IDS)
def test_crash_safety_sweep(arcade, game_id):
    """No unhandled exception escapes MyAgent.choose_action for any public
    game, and the run always reaches a terminal GameState or the step cap
    -- baseline-100/baseline-110's "no regressions" promotion criterion,
    verified against the FULL 25-game public set for the first time (prior
    sessions only ever exercised ls20+vc33).
    """
    env = arcade.make(game_id)
    if env is None:
        pytest.skip(f"arcade.make({game_id!r}) returned None -- game unavailable")

    agent = MyAgent(
        card_id="regression-sweep",
        game_id=game_id,
        agent_name=f"regression-sweep.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
    )
    agent.MAX_ACTIONS = SWEEP_STEP_CAP

    try:
        agent.main()
    except Exception as exc:  # noqa: BLE001 - the exact thing this test checks for
        pytest.fail(
            f"{game_id}: unhandled exception escaped MyAgent.main() after "
            f"{agent.action_counter} action(s): {type(exc).__name__}: {exc}"
        )

    # vendor/ARC-AGI-3-Agents/agents/agent.py's main() loop condition is
    # `action_counter <= MAX_ACTIONS` checked BEFORE incrementing, so the
    # loop body runs once more than MAX_ACTIONS before exiting -- verified
    # directly against the vendored framework, not assumed.
    assert agent.action_counter <= SWEEP_STEP_CAP + 1
    assert agent.frames[-1].state is not None
