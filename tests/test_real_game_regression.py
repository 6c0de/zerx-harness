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

from agent.my_agent import MyAgent, _to_game_frame  # noqa: E402
from zerx.config import Config  # noqa: E402
from zerx.heuristics import DeadSignatureTracker  # noqa: E402
from zerx.memory import MemoryState  # noqa: E402
from zerx.model_backend import FakeModelBackend  # noqa: E402
from zerx.perception import perceive  # noqa: E402
from zerx.policy import decide  # noqa: E402
from zerx.types import ActionName  # noqa: E402

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


def _live_frame(arcade, game_id: str):
    """One real, post-reset GameFrame from the live engine, translated the
    same way agent/my_agent.py does. A freshly-constructed agent's
    frames[-1] is the framework's un-reset placeholder (empty grid,
    is_game_over=True, legal_actions={RESET} only) -- one real step is
    needed so the engine returns an actual playable frame with real grid
    content. decide() is pure/local, so a single fetched frame is enough
    to characterize its behavior deterministically after that -- no need
    to step the live engine repeatedly for these tests.
    """
    env = arcade.make(game_id)
    assert env is not None, f"arcade.make({game_id!r}) returned None"
    agent = MyAgent(
        card_id="characterization",
        game_id=game_id,
        agent_name=f"characterization.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
    )
    agent.MAX_ACTIONS = 1
    agent.main()
    frame = _to_game_frame(agent.frames[-1])
    assert not frame.is_game_over, (
        f"{game_id}: still is_game_over=True after one real step -- "
        "the game may need more than one RESET to start playable"
    )
    return frame


def test_ls20_fallback_loop_is_fully_explained_by_missing_backend(arcade):
    """Mechanism A (plan doc): ls20 has no ACTION6 in its legal-action set,
    so decide() never reaches the candidate/heuristic system at all and
    always falls to the same static _deterministic_fallback choice. This
    locks in that today's (Track-1-fix-pending) behavior is exactly
    ACTION1, every call, regardless of zerx/heuristics.py.
    """
    frame = _live_frame(arcade, "ls20")
    assert ActionName.ACTION6 not in frame.legal_actions
    backend = FakeModelBackend(responses=[])  # every .generate() raises
    memory = MemoryState()
    dead_signatures = DeadSignatureTracker()
    actions = set()
    sources = set()
    for _ in range(20):
        decision, memory = decide(
            frame=frame, history=(), memory=memory,
            dead_signatures=dead_signatures, config=Config(),
            backend=backend, actions_taken=0,
        )
        actions.add(decision.action.name)
        sources.add(decision.source)
    assert actions == {ActionName.ACTION1}
    assert sources == {"fallback_deterministic"}


def test_vc33_fallback_loop_never_diversifies_when_transitions_report_effective(arcade):
    """Mechanism B (plan doc): vc33 has ACTION6 legal and multiple ranked
    click candidates, so the candidate/heuristic path IS reachable -- but
    zerx/heuristics.py's DeadSignatureTracker never down-ranks the
    repeatedly-chosen candidate here, because this test simulates exactly
    what the real agent/my_agent.py wiring does when zerx/transitions.py's
    whole-grid diff reports effective=True every step (STRATEGY.md 5.4's
    documented HUD-vs-gameplay-change limitation, confirmed live this
    session on vc33's animated top-row bar). This is not a bug in
    zerx/heuristics.py -- it faithfully honors the effective value it's
    given; this test locks in that faithful (but, on this game, misled)
    behavior so a future exp-150 fix has a measurable "before".
    """
    frame = _live_frame(arcade, "vc33")
    assert ActionName.ACTION6 in frame.legal_actions
    backend = FakeModelBackend(responses=[])
    memory = MemoryState()
    dead_signatures = DeadSignatureTracker()
    coordinates = set()
    sources = set()
    for _ in range(20):
        decision, memory = decide(
            frame=frame, history=(), memory=memory,
            dead_signatures=dead_signatures, config=Config(),
            backend=backend, actions_taken=0,
        )
        assert decision.action.name == ActionName.ACTION6
        coordinates.add((decision.action.x, decision.action.y))
        sources.add(decision.source)
        # Matches agent/my_agent.py's real outcome-feedback wiring, driven
        # by the confirmed-live finding: zerx/transitions.py reports
        # effective=True every step on vc33 regardless of the click target.
        assert decision.target_object_label is not None
        target = next(
            obj for obj in perceive(frame).objects
            if obj.label == decision.target_object_label
        )
        dead_signatures.record_outcome(target, effective=True)
    assert len(coordinates) == 1, (
        "expected zero coordinate variation -- if this now fails, "
        "DeadSignatureTracker's penalty mechanism started diversifying "
        "the choice, which would mean exp-150's fix (or an equivalent) "
        "landed and this characterization test should be revisited"
    )
    assert sources <= {"heuristic", "fallback_heuristic"}
