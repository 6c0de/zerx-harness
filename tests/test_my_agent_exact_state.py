"""Tests for baseline-115-exact-state-memory's wiring into
agent/my_agent.py: feeding every action's outcome into ExactStateMemory,
and swapping out a decision whose exact (state, action) pair is already
known to be a no-op. Mirrors tests/test_my_agent.py's setup (real vendored
ARC-AGI-3-Agents framework, arc_env=None, no live game environment
needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ARC-AGI-3-Agents"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from arcengine import FrameData, GameAction, GameState  # noqa: E402

from agent.my_agent import MyAgent  # noqa: E402


def _make_agent(monkeypatch, suppression_on: bool) -> MyAgent:
    if suppression_on:
        monkeypatch.setenv("ZERX_EXACT_STATE_SUPPRESSION_ON", "true")
    else:
        monkeypatch.delenv("ZERX_EXACT_STATE_SUPPRESSION_ON", raising=False)
    return MyAgent(
        card_id="test-card",
        game_id="test-game",
        agent_name="test-agent",
        ROOT_URL="http://example.invalid",
        record=False,
        arc_env=None,
    )


def _uniform_frame() -> FrameData:
    # All-zero grid -> perception finds zero non-background objects, so
    # decide() never proposes an ACTION6 click and no model backend call
    # can succeed (no server listening) -> it deterministically falls
    # through to zerx.policy._deterministic_fallback, which is the code
    # path this test exercises.
    return FrameData(
        frame=[[[0, 0], [0, 0]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 5],  # -> legal = {ACTION1, ACTION5, RESET}
    )


def test_exact_state_suppression_off_by_default_repeats_the_same_fallback_action(monkeypatch):
    frame = _uniform_frame()
    agent = _make_agent(monkeypatch, suppression_on=False)

    first = agent.choose_action([frame], frame)
    second = agent.choose_action([frame, frame], frame)

    # ACTION5 is _FALLBACK_PREFERENCE's first legal entry both times --
    # suppression is off, so nothing changes that.
    assert first is GameAction.ACTION5
    assert second is GameAction.ACTION5


def test_exact_state_suppression_on_swaps_a_known_noop_for_the_next_legal_action(monkeypatch):
    frame = _uniform_frame()
    agent = _make_agent(monkeypatch, suppression_on=True)

    first = agent.choose_action([frame], frame)
    assert first is GameAction.ACTION5

    # Same frame again: zero visible change, zero score delta for the
    # pending ACTION5 -> recorded as a known no-op for this exact
    # (state, action) pair BEFORE decide() runs again this same call ->
    # decide() would deterministically re-propose ACTION5, but the
    # post-check swaps it for the next legal preference (ACTION1) instead.
    second = agent.choose_action([frame, frame], frame)
    assert second is GameAction.ACTION1
    assert second.reasoning["source"] == "fallback_exact_state_suppressed"


def test_exact_state_suppression_does_not_affect_the_first_ever_action(monkeypatch):
    """With no prior evidence, ExactStateMemory is empty -- the first call
    on any frame must behave identically whether the flag is on or off.
    """
    frame = _uniform_frame()
    agent_off = _make_agent(monkeypatch, suppression_on=False)
    result_off = agent_off.choose_action([frame], frame)

    frame2 = _uniform_frame()
    agent_on = _make_agent(monkeypatch, suppression_on=True)
    result_on = agent_on.choose_action([frame2], frame2)

    assert result_off is result_on is GameAction.ACTION5
