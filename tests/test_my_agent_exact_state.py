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


def _make_agent(monkeypatch, suppression_on: bool, heuristic_first: bool = False) -> MyAgent:
    if heuristic_first:
        # Any candidate counts as confident, so the heuristic path proposes
        # the same ACTION6 click every step for an unchanging frame.
        monkeypatch.setenv("ZERX_HEURISTIC_FIRST", "true")
        monkeypatch.setenv("ZERX_HEURISTIC_CONFIDENCE_THRESHOLD", "0.0")
    else:
        monkeypatch.delenv("ZERX_HEURISTIC_FIRST", raising=False)
        monkeypatch.delenv("ZERX_HEURISTIC_CONFIDENCE_THRESHOLD", raising=False)
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


def _clickable_frame() -> FrameData:
    # One non-background cell -> exactly one ranked click candidate, so the
    # heuristic path proposes the identical ACTION6 coordinate every step.
    return FrameData(
        frame=[[[0, 0], [0, 5]]],
        state=GameState.NOT_FINISHED,
        available_actions=[1, 5, 6],  # -> legal = {ACTION1, ACTION5, ACTION6, RESET}
    )


def test_exact_state_suppression_off_still_rotates_the_deterministic_fallback(monkeypatch):
    """With suppression off, the deterministic fallback rotates through the
    legal preference order keyed on actions_taken (zerx/policy.py). It used
    to return _FALLBACK_PREFERENCE's first legal entry on every single step,
    which pinned a model-less game to one action forever; that is the
    behavior this asserts is gone, without suppression being involved.
    """
    frame = _uniform_frame()
    agent = _make_agent(monkeypatch, suppression_on=False)

    first = agent.choose_action([frame], frame)
    second = agent.choose_action([frame, frame], frame)

    assert first is GameAction.ACTION5  # ordered[0] at actions_taken=0
    assert second is GameAction.ACTION1  # ordered[1] at actions_taken=1
    assert second.reasoning["source"] == "fallback_deterministic"


def test_exact_state_suppression_on_swaps_a_known_noop_for_the_next_legal_action(monkeypatch):
    """Suppression's real target is an action the policy keeps *re-proposing*
    for the same exact state. The deterministic fallback no longer does that
    (it rotates), so this drives the heuristic path instead: a single
    clickable object yields the identical ACTION6 click every step, and the
    frame never changes, so that exact (state, ACTION6:x,y) pair becomes a
    recorded no-op and must be swapped out.
    """
    frame = _clickable_frame()
    agent = _make_agent(monkeypatch, suppression_on=True, heuristic_first=True)

    first = agent.choose_action([frame], frame)
    assert first is GameAction.ACTION6
    assert first.reasoning["source"] == "heuristic"

    second = agent.choose_action([frame, frame], frame)
    assert second is GameAction.ACTION5  # first legal _FALLBACK_PREFERENCE entry
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


def test_exact_state_suppression_never_overrides_the_terminal_reset_shortcut(monkeypatch):
    """C1 regression: decide()'s terminal short-circuit (frame.is_game_over
    -> always RESET, zerx/policy.py) must never be swapped out by the
    suppression-check block, even once (state, RESET) itself becomes a
    recorded no-op. The frame advertises non-RESET available_actions
    ([1, 5]) precisely so the pre-fix bug path -- RESET getting recorded as
    a no-op transition and then swapped for ACTION5 -- is actually
    exercised, not trivially avoided by an empty legal-action set.
    """
    frame = FrameData(
        frame=[[[0, 0], [0, 0]]],
        state=GameState.GAME_OVER,
        available_actions=[1, 5],
    )
    agent = _make_agent(monkeypatch, suppression_on=True)

    for _ in range(4):
        result = agent.choose_action([frame], frame)
        assert result is GameAction.RESET


def test_exact_state_suppression_cascades_past_a_second_suppressed_alternative(monkeypatch):
    """Once the first replacement (ACTION5) is ALSO a recorded no-op for this
    exact state, the swap must keep walking _FALLBACK_PREFERENCE rather than
    re-emitting a known-dead action -- it must reach ACTION1.

    Wiring (verified by running this test, not guessed):
      call 1: no evidence -> heuristic proposes ACTION6 at the object.
      call 2: outcome feedback records (hash, ACTION6:x,y) as a no-op ->
        the heuristic re-proposes the identical click -> swapped for ACTION5.
      call 3: (hash, ACTION5) is now a no-op too -> the heuristic still
        re-proposes the same click -> ACTION5 is the first alternative but is
        suppressed -> must cascade past it to ACTION1.
    """
    frame = _clickable_frame()
    agent = _make_agent(monkeypatch, suppression_on=True, heuristic_first=True)

    first = agent.choose_action([frame], frame)
    assert first is GameAction.ACTION6

    second = agent.choose_action([frame, frame], frame)
    assert second is GameAction.ACTION5
    assert second.reasoning["source"] == "fallback_exact_state_suppressed"

    third = agent.choose_action([frame, frame, frame], frame)
    assert third is GameAction.ACTION1
    assert third.reasoning["source"] == "fallback_exact_state_suppressed"
