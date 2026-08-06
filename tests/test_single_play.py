"""Tests for `zerx.single_play`, the policy written for the real scoring rules.

The rules these encode were measured through a real competition-mode gateway
(`eval/gateway_smoke.py`), not read off documentation: one play per game,
actions cumulative, each level charged from the previous level's completion,
and `level_score = min(115, (baseline / actions) ** 2 * 100)`.
"""
from __future__ import annotations

import pytest

from zerx.single_play import RESET, SinglePlayAgent, click_candidates


def solid(value: int = 0, size: int = 8):
    return tuple(tuple(value for _ in range(size)) for _ in range(size))


def with_dot(x: int, y: int, colour: int = 3, size: int = 8):
    return tuple(
        tuple(colour if (cx, cy) == (x, y) else 0 for cx in range(size))
        for cy in range(size)
    )


LEGAL = ("ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET")


def test_a_game_over_frame_is_answered_with_reset():
    """The engine rejects everything but RESET on a game over (docs/actions.md);
    anything else would be a wasted action charged to the current level."""
    agent = SinglePlayAgent(seed=1)
    assert agent.step(solid(), LEGAL, 0, game_over=True, won=False) == RESET


def test_a_win_ends_the_game_rather_than_spending_more_actions():
    agent = SinglePlayAgent(seed=1)
    assert agent.step(solid(), LEGAL, 3, game_over=False, won=True) is None


def test_an_action_that_changes_nothing_is_not_offered_again_from_that_board():
    """The core of the careful phase. Repeating a known no-op is the cheapest
    way to lose a level, because every repeat is charged to it."""
    agent = SinglePlayAgent(seed=1, sticky=0.0)
    board = solid()
    first = agent.step(board, LEGAL, 0, False, False)
    # Same board back: the action did nothing.
    second = agent.step(board, LEGAL, 0, False, False)
    assert first != second, "a no-op action was proposed twice from one board"


def test_an_action_that_changed_the_board_is_repeated():
    """Grid worlds need runs of the same move; a policy that re-probes after
    every success pays for the same discovery repeatedly."""
    agent = SinglePlayAgent(seed=1, sticky=1.0)
    first = agent.step(with_dot(1, 1), LEGAL, 0, False, False)
    second = agent.step(with_dot(2, 1), LEGAL, 0, False, False)
    assert second == first


def test_animation_alone_does_not_count_as_progress():
    """HUD tickers and animations change the frame on every step regardless of
    what the agent did. Treating that as 'the action worked' is what made an
    earlier version repeat one move forever, and it is why every level it found
    came from the reckless phase and none from the careful one.
    """
    agent = SinglePlayAgent(seed=1, sticky=1.0, noise_fraction=0.35)
    # One cell flips on every single step; nothing else ever changes.
    for step in range(30):
        grid = with_dot(0, 0, colour=step % 2)
        agent.step(grid, LEGAL, 0, False, False)
    stats = [s for s in agent._stats.values() if s.uses > 4]
    assert stats, "expected some action to have been used repeatedly"
    assert all(s.rate < 0.5 for s in stats), (
        "a cell that changes every step was still counted as a real effect"
    )


def test_the_policy_turns_reckless_once_the_level_is_no_longer_worth_winning():
    """Under RHAE a level completed at ~10x the human baseline is worth about
    1%, so protecting it further buys nothing — while the *next* level's count
    has not started, making those same actions free.
    """
    agent = SinglePlayAgent(seed=1, careful_budget=5)
    assert agent.careful
    for _ in range(6):
        agent.step(solid(), LEGAL, 0, False, False)
    assert not agent.careful


def test_completing_a_level_restores_the_careful_phase():
    agent = SinglePlayAgent(seed=1, careful_budget=5)
    for _ in range(6):
        agent.step(solid(), LEGAL, 0, False, False)
    assert not agent.careful
    agent.step(solid(), LEGAL, 1, False, False)  # a level completed
    assert agent.careful, "the next level starts with a fresh action count"


def test_budget_exhaustion_ends_the_game():
    agent = SinglePlayAgent(seed=1, max_actions=3)
    for _ in range(3):
        assert agent.step(solid(), LEGAL, 0, False, False) is not None
    assert agent.step(solid(), LEGAL, 0, False, False) is None


def test_click_candidates_prefer_the_rare_colour_over_the_background():
    grid = with_dot(5, 6, colour=9, size=16)
    assert click_candidates(grid)[0] == (5, 6)


def test_click_candidates_survive_an_empty_frame():
    """A NOT_PLAYED frame renders no grid at all; the policy must still be able
    to name a coordinate rather than raise inside the agent loop."""
    assert click_candidates(()) == [(32, 32)]


@pytest.mark.parametrize("legal", [("RESET",), ()])
def test_no_legal_action_falls_back_to_reset(legal):
    agent = SinglePlayAgent(seed=1)
    assert agent.step(solid(), legal, 0, False, False) == RESET
