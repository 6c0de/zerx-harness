"""Regression tests from the 2026-08-06 full-repository audit
(docs/audits/2026-08-06-full-repository-audit.md).

Each test below fails against master and passes after the corresponding
fix on feat/policy-prompt-legal-budget.
"""
from __future__ import annotations

from zerx.budget import BudgetSignal
from zerx.memory import MemoryState
from zerx.perception import perceive
from zerx.policy import _MAX_PROMPT_OBJECTS, build_prompt
from zerx.transitions import TransitionLedger, _diff
from zerx.types import Action, ActionName, GameFrame

LEGAL = frozenset({ActionName.RESET, ActionName.ACTION1, ActionName.ACTION6})


def _frame(grid, is_game_over=False, score=0):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=LEGAL,
        is_game_over=is_game_over,
        score=score,
    )


# --- ARC-AUDIT-004: _diff crashed / lied on a grid-shape change ------------


def test_diff_does_not_raise_when_the_after_grid_is_empty():
    """A GAME_OVER / NOT_PLAYED FrameData carries `frame == []`, so the
    adapter produces an empty grid. Before the fix this raised IndexError
    out of TransitionLedger.finalize(), which my_agent's crash boundary then
    swallowed — silently discarding that entire decision step.
    """
    before = _frame([[1, 1], [1, 1]])
    after = _frame([], is_game_over=True)

    changed, bbox = _diff(before, after)

    assert changed > 0, "a board vanishing is a change, not a no-op"
    assert bbox is None


def test_diff_reports_a_change_when_the_first_real_grid_appears():
    """empty -> 64x64 previously reported (0, None) ("nothing happened")
    because the loop bounds came from `before` only, so the very first real
    frame of every game looked like a dead action.
    """
    before = _frame([])
    after = _frame([[1, 2], [3, 4]])

    changed, _ = _diff(before, after)

    assert changed > 0


def test_ledger_marks_a_shape_change_as_an_effective_transition():
    ledger = TransitionLedger()
    ledger.begin(_frame([]), Action(name=ActionName.RESET))
    record = ledger.finalize(_frame([[5, 5], [5, 5]]))

    assert record is not None
    assert record.effective is True


# --- ARC-AUDIT-005: levels_completed was discarded at the adapter ----------


def test_score_delta_reports_level_completion():
    """`GameFrame.score` now carries FrameData.levels_completed, so the
    transition that completes a level has score_delta == 1 and is
    `effective` even if the pixels happen not to change.
    """
    ledger = TransitionLedger()
    identical = [[7, 7], [7, 7]]
    ledger.begin(_frame(identical, score=0), Action(name=ActionName.ACTION1))
    record = ledger.finalize(_frame(identical, score=1))

    assert record is not None
    assert record.score_delta == 1
    assert record.effective is True, "completing a level must count as effective"


def test_prompt_object_table_is_bounded_on_a_pathological_frame():
    """A two-colour 64x64 checkerboard segments into 4096 single-cell
    objects — a legal frame that rendered ~49k tokens of object table and
    would overflow the context window.
    """
    grid = tuple(tuple(1 if (x + y) % 2 == 0 else 2 for x in range(64)) for y in range(64))
    perception = perceive(_frame(grid))
    assert len(perception.objects) == 4096  # the input really is pathological

    prompt = build_prompt(perception, MemoryState())

    listed = sum(1 for line in prompt.splitlines() if line.startswith("- obj"))
    assert listed <= _MAX_PROMPT_OBJECTS
    assert "more objects not listed" in prompt, "truncation must be disclosed"
    assert len(prompt) < 40_000, f"prompt still too large: {len(prompt)} chars"


def test_prompt_lists_every_object_when_under_the_cap():
    grid = [[1, 0], [0, 2]]
    perception = perceive(_frame(grid))
    prompt = build_prompt(perception, MemoryState())

    assert "more objects not listed" not in prompt
    listed = sum(1 for line in prompt.splitlines() if line.startswith("- obj"))
    assert listed == len(perception.objects)


# --- ARC-AUDIT-007: fake backend off `local` must be loud -----------------


def test_fake_backend_on_kaggle_platform_logs_an_error(caplog):
    """A missing ZERX_BACKEND on Kaggle silently yields a model-free run
    (every generate() raises, agent degrades to heuristics). That must be
    visible in the logs rather than discovered from the leaderboard.
    """
    import logging

    from zerx.config import Config
    from zerx.model_backend import select_backend

    with caplog.at_level(logging.ERROR):
        select_backend(Config(backend="fake", platform="kaggle"))

    assert any("heuristics-only" in record.message for record in caplog.records)


def test_fake_backend_on_local_platform_is_silent(caplog):
    import logging

    from zerx.config import Config
    from zerx.model_backend import select_backend

    with caplog.at_level(logging.ERROR):
        select_backend(Config(backend="fake", platform="local"))

    assert not caplog.records


# --- prompt still carries the legal-action / budget signals ---------------


def test_prompt_still_carries_legal_actions_and_budget_after_the_object_cap():
    perception = perceive(_frame([[1, 0], [0, 2]]))
    budget = BudgetSignal(actions_taken=9, soft_cap=20, should_favor_execution=False)

    prompt = build_prompt(
        perception, MemoryState(), legal_actions=LEGAL, budget=budget
    )

    assert "ACTION6" in prompt
    assert "9" in prompt and "20" in prompt
