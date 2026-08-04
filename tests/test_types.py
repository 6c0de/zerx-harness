import pytest

from zerx.types import Action, ActionName, GameFrame


def test_action6_requires_coordinates():
    with pytest.raises(ValueError):
        Action(name=ActionName.ACTION6)


def test_action6_rejects_out_of_range_coordinates():
    with pytest.raises(ValueError):
        Action(name=ActionName.ACTION6, x=64, y=0)
    with pytest.raises(ValueError):
        Action(name=ActionName.ACTION6, x=0, y=-1)


def test_action6_accepts_boundary_coordinates():
    Action(name=ActionName.ACTION6, x=0, y=0)
    Action(name=ActionName.ACTION6, x=63, y=63)


def test_non_click_action_rejects_coordinates():
    with pytest.raises(ValueError):
        Action(name=ActionName.ACTION1, x=1, y=1)


def test_simple_action_constructs():
    action = Action(name=ActionName.RESET)
    assert action.name == ActionName.RESET
    assert action.x is None and action.y is None


def test_gameframe_is_frozen():
    frame = GameFrame(
        grid=((0, 0), (0, 0)),
        legal_actions=frozenset({ActionName.ACTION1}),
        is_game_over=False,
    )
    with pytest.raises(Exception):
        frame.is_game_over = True
