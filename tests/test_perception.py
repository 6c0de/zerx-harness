from zerx.perception import perceive
from zerx.types import ActionName, GameFrame


def _frame(grid):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=frozenset({ActionName.ACTION1}),
        is_game_over=False,
    )


def test_perceive_empty_grid_has_no_objects():
    result = perceive(_frame([[0, 0], [0, 0]]))
    assert result.objects == ()


def test_perceive_single_cell_object():
    result = perceive(_frame([[0, 0], [0, 5]]))
    assert len(result.objects) == 1
    obj = result.objects[0]
    assert obj.color == 5
    assert obj.size == 1
    assert obj.bbox == (1, 1, 1, 1)


def test_perceive_groups_contiguous_same_color():
    grid = [
        [0, 3, 3],
        [0, 3, 0],
        [0, 0, 0],
    ]
    result = perceive(_frame(grid))
    assert len(result.objects) == 1
    assert result.objects[0].size == 3


def test_perceive_separates_touching_different_colors():
    grid = [[2, 4]]
    result = perceive(_frame(grid))
    colors = sorted(obj.color for obj in result.objects)
    assert colors == [2, 4]


def test_perceive_separates_diagonal_same_color_as_two_objects():
    # 4-connectivity only: diagonal touches don't merge.
    grid = [
        [3, 0],
        [0, 3],
    ]
    result = perceive(_frame(grid))
    assert len(result.objects) == 2


def test_ascii_grid_matches_dimensions_and_hex_encodes_colors():
    grid = [[0, 10], [1, 2]]
    result = perceive(_frame(grid))
    rows = result.ascii_grid.split("\n")
    assert len(rows) == 2
    assert all(len(row) == 2 for row in rows)
    assert rows[0] == "0a"
