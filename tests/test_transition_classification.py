from zerx.scene import classify_transition, correspond_objects, perceive_scene
from zerx.types import ActionName, GameFrame


def _frame(grid):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=frozenset({ActionName.ACTION1}),
        is_game_over=False,
    )


def _classify(before_grid, after_grid, terminal=False, level_delta=0, width=5, height=5):
    before = perceive_scene(_frame(before_grid))
    after = perceive_scene(_frame(after_grid))
    correspondence = correspond_objects(before, after)
    return classify_transition(before, after, correspondence, terminal, level_delta, width, height)


def test_true_no_op_is_no_change():
    grid = [[5, 0], [0, 0]]
    assert _classify(grid, grid, width=2, height=2) == "NO_CHANGE"


def test_small_edge_object_change_is_hud_only():
    before = [[0, 0], [0, 0]]
    after = [[0, 0], [1, 0]]
    assert _classify(before, after, width=2, height=2) == "HUD_ONLY"


def test_object_moving_same_shape_and_color_is_object_move():
    before = [
        [0, 0, 0, 0, 0],
        [0, 5, 5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    after = [
        [0, 0, 0, 0, 0],
        [0, 0, 5, 5, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert _classify(before, after) == "OBJECT_MOVE"


def test_recoloring_in_place_is_recolor_or_transform():
    before = [
        [0, 0, 0, 0, 0],
        [0, 5, 5, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    after = [
        [0, 0, 0, 0, 0],
        [0, 6, 6, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    assert _classify(before, after) == "RECOLOR_OR_TRANSFORM"


def test_large_object_appearing_is_object_appear_disappear():
    before = [[0] * 5 for _ in range(5)]
    after = [
        [0, 0, 0, 0, 0],
        [0, 7, 7, 7, 0],
        [0, 7, 7, 7, 0],
        [0, 7, 7, 7, 0],
        [0, 0, 0, 0, 0],
    ]
    assert _classify(before, after) == "OBJECT_APPEAR_DISAPPEAR"


def test_animation_frame_noise_at_edge_is_hud_only_not_confident_progress():
    # a 1-cell "timer" flicker at the frame edge must not read as real progress
    before = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    after = [[0, 0, 2], [0, 0, 0], [0, 0, 0]]
    assert _classify(before, after, width=3, height=3) == "HUD_ONLY"


def test_reset_style_full_frame_replacement_is_object_appear_disappear():
    before = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]
    after = [
        [8, 8, 8],
        [8, 0, 8],
        [8, 8, 8],
    ]
    assert _classify(before, after, width=3, height=3) == "OBJECT_APPEAR_DISAPPEAR"


def test_level_completion_is_level_boundary_regardless_of_diff():
    before = [[0, 0], [0, 0]]
    after = [[0, 0], [0, 0]]
    assert _classify(before, after, level_delta=1, width=2, height=2) == "LEVEL_BOUNDARY"


def test_game_over_is_terminal_regardless_of_diff():
    before = [[0, 0], [0, 0]]
    after = [[0, 0], [0, 0]]
    assert _classify(before, after, terminal=True, width=2, height=2) == "TERMINAL"


def test_low_confidence_fallback_match_with_color_change_is_appear_disappear_not_recolor():
    # a matched pair with a real color change but weak positional overlap
    # (the fallback tier's least-bad guess, not a confident correspondence)
    # must not be trusted as "the same object, just recolored."
    before = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]
    after = [
        [8, 8, 8],
        [8, 0, 8],
        [8, 8, 8],
    ]
    assert _classify(before, after, width=3, height=3) == "OBJECT_APPEAR_DISAPPEAR"
