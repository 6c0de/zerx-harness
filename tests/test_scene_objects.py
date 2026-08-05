from zerx.scene import perceive_scene
from zerx.types import ActionName, GameFrame


def _frame(grid):
    return GameFrame(
        grid=tuple(tuple(row) for row in grid),
        legal_actions=frozenset({ActionName.ACTION1}),
        is_game_over=False,
    )


def _signed_area(loop):
    total = 0.0
    n = len(loop)
    for i in range(n):
        x0, y0 = loop[i]
        x1, y1 = loop[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total / 2


def test_perceive_scene_empty_grid_has_no_objects():
    scene = perceive_scene(_frame([[0, 0], [0, 0]]))
    assert scene == ()


def test_perceive_scene_basic_fields():
    grid = [
        [0, 0, 0],
        [0, 5, 0],
        [0, 0, 0],
    ]
    scene = perceive_scene(_frame(grid))
    assert len(scene) == 1
    obj = scene[0]
    assert obj.object_id == 0
    assert obj.color == 5
    assert obj.area == 1
    assert obj.bbox == (1, 1, 1, 1)
    assert obj.centroid == (1.0, 1.0)
    assert obj.child_ids == ()
    assert obj.adjacent_ids == ()


def test_boundary_is_closed_clockwise_and_covers_all_corners():
    grid = [[3, 3], [3, 3]]
    scene = perceive_scene(_frame(grid))
    obj = scene[0]
    assert set(obj.boundary) == {(0, 0), (2, 0), (2, 2), (0, 2)}
    assert len(obj.boundary) == 4
    assert _signed_area(obj.boundary) > 0  # clockwise in grid-line (y-down) coordinates


def test_thin_strip_object_boundary_is_a_simplified_rectangle():
    grid = [[9, 9, 9, 9]]
    scene = perceive_scene(_frame(grid))
    obj = scene[0]
    assert set(obj.boundary) == {(0, 0), (4, 0), (4, 1), (0, 1)}
    assert obj.area == 4


def test_shape_hash_stable_under_translation():
    left = perceive_scene(_frame([[5, 5, 0], [0, 0, 0]]))[0]
    right = perceive_scene(_frame([[0, 5, 5], [0, 0, 0]]))[0]
    assert left.shape_hash == right.shape_hash
    assert left.bbox != right.bbox


def test_shape_hash_differs_by_color():
    a = perceive_scene(_frame([[5, 0], [0, 0]]))[0]
    b = perceive_scene(_frame([[6, 0], [0, 0]]))[0]
    assert a.shape_hash != b.shape_hash


def test_duplicate_shapes_get_distinct_object_ids_and_same_hash():
    grid = [
        [5, 0, 0, 5],
        [0, 0, 0, 0],
    ]
    scene = perceive_scene(_frame(grid))
    assert len(scene) == 2
    assert scene[0].shape_hash == scene[1].shape_hash
    assert scene[0].object_id != scene[1].object_id


def test_adjacent_different_color_objects_reference_each_other():
    grid = [[2, 4]]
    scene = perceive_scene(_frame(grid))
    a, b = scene
    assert b.object_id in a.adjacent_ids
    assert a.object_id in b.adjacent_ids


def test_diagonal_same_color_objects_are_not_adjacent():
    # 4-connectivity: these are already two separate objects (see
    # test_perception.py's equivalent case), and must not be marked adjacent.
    grid = [
        [3, 0],
        [0, 3],
    ]
    scene = perceive_scene(_frame(grid))
    a, b = scene
    assert b.object_id not in a.adjacent_ids
    assert a.object_id not in b.adjacent_ids


def test_border_component_segments_correctly():
    grid = [
        [7, 7, 0],
        [0, 0, 0],
    ]
    scene = perceive_scene(_frame(grid))
    assert len(scene) == 1
    assert scene[0].bbox == (0, 0, 1, 0)


def test_nested_object_is_a_child_of_its_enclosing_ring():
    grid = [
        [3, 3, 3, 3, 3],
        [3, 0, 0, 0, 3],
        [3, 0, 5, 0, 3],
        [3, 0, 0, 0, 3],
        [3, 3, 3, 3, 3],
    ]
    scene = perceive_scene(_frame(grid))
    by_color = {o.color: o for o in scene}
    ring, center = by_color[3], by_color[5]
    assert center.object_id in ring.child_ids
    assert ring.child_ids == (center.object_id,)
    assert center.child_ids == ()
    # outer boundary must trace the ring's outside, not the hole -- 4 corners, not 8+
    assert set(ring.boundary) == {(0, 0), (5, 0), (5, 5), (0, 5)}


def test_boundary_handles_a_hole_that_touches_the_outer_edge_at_one_vertex():
    # a single background cell fully enclosed by object cells, positioned so
    # the enclosing shape also has a notch touching that same grid vertex --
    # this is the "pinch point" case a naive loop-collection boundary tracer
    # can resolve ambiguously. cells: (1,0)(2,0)(3,0)/(1,1)___(3,1)/__(2,2)(3,2)
    grid = [
        [0, 1, 1, 1],
        [0, 1, 0, 1],
        [0, 0, 1, 1],
    ]
    scene = perceive_scene(_frame(grid))
    assert len(scene) == 1
    obj = scene[0]
    # deterministic across repeated calls -- no dependency on set/dict iteration order
    again = perceive_scene(_frame(grid))[0]
    assert obj.boundary == again.boundary
    # the shared pinch vertex (2, 2) legitimately appears twice: it's where
    # the outer boundary and the enclosed hole's boundary touch at one point
    assert obj.boundary.count((2, 2)) == 2
    assert len(obj.boundary) == 10
