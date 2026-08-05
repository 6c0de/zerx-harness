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


from zerx.scene import correspond_objects, find_correspondences


def test_correspond_objects_matches_unique_shapes():
    before = perceive_scene(_frame([[5, 0], [0, 0]]))
    after = perceive_scene(_frame([[0, 5], [0, 0]]))
    mapping = correspond_objects(before, after)
    assert mapping == {before[0].object_id: after[0].object_id}


def test_correspond_objects_disambiguates_duplicate_shapes_by_nearest_centroid():
    before = perceive_scene(_frame([
        [5, 0, 0, 0, 5],
        [0, 0, 0, 0, 0],
    ]))
    after = perceive_scene(_frame([
        [5, 0, 0, 0, 0],
        [0, 0, 0, 0, 5],
    ]))
    left_before = min(before, key=lambda o: o.centroid[0])
    right_before = max(before, key=lambda o: o.centroid[0])
    left_after = min(after, key=lambda o: o.centroid[0])
    right_after = max(after, key=lambda o: o.centroid[0])
    mapping = correspond_objects(before, after)
    # both duplicates share a shape_hash -- must resolve by nearest centroid,
    # not silently pick a fixed index for both.
    assert mapping[left_before.object_id] == left_after.object_id
    assert mapping[right_before.object_id] == right_after.object_id


def test_correspond_objects_none_when_object_disappears():
    before = perceive_scene(_frame([[5, 0], [0, 0]]))
    after = perceive_scene(_frame([[0, 0], [0, 0]]))
    mapping = correspond_objects(before, after)
    assert mapping[before[0].object_id] is None


def test_correspond_objects_falls_back_to_overlap_when_shape_changes_same_color():
    before = perceive_scene(_frame([[5, 0], [0, 0]]))
    after = perceive_scene(_frame([[5, 5], [0, 0]]))
    mapping = correspond_objects(before, after)
    assert mapping[before[0].object_id] == after[0].object_id


def test_correspond_objects_matches_recolor_in_place_via_overlap_fallback():
    # shape_hash includes color (STRATEGY.md SS5.3), so a pure recolor never
    # matches by hash -- this is the fallback path that must still find it.
    before = perceive_scene(_frame([[5, 5], [0, 0]]))
    after = perceive_scene(_frame([[6, 6], [0, 0]]))
    mapping = correspond_objects(before, after)
    assert mapping[before[0].object_id] == after[0].object_id


def test_find_correspondences_is_the_same_function():
    assert find_correspondences is correspond_objects


from zerx.scene import compare_frames, inspect_local_crop, list_salient_objects


def test_list_salient_objects_ranks_small_rare_object_first():
    grid = [
        [1, 1, 1, 2],
        [1, 1, 1, 0],
        [1, 1, 1, 0],
    ]
    scene = perceive_scene(_frame(grid))
    ranked = list_salient_objects(scene)
    assert ranked[0].color == 2


def test_list_salient_objects_empty_scene_returns_empty():
    assert list_salient_objects(()) == ()


def test_compare_frames_reports_no_change_for_identical_scenes():
    scene = perceive_scene(_frame([[5, 0], [0, 0]]))
    assert "no_change" in compare_frames(scene, scene)


def test_compare_frames_reports_appeared_and_disappeared_counts():
    before = perceive_scene(_frame([[5, 0], [0, 0]]))
    after = perceive_scene(_frame([[0, 6], [0, 0]]))
    summary = compare_frames(before, after)
    assert "disappeared=1" in summary
    assert "appeared=1" in summary


def test_inspect_local_crop_returns_requested_region_only():
    grid = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
    ]
    text = inspect_local_crop(_frame(grid), (1, 1, 2, 2))
    assert text == "45\n78"


def test_inspect_local_crop_does_not_return_the_full_grid():
    grid = [[i for i in range(10)] for _ in range(10)]
    text = inspect_local_crop(_frame(grid), (0, 0, 2, 2))
    assert len(text.splitlines()) == 3
    assert all(len(row) == 3 for row in text.splitlines())
