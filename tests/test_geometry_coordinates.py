"""Phase 13A.1 — coordinate-system abstraction.

Proves coordinates are float-capable and N-dimensional through one geometry home,
so a 3D float substrate (AI2-THOR) needs no per-call-site rewrite. MiniGrid's
integral 2D behavior is preserved (the green-keeping contract).
"""
from __future__ import annotations

import math

from jeenom import geometry
from jeenom.schemas import SceneModel, SceneObject, WorldModelSample


def test_as_coord_preserves_int_and_float():
    # MiniGrid integral coords stay int (display/equality unchanged)...
    assert geometry.as_coord(3) == 3 and isinstance(geometry.as_coord(3), int)
    assert geometry.as_coord(3.0) == 3 and isinstance(geometry.as_coord(3.0), int)
    # ...but a genuine continuous coord is preserved as float.
    assert geometry.as_coord(3.5) == 3.5 and isinstance(geometry.as_coord(3.5), float)


def test_metrics_are_n_dimensional():
    # 2D
    assert geometry.manhattan((0, 0), (3, 4)) == 7
    assert geometry.euclidean((0, 0), (3, 4)) == 5.0
    # 3D — same functions, three components, no special-casing
    assert geometry.manhattan((0, 0, 0), (1, 2, 2)) == 5
    assert geometry.euclidean((0, 0, 0), (1, 2, 2)) == 3.0
    # float coords flow through
    assert math.isclose(geometry.manhattan((0.5, 0.5), (1.0, 1.0)), 1.0)


def test_scene_object_coord_includes_z_only_when_present():
    flat = SceneObject(object_type="door", color="red", x=2, y=3)
    assert flat.coord == (2, 3)
    deep = SceneObject(object_type="door", color="red", x=2.0, y=3.0, z=1.5)
    assert deep.coord == (2.0, 3.0, 1.5)


def test_scene_model_distance_delegates_and_supports_3d():
    obj_2d = SceneObject(object_type="door", color="red", x=3, y=4)
    scene_2d = SceneModel(
        agent_x=0, agent_y=0, agent_dir=0, grid_width=8, grid_height=8,
        objects=[obj_2d], source="task_sense",
    )
    assert scene_2d.manhattan_distance_from_agent(obj_2d) == 7

    obj_3d = SceneObject(object_type="cup", color=None, x=1, y=2, z=2)
    scene_3d = SceneModel(
        agent_x=0, agent_y=0, agent_dir=0, grid_width=0, grid_height=0,
        objects=[obj_3d], source="task_sense", agent_z=0,
    )
    assert scene_3d.manhattan_distance_from_agent(obj_3d) == 5


def test_from_world_model_sample_does_not_truncate_floats():
    sample = WorldModelSample(
        agent_pose={"x": 1.5, "y": 2.25, "dir": 0, "z": 0.5},
        grid_objects=[{"type": "cup", "color": None, "x": 3.5, "y": 4.0, "z": 1.0}],
        grid_size=(0, 0),
        step_count=0,
    )
    scene = SceneModel.from_world_model_sample(sample, source="task_sense")
    assert scene.agent_x == 1.5 and scene.agent_y == 2.25 and scene.agent_z == 0.5
    obj = scene.objects[0]
    assert obj.x == 3.5 and obj.coord == (3.5, 4, 1.0)


def test_minigrid_integral_path_unchanged():
    # The green-keeping contract: integral input renders/compares as int, no "3.0".
    sample = WorldModelSample(
        agent_pose={"x": 1, "y": 1, "dir": 0},
        grid_objects=[{"type": "door", "color": "red", "x": 3, "y": 5}],
        grid_size=(8, 8),
        step_count=0,
    )
    scene = SceneModel.from_world_model_sample(sample, source="task_sense")
    assert isinstance(scene.agent_x, int) and scene.agent_x == 1
    obj = scene.objects[0]
    assert isinstance(obj.x, int) and obj.coord == (3, 5) and obj.z is None
