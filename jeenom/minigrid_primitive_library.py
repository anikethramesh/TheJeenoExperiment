from __future__ import annotations

from .primitive_library import PrimitiveSpec


MINIGRID_GROUNDING_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "visible_doors": PrimitiveSpec(
        name="visible_doors",
        consumes=("scene.grid_objects",),
        produces=("door_candidates",),
        description="List currently visible/available door objects in the MiniGrid scene.",
        runtime_kind="python",
        runtime_value="scene_doors",
    ),
    "closest_door.manhattan.agent": PrimitiveSpec(
        name="closest_door.manhattan.agent",
        consumes=("scene.door_candidates", "agent_pose"),
        produces=("grounded_target", "distance"),
        description="Rank door candidates by Manhattan distance from the agent.",
        runtime_kind="python",
        runtime_value="ground_closest_door_manhattan",
    ),
    "closest_door.euclidean.agent": PrimitiveSpec(
        name="closest_door.euclidean.agent",
        consumes=("scene.door_candidates", "agent_pose"),
        produces=("grounded_target", "distance"),
        description="Rank door candidates by Euclidean distance from the agent.",
        implementation_status="synthesizable",
        safe_to_synthesize=True,
        runtime_kind=None,
        runtime_value=None,
    ),
    "unique_door.color_filter": PrimitiveSpec(
        name="unique_door.color_filter",
        consumes=("scene.door_candidates", "selector.color", "selector.exclude_color"),
        produces=("grounded_target", "distance"),
        description="Resolve a unique door by include/exclude color constraints.",
        runtime_kind="python",
        runtime_value="ground_unique_door_color_filter",
    ),
    "all_doors.ranked.manhattan.agent": PrimitiveSpec(
        name="all_doors.ranked.manhattan.agent",
        consumes=("scene.door_candidates", "agent_pose"),
        produces=("ranked_door_list", "distances"),
        description=(
            "List all visible doors ranked by Manhattan distance from the agent. "
            "Query only — no target is selected and no motion occurs."
        ),
        runtime_kind="python",
        runtime_value="ground_all_doors_ranked_manhattan",
    ),
    "all_doors.ranked.euclidean.agent": PrimitiveSpec(
        name="all_doors.ranked.euclidean.agent",
        consumes=("scene.door_candidates", "agent_pose"),
        produces=("ranked_door_list", "distances"),
        description=(
            "List all visible doors ranked by Euclidean distance from the agent. "
            "Query only — no target is selected and no motion occurs."
        ),
        implementation_status="synthesizable",
        safe_to_synthesize=True,
        runtime_kind=None,
        runtime_value=None,
    ),
}
