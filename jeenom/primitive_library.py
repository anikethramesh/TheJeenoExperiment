from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    description: str = ""
    safety_notes: str | None = None
    required_action_primitives: tuple[str, ...] = ()
    runtime_kind: str | None = None
    runtime_value: int | str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


TASK_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "locate_object": PrimitiveSpec(
        name="locate_object",
        consumes=("object_location",),
        produces=("target_location_candidate",),
        description="Find a target object in the current world model.",
        safety_notes="Requires a grounded object query; do not hallucinate target locations.",
    ),
    "navigate_to_object": PrimitiveSpec(
        name="navigate_to_object",
        consumes=("agent_pose", "object_location", "occupancy_grid"),
        produces=("adjacency_to_target",),
        description="Move the agent toward the target using available navigation primitives.",
        safety_notes="Must only use validated action primitives; never call env.step directly from the compiler.",
        required_action_primitives=(
            "plan_grid_path",
            "execute_next_path_action",
            "turn_left",
            "turn_right",
            "move_forward",
        ),
    ),
    "verify_adjacent": PrimitiveSpec(
        name="verify_adjacent",
        consumes=("adjacency_to_target",),
        produces=("target_verified",),
        description="Check whether the agent reached the required completion position.",
        safety_notes="Only succeed when the runtime evidence indicates adjacency.",
    ),
    "done": PrimitiveSpec(
        name="done",
        consumes=("adjacency_to_target",),
        produces=("task_complete",),
        description="Issue the completion action once the target condition is met.",
        safety_notes="Must not fire before the runtime verifies the completion condition.",
        required_action_primitives=("done",),
    ),
    "ask_operator": PrimitiveSpec(
        name="ask_operator",
        consumes=(),
        produces=("clarification",),
        description="Request clarification from the operator.",
        safety_notes="Use only when the task is underspecified.",
    ),
    "replan": PrimitiveSpec(
        name="replan",
        consumes=("error_state",),
        produces=("new_procedure",),
        description="Recompose the plan when the current one is blocked.",
        safety_notes="Use after a validated failure, not preemptively.",
    ),
    "abort": PrimitiveSpec(
        name="abort",
        consumes=(),
        produces=("aborted",),
        description="Stop execution when the task cannot be completed safely.",
        safety_notes="Use only for impossible or unsafe states.",
    ),
}


SENSING_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "parse_grid_objects": PrimitiveSpec(
        name="parse_grid_objects",
        consumes=("raw_image",),
        produces=("grid_objects", "agent_pose"),
        description="Parse the fully observable MiniGrid image into symbolic objects.",
        safety_notes="The runtime parser owns the actual decoding logic.",
    ),
    "find_object_by_color_type": PrimitiveSpec(
        name="find_object_by_color_type",
        consumes=("grid_objects",),
        produces=("object_location", "target_visible", "target_object"),
        description="Select the target object by color and type from parsed objects.",
    ),
    "get_agent_pose": PrimitiveSpec(
        name="get_agent_pose",
        consumes=("grid_objects",),
        produces=("agent_pose",),
        description="Recover the agent position and orientation from the parsed grid.",
    ),
    "check_adjacency": PrimitiveSpec(
        name="check_adjacency",
        consumes=("agent_pose", "object_location"),
        produces=("adjacency_to_target",),
        description="Check whether the agent is in the completion position relative to the target.",
    ),
    "build_occupancy_grid": PrimitiveSpec(
        name="build_occupancy_grid",
        consumes=("grid_objects", "raw_image"),
        produces=("occupancy_grid", "passable_positions", "grid_size"),
        description="Build a passability map for navigation from parsed grid content.",
    ),
}


ACTION_PRIMITIVES: dict[str, PrimitiveSpec] = {
    "plan_grid_path": PrimitiveSpec(
        name="plan_grid_path",
        consumes=("agent_pose", "target_location", "occupancy_grid"),
        produces=("planned_action_names", "path"),
        description="Plan a path in the MiniGrid occupancy map.",
        safety_notes="Runtime path planner must reject unreachable targets.",
        runtime_kind="python",
        runtime_value="plan_grid_path",
    ),
    "execute_next_path_action": PrimitiveSpec(
        name="execute_next_path_action",
        consumes=("planned_action_names",),
        produces=("executed_action",),
        description="Execute the first action from the current planned path.",
        runtime_kind="python",
        runtime_value="execute_next_path_action",
    ),
    "turn_left": PrimitiveSpec(
        name="turn_left",
        description="Turn the MiniGrid agent left.",
        runtime_kind="env_action",
        runtime_value=0,
    ),
    "turn_right": PrimitiveSpec(
        name="turn_right",
        description="Turn the MiniGrid agent right.",
        runtime_kind="env_action",
        runtime_value=1,
    ),
    "move_forward": PrimitiveSpec(
        name="move_forward",
        description="Move the MiniGrid agent forward.",
        runtime_kind="env_action",
        runtime_value=2,
    ),
    "pickup": PrimitiveSpec(
        name="pickup",
        description="Pick up an item in MiniGrid.",
        runtime_kind="env_action",
        runtime_value=3,
    ),
    "drop": PrimitiveSpec(
        name="drop",
        description="Drop the currently carried item in MiniGrid.",
        runtime_kind="env_action",
        runtime_value=4,
    ),
    "toggle": PrimitiveSpec(
        name="toggle",
        description="Toggle the front cell interaction in MiniGrid.",
        runtime_kind="env_action",
        runtime_value=5,
    ),
    "done": PrimitiveSpec(
        name="done",
        description="Emit the MiniGrid done action.",
        runtime_kind="env_action",
        runtime_value=6,
    ),
}


def library_payload(library: dict[str, PrimitiveSpec]) -> list[dict[str, Any]]:
    return [library[name].payload() for name in sorted(library)]


def primitive_names(library: dict[str, PrimitiveSpec]) -> list[str]:
    return sorted(library)


def produced_evidence(library: dict[str, PrimitiveSpec]) -> set[str]:
    outputs: set[str] = set()
    for spec in library.values():
        outputs.update(spec.produces)
    return outputs
