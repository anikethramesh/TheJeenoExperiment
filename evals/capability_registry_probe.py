from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from pprint import pprint
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import LLMCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.primitive_library import (
    ACTION_PRIMITIVES,
    GROUNDING_PRIMITIVES,
    SENSING_PRIMITIVES,
    TASK_PRIMITIVES,
)


def _names(prefix: str, source: dict[str, Any]) -> set[str]:
    return {f"{prefix}.{name}" for name in source}


def main() -> int:
    registry = CapabilityRegistry.minigrid_default()
    summary = registry.compact_summary()
    registry_names = set(registry.primitive_names())

    task_names = _names("task", TASK_PRIMITIVES)
    sensing_names = _names("sensing", SENSING_PRIMITIVES)
    action_names = _names("action", ACTION_PRIMITIVES)
    grounding_names = _names("grounding", GROUNDING_PRIMITIVES)

    help_session = OperatorStationSession(
        compiler=LLMCompiler(api_key="test-key"),
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    help_response = help_session.handle_utterance("what can you do")

    plan_grid_path = registry.primitive("action.plan_grid_path")
    euclidean = registry.readiness_for_selector(
        {
            "object_type": "door",
            "color": None,
            "exclude_color": None,
            "relation": "closest",
            "distance_metric": "euclidean",
            "distance_reference": "agent",
        }
    )
    pickup = registry.readiness_for_task(task_type="pickup", object_type="key")

    checks = {
        "registry_name": summary["name"] == "minigrid_primitive_registry_v1",
        "lists_all_task_primitives": task_names.issubset(registry_names),
        "lists_all_sensing_primitives": sensing_names.issubset(registry_names),
        "lists_all_action_primitives": action_names.issubset(registry_names),
        "lists_grounding_primitives": grounding_names.issubset(registry_names),
        "compact_summary_has_all_layers": {"task", "grounding", "sensing", "action"}.issubset(
            set(summary["primitives"])
        ),
        "exposes_consumes_produces": bool(plan_grid_path)
        and plan_grid_path.inputs == ["agent_pose", "target_location", "occupancy_grid"]
        and plan_grid_path.outputs == ["planned_action_names", "path"],
        "exposes_runtime_binding": bool(plan_grid_path)
        and plan_grid_path.runtime_binding == {
            "kind": "python",
            "value": "plan_grid_path",
        },
        "euclidean_is_synthesizable": euclidean["status"]
        == "synthesizable_missing_primitive",
        "pickup_key_is_unsupported_task": pickup["status"] == "unsupported"
        and pickup["primitive"] == "task.pickup.key",
        "help_query_uses_registry": help_response.startswith("CAPABILITIES")
        and "task.go_to_object.door" in help_response
        and "sensing.parse_grid_objects" in help_response
        and "action.plan_grid_path" in help_response
        and "grounding.closest_door.euclidean.agent" in help_response,
    }

    print("CAPABILITY REGISTRY PROBE")
    print()
    print("SOURCE COUNTS")
    pprint(
        {
            "TASK_PRIMITIVES": len(TASK_PRIMITIVES),
            "GROUNDING_PRIMITIVES": len(GROUNDING_PRIMITIVES),
            "SENSING_PRIMITIVES": len(SENSING_PRIMITIVES),
            "ACTION_PRIMITIVES": len(ACTION_PRIMITIVES),
            "registry_total": len(registry_names),
        }
    )
    print()
    print("REGISTRY")
    pprint(
        {
            "name": summary["name"],
            "layers": {
                layer: len(items)
                for layer, items in summary["primitives"].items()
            },
        }
    )
    print()
    print("READINESS")
    pprint({"euclidean": euclidean, "pickup_key": pickup})
    print()
    print("HELP RESPONSE")
    print(help_response)
    print()
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
