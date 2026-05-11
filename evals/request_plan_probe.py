from __future__ import annotations

import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.capability_registry import CapabilityRegistry
from jeenom.readiness_graph import evaluate_request_plan
from jeenom.request_planner import build_request_plan
from jeenom.schemas import OperatorIntent


def _grounding_plan(
    *,
    operation: str,
    metric: str | None,
    order: str | None = None,
    ordinal: int | None = None,
    distance_value: int | None = None,
    comparison: str | None = None,
    required_capabilities: list[str] | None = None,
    answer_fields: list[str] | None = None,
) -> dict:
    primitive_handle = None
    if metric is not None:
        primitive_handle = f"grounding.all_doors.ranked.{metric}.agent"
    return {
        "object_type": "door",
        "operation": operation,
        "primitive_handle": primitive_handle,
        "metric": metric,
        "reference": "agent" if metric else None,
        "order": order,
        "ordinal": ordinal,
        "color": None,
        "exclude_colors": [],
        "distance_value": distance_value,
        "comparison": comparison,
        "tie_policy": "clarify",
        "answer_fields": answer_fields or [],
        "required_capabilities": required_capabilities or (
            [primitive_handle] if primitive_handle else []
        ),
        "preserved_constraints": [],
    }


CASES = [
    (
        "what doors do you see?",
        OperatorIntent(
            intent_type="status_query",
            status_query="scene",
            confidence=1.0,
            reason="Scene query.",
        ),
    ),
    (
        "how far are all doors from you?",
        OperatorIntent(
            intent_type="status_query",
            status_query="ground_target",
            grounding_query_plan=_grounding_plan(
                operation="rank",
                metric="manhattan",
                answer_fields=["distance"],
            ),
            confidence=1.0,
            reason="Rank visible doors by distance.",
        ),
    ),
    (
        "what is the farthest door?",
        OperatorIntent(
            intent_type="status_query",
            status_query="ground_target",
            grounding_query_plan=_grounding_plan(
                operation="answer",
                metric="manhattan",
                order="descending",
                ordinal=1,
                answer_fields=["farthest"],
            ),
            confidence=1.0,
            reason="Answer farthest door.",
        ),
    ),
    (
        "go to the closest door",
        OperatorIntent(
            intent_type="task_instruction",
            task_type="go_to_object",
            target_selector={
                "object_type": "door",
                "color": None,
                "exclude_colors": [],
                "relation": "closest",
                "distance_metric": None,
                "distance_reference": None,
            },
            capability_status="needs_clarification",
            confidence=1.0,
            reason="Closest target is missing the distance metric.",
        ),
    ),
    (
        "go to the door with the highest Euclidean distance below 10",
        OperatorIntent(
            intent_type="task_instruction",
            task_type="go_to_object",
            grounding_query_plan=_grounding_plan(
                operation="filter",
                metric="euclidean",
                order="descending",
                ordinal=1,
                distance_value=10,
                comparison="below",
                required_capabilities=[
                    "grounding.all_doors.ranked.euclidean.agent",
                    "claims.filter.threshold.euclidean",
                    "task.go_to_object.door",
                ],
                answer_fields=["distance"],
            ),
            capability_status="synthesizable",
            confidence=1.0,
            reason="Needs Euclidean ranking, threshold filtering, then task execution.",
        ),
    ),
    (
        "pick up the key",
        OperatorIntent(
            intent_type="unsupported",
            required_capabilities=["task.pickup.key"],
            capability_status="unsupported",
            confidence=1.0,
            reason="Pickup key task is unsupported.",
        ),
    ),
]


def main() -> int:
    registry = CapabilityRegistry.minigrid_default()
    print("REQUEST PLAN / READINESS GRAPH PROBE\n")
    failures = 0
    for utterance, intent in CASES:
        plan = build_request_plan(utterance, intent)
        graph = evaluate_request_plan(plan, registry=registry)
        print(f"CASE: {utterance}")
        print("PLAN")
        pprint(plan.as_dict())
        print("READINESS")
        pprint(graph.as_dict())
        print()

        if not plan.steps:
            failures += 1
        if utterance.startswith("go to the door with"):
            required = {step.required_handle for step in plan.steps}
            expected = {
                "grounding.all_doors.ranked.euclidean.agent",
                "claims.filter.threshold.euclidean",
                "task.go_to_object.door",
            }
            if not expected.issubset(required):
                failures += 1
            if graph.next_action != "propose_synthesis":
                failures += 1
        if utterance == "pick up the key" and graph.next_action != "refuse":
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
