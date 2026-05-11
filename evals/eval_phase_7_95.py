"""Consolidated Phase 7.95 eval — RequestPlan and ReadinessGraph.

Covers:
- RequestPlan decomposition from OperatorIntent + GroundingQueryPlan.
- ReadinessGraph evaluation against CapabilityRegistry.
- Multi-step dependency chain for complex utterances.
- Graph next_action verdicts (execute, propose_synthesis, refuse, clarify).
- Station episodic memory recording of plan/graph.

Migrated from: request_plan_probe.py (expanded with authoritative dispatch tests).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from pprint import pprint
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
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


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session() -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )


def _run(fn):
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        return fn()


def _check_plan_decomposition(checks: dict[str, bool]) -> None:
    registry = CapabilityRegistry.minigrid_default()
    for utterance, intent in CASES:
        plan = build_request_plan(utterance, intent)
        graph = evaluate_request_plan(plan, registry=registry)

        tag = utterance[:30].replace(" ", "_").replace("?", "")
        checks[f"{tag}_has_steps"] = len(plan.steps) > 0

        if utterance.startswith("go to the door with"):
            required = {step.required_handle for step in plan.steps}
            expected = {
                "grounding.all_doors.ranked.euclidean.agent",
                "claims.filter.threshold.euclidean",
                "task.go_to_object.door",
            }
            checks["euclidean_filter_has_all_steps"] = expected.issubset(required)
            checks["euclidean_filter_proposes_synthesis"] = graph.next_action == "propose_synthesis"

        if utterance == "pick up the key":
            checks["pickup_key_refuses"] = graph.next_action == "refuse"

        if utterance == "what doors do you see?":
            checks["scene_query_is_answerable"] = graph.next_action in ("answer_query", "execute")

        if utterance == "go to the closest door":
            checks["closest_needs_clarification"] = graph.next_action in (
                "ask_clarification", "needs_clarification"
            )

    print("PLAN DECOMPOSITION checks complete\n")


def _check_station_episodic_recording(checks: dict[str, bool]) -> None:
    session = _make_session()
    _run(lambda: session.handle_utterance("go to the red door"))

    # Use a query that triggers plan recording
    _run(lambda: session.handle_utterance("which door is closest by manhattan distance"))

    episodic = session.memory.episodic_memory or {}
    checks["episodic_has_last_request_plan"] = "last_request_plan" in episodic
    checks["episodic_has_last_readiness_graph"] = "last_readiness_graph" in episodic

    print("STATION EPISODIC RECORDING checks complete\n")


def main() -> int:
    checks: dict[str, bool] = {}

    print("CONSOLIDATED EVAL: PHASE 7.95 (RequestPlan + ReadinessGraph)\n")

    print("── Plan Decomposition ──")
    _check_plan_decomposition(checks)

    print("── Station Episodic Recording ──")
    _check_station_episodic_recording(checks)

    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    n_pass = sum(checks.values())
    print(f"\n{n_pass}/{len(checks)} passed")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
