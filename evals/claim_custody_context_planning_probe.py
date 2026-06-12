"""Phase 10H probe: planner/verifier consume OperationalContext semantics."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from harness import emit_result, make_session


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        from jeenom.intent_verifier import IntentVerifier
        from jeenom.planning_semantics import PlanningSemantics
        from jeenom.request_planner import build_request_plan
        from jeenom.schemas import OperationalContext, OperatorIntent
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        IntentVerifier = None  # type: ignore[assignment]
        PlanningSemantics = None  # type: ignore[assignment]
        build_request_plan = None  # type: ignore[assignment]
        OperationalContext = None  # type: ignore[assignment]
        OperatorIntent = None  # type: ignore[assignment]
        details["context_planning_import_error"] = f"{type(exc).__name__}: {exc}"

    metrics["planning_semantics_module_exists"] = PlanningSemantics is not None
    metrics["request_planner_accepts_planning_semantics"] = (
        build_request_plan is not None
        and "planning_semantics" in inspect.signature(build_request_plan).parameters
    )
    metrics["intent_verifier_accepts_planning_semantics"] = (
        IntentVerifier is not None
        and "planning_semantics" in inspect.signature(IntentVerifier).parameters
    )

    if all((PlanningSemantics, build_request_plan, OperationalContext, OperatorIntent)):
        try:
            token_context = OperationalContext(
                context_id="probe.tokens",
                substrate_id="probe",
                object_vocabulary=["token"],
                attribute_vocabulary=["name", "score"],
                grounding_semantics={
                    "distance_metrics": ["score"],
                    "distance_references": ["agent"],
                    "ranked_claims_output": "active_claims.ranked_tokens",
                    "capability_handles": {
                        "ranked": "grounding.all_{object_type_plural}.ranked.{metric}.agent",
                        "filter_threshold": "claims.filter.threshold.{metric}",
                        "unique": "grounding.unique_{object_type}.name_filter",
                        "task_go_to_object": "task.go_to_object.{object_type}",
                    },
                },
                reference_semantics={
                    "closest": {"default_metric": "score", "reference": "agent"},
                },
            )
            semantics = PlanningSemantics(token_context)
            intent = OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                grounding_query_plan={
                    "operation": "rank",
                    "object_type": "token",
                    "metric": "score",
                    "primitive_handle": None,
                    "required_capabilities": [],
                },
                confidence=1.0,
                reason="Probe token ranking.",
            )
            plan = build_request_plan(
                "rank all tokens by score",
                intent,
                planning_semantics=semantics,
            )
            plan_repr = repr(plan.as_dict())
            details["token_plan"] = plan.as_dict()
            required_handles = [
                step.required_handle
                for step in plan.steps
                if step.required_handle is not None
            ]
            metrics["token_context_plan_uses_token_handle"] = (
                "grounding.all_tokens.ranked.score.agent" in required_handles
            )
            metrics["token_context_plan_not_minigrid_door_default"] = "all_doors" not in plan_repr
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["context_plan_error"] = f"{type(exc).__name__}: {exc}"
            metrics["token_context_plan_uses_token_handle"] = False
            metrics["token_context_plan_not_minigrid_door_default"] = False
    else:
        metrics["token_context_plan_uses_token_handle"] = False
        metrics["token_context_plan_not_minigrid_door_default"] = False

    if all((PlanningSemantics, IntentVerifier, OperationalContext, OperatorIntent)):
        try:
            token_context = OperationalContext(
                context_id="probe.tokens",
                substrate_id="probe",
                object_vocabulary=["token"],
                attribute_vocabulary=["name", "score"],
                grounding_semantics={
                    "distance_metrics": ["score"],
                    "distance_references": ["agent"],
                    "capability_handles": {
                        "ranked": "grounding.all_{object_type_plural}.ranked.{metric}.agent",
                    },
                },
                reference_semantics={"closest": {"default_metric": "score"}},
            )
            semantics = PlanningSemantics(token_context)
            verifier = IntentVerifier(planning_semantics=semantics)
            intent = OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                required_capabilities=[],
                confidence=1.0,
                reason="Probe token ranking.",
            )
            enriched, result = verifier.enrich("rank all tokens by score", intent)
            details["token_verifier_handles"] = list(enriched.required_capabilities)
            metrics["token_context_verifier_injects_token_handle"] = (
                "grounding.all_tokens.ranked.score.agent"
                in set(enriched.required_capabilities or [])
            )
            metrics["token_context_verifier_not_minigrid_door_default"] = not any(
                "all_doors" in handle
                for handle in (enriched.required_capabilities or [])
            )
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["context_verifier_error"] = f"{type(exc).__name__}: {exc}"
            metrics["token_context_verifier_injects_token_handle"] = False
            metrics["token_context_verifier_not_minigrid_door_default"] = False
    else:
        metrics["token_context_verifier_injects_token_handle"] = False
        metrics["token_context_verifier_not_minigrid_door_default"] = False

    try:
        session = make_session()
        metrics["station_has_planning_semantics"] = hasattr(session, "planning_semantics")
        metrics["station_verifier_bound_to_planning_semantics"] = (
            getattr(getattr(session, "intent_verifier", None), "planning_semantics", None)
            is getattr(session, "planning_semantics", None)
        )
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        details["station_semantics_error"] = f"{type(exc).__name__}: {exc}"
        metrics["station_has_planning_semantics"] = False
        metrics["station_verifier_bound_to_planning_semantics"] = False

    request_source = (ROOT / "jeenom" / "request_planner.py").read_text()
    verifier_source = (ROOT / "jeenom" / "intent_verifier.py").read_text()
    forbidden_planner_literals = [
        "grounding.all_doors.ranked",
        "grounding.closest_door.",
        "grounding.unique_door.color_filter",
        "task.go_to_object.door",
        "_COLORS",
    ]
    forbidden_verifier_literals = [
        "grounding.all_doors.ranked",
        "door|doors",
        '"door"',
    ]
    details["remaining_request_planner_literals"] = [
        literal for literal in forbidden_planner_literals if literal in request_source
    ]
    details["remaining_intent_verifier_literals"] = [
        literal for literal in forbidden_verifier_literals if literal in verifier_source
    ]
    metrics["request_planner_core_literals_extracted"] = not details[
        "remaining_request_planner_literals"
    ]
    metrics["intent_verifier_core_literals_extracted"] = not details[
        "remaining_intent_verifier_literals"
    ]

    metrics["phase10_context_planning_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase10_context_planning_holds")


if __name__ == "__main__":
    raise SystemExit(main())
