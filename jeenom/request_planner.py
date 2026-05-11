from __future__ import annotations

import re
from typing import Any

from .schemas import OperatorIntent, RequestPlan, RequestPlanStep


_COLORS = ("red", "green", "blue", "yellow", "purple", "grey")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _signals(text: str) -> list[str]:
    normalized = _normalize(text)
    signals: list[str] = []
    checks = {
        "superlative.closest": ("closest", "nearest", "shortest"),
        "superlative.farthest": ("farthest", "furthest", "most distant", "least close"),
        "ordinal": ("first", "second", "third", "fourth", "fifth"),
        "cardinality.all": ("all", "each", "every"),
        "negation": ("not", "except", "other than"),
        "metric.euclidean": ("euclidean",),
        "metric.manhattan": ("manhattan",),
        "threshold": ("above", "below", "within", "at least", "at most"),
        "reference": ("that", "same", "previous", "last"),
    }
    for signal, terms in checks.items():
        if any(term in normalized for term in terms):
            signals.append(signal)
    for color in _COLORS:
        if re.search(rf"\b{color}\b", normalized):
            signals.append(f"color.{color}")
    if re.search(r"\bdoor(s)?\b", normalized):
        signals.append("object_type.door")
    return signals


def _objective_type(intent: OperatorIntent) -> str:
    if intent.intent_type == "task_instruction":
        return "task"
    if intent.intent_type == "knowledge_update":
        return "knowledge_update"
    if intent.intent_type in {"status_query", "claim_reference", "cache_query"}:
        return "query"
    if intent.intent_type in {"reset", "quit", "accept_proposal", "reject_proposal"}:
        return "control"
    return "unsupported"


def _expected_response(intent: OperatorIntent) -> str:
    if intent.capability_status == "needs_clarification":
        return "ask_clarification"
    if intent.capability_status == "synthesizable":
        return "propose_synthesis"
    if intent.intent_type == "task_instruction":
        return "execute_task"
    if intent.intent_type == "knowledge_update":
        return "update_memory"
    if intent.intent_type in {"status_query", "claim_reference", "cache_query"}:
        return "answer_query"
    return "refuse"


def _metric_from_text_or_plan(text: str, plan: dict[str, Any] | None) -> str | None:
    if plan is not None and plan.get("metric"):
        return str(plan["metric"])
    normalized = _normalize(text)
    if "euclidean" in normalized:
        return "euclidean"
    if "manhattan" in normalized:
        return "manhattan"
    return None


def _comparison_from_text_or_plan(text: str, plan: dict[str, Any] | None) -> str | None:
    if plan is not None and plan.get("comparison"):
        return str(plan["comparison"])
    normalized = _normalize(text)
    if "at least" in normalized:
        return "at_least"
    if "at most" in normalized or "within" in normalized:
        return "at_most"
    if "above" in normalized or "greater than" in normalized or "over " in normalized:
        return "above"
    if "below" in normalized or "less than" in normalized or "under " in normalized:
        return "below"
    return None


def _distance_value_from_text_or_plan(text: str, plan: dict[str, Any] | None) -> int | None:
    if plan is not None and plan.get("distance_value") is not None:
        return int(plan["distance_value"])
    match = re.search(r"\b(\d+)\b", text)
    if match is None:
        return None
    return int(match.group(1))


def _ranked_handle(metric: str | None) -> str | None:
    if metric is None:
        return None
    return f"grounding.all_doors.ranked.{metric}.agent"


def _filter_handle(metric: str | None) -> str | None:
    if metric is None:
        return None
    return f"claims.filter.threshold.{metric}"


def _target_handle(intent: OperatorIntent) -> str | None:
    selector = intent.target_selector or {}
    relation = selector.get("relation")
    metric = selector.get("distance_metric")
    reference = selector.get("distance_reference")
    if relation == "closest":
        if metric is None or reference is None:
            return None
        return f"grounding.closest_door.{metric}.{reference}"
    if selector:
        return "grounding.unique_door.color_filter"
    target = intent.target or {}
    if target.get("object_type") == "door":
        return "grounding.unique_door.color_filter"
    return None


def _operation_from_query_plan(plan: dict[str, Any] | None) -> str:
    if plan is None:
        return "answer"
    operation = str(plan.get("operation") or "answer")
    if operation in {"list", "rank"}:
        return "rank"
    if operation in {"filter", "select", "answer"}:
        return operation
    return "answer"


def build_request_plan(
    utterance: str,
    intent: OperatorIntent,
    *,
    active_claims_summary: dict[str, Any] | None = None,
) -> RequestPlan:
    """Build a typed, dependency-aware request plan from validated operator intent.

    This is intentionally side-effect free. The station and readiness graph decide
    whether the plan can execute, clarify, synthesize, or refuse.
    """

    request_id = f"request:{abs(hash((utterance, intent.intent_type))) % 1_000_000}"
    objective_type = _objective_type(intent)
    expected_response = _expected_response(intent)
    steps: list[RequestPlanStep] = []
    plan = intent.grounding_query_plan
    metric = _metric_from_text_or_plan(utterance, plan)
    comparison = _comparison_from_text_or_plan(utterance, plan)
    distance_value = _distance_value_from_text_or_plan(utterance, plan)

    if objective_type == "control":
        steps.append(
            RequestPlanStep(
                step_id="control",
                layer="control",
                operation="reset" if intent.intent_type == "reset" else "answer",
                outputs=["control_response"],
            )
        )
        return RequestPlan(
            request_id=request_id,
            original_utterance=utterance,
            objective_type=objective_type,
            objective_summary=intent.reason or intent.intent_type,
            steps=steps,
            preservation_signals=_signals(utterance),
            expected_response=expected_response,
        )

    if objective_type == "unsupported" and intent.required_capabilities:
        for idx, handle in enumerate(intent.required_capabilities):
            layer = handle.split(".", 1)[0] if "." in handle else "control"
            steps.append(
                RequestPlanStep(
                    step_id=f"unsupported_capability_{idx + 1}",
                    layer=layer,
                    operation="refuse",
                    required_handle=handle,
                    outputs=["operator_response"],
                )
            )

    if objective_type == "knowledge_update":
        steps.append(
            RequestPlanStep(
                step_id="update_knowledge",
                layer="memory",
                operation="update",
                inputs={"knowledge_update": intent.knowledge_update or {}},
                outputs=["durable_knowledge"],
                memory_writes=["knowledge.delivery_target"],
            )
        )

    if intent.reference == "delivery_target":
        steps.append(
            RequestPlanStep(
                step_id="read_delivery_target",
                layer="memory",
                operation="read",
                outputs=["target_fact"],
                memory_reads=["knowledge.delivery_target"],
            )
        )
    if intent.reference in {"last_target", "last_task"}:
        steps.append(
            RequestPlanStep(
                step_id=f"read_{intent.reference}",
                layer="memory",
                operation="read",
                outputs=[intent.reference],
                memory_reads=[f"episodic.{intent.reference}"],
            )
        )

    ranked_handle = None
    if plan is not None:
        ranked_handle = plan.get("primitive_handle")
        required = list(plan.get("required_capabilities") or [])
        if ranked_handle is None:
            ranked_handle = next(
                (handle for handle in required if "all_doors.ranked" in handle),
                _ranked_handle(metric),
            )
        if plan.get("operation") in {"rank", "list", "answer", "filter", "select"}:
            steps.append(
                RequestPlanStep(
                    step_id="rank_scene_doors",
                    layer="grounding",
                    operation="rank",
                    required_handle=ranked_handle,
                    inputs={"object_type": plan.get("object_type", "door")},
                    outputs=["active_claims.ranked_scene_doors"],
                    constraints={
                        "metric": metric,
                        "reference": plan.get("reference") or "agent",
                    },
                )
            )
    elif intent.target_selector is not None:
        handle = _target_handle(intent)
        steps.append(
            RequestPlanStep(
                step_id="ground_target",
                layer="grounding",
                operation="ground",
                required_handle=handle,
                inputs={"target_selector": intent.target_selector},
                outputs=["grounded_target"],
                constraints=dict(intent.target_selector),
            )
        )

    if plan is not None and distance_value is not None:
        depends = ["rank_scene_doors"] if any(s.step_id == "rank_scene_doors" for s in steps) else []
        steps.append(
            RequestPlanStep(
                step_id="filter_distance_threshold",
                layer="claims",
                operation="filter",
                required_handle=_filter_handle(metric),
                inputs={"entries": "active_claims.ranked_scene_doors"},
                outputs=["filtered_candidates"],
                depends_on=depends,
                constraints={
                    "metric": metric,
                    "comparison": comparison,
                    "threshold": distance_value,
                },
                memory_reads=["active_claims.ranked_scene_doors"],
                scene_fingerprint_required=True,
            )
        )

    if plan is not None and (
        plan.get("ordinal") is not None
        or plan.get("order") is not None
        or any(str(field).endswith(("closest", "farthest")) for field in plan.get("answer_fields", []))
    ):
        source = (
            "filtered_candidates"
            if any(s.step_id == "filter_distance_threshold" for s in steps)
            else "active_claims.ranked_scene_doors"
        )
        depends = [
            steps[-1].step_id
        ] if steps else []
        steps.append(
            RequestPlanStep(
                step_id="select_grounded_target",
                layer="claims",
                operation="select",
                inputs={"entries": source},
                outputs=["grounded_target"],
                depends_on=depends,
                constraints={
                    "order": plan.get("order"),
                    "ordinal": plan.get("ordinal"),
                    "answer_fields": list(plan.get("answer_fields") or []),
                    "metric": metric,
                },
                tie_policy=plan.get("tie_policy") or "clarify",
                memory_reads=[source],
                scene_fingerprint_required=True,
            )
        )

    if objective_type == "task":
        task_dep = []
        if any(s.outputs and "grounded_target" in s.outputs for s in steps):
            task_dep = [steps[-1].step_id]
        target = intent.target or {}
        object_type = target.get("object_type") or (intent.target_selector or {}).get("object_type")
        if object_type is None and intent.task_type == "go_to_object":
            object_type = "door"
        steps.append(
            RequestPlanStep(
                step_id="execute_task",
                layer="task",
                operation="execute",
                required_handle=(
                    "task.go_to_object.door"
                    if intent.task_type == "go_to_object" or object_type == "door"
                    else None
                ),
                inputs={"task_type": intent.task_type, "object_type": object_type},
                outputs=["task_result"],
                depends_on=task_dep,
                memory_writes=[
                    "episodic.last_request_plan",
                    "episodic.last_grounded_target",
                    "episodic.last_task",
                    "episodic.last_result",
                ],
            )
        )

    if objective_type == "query":
        if not steps and intent.status_query in {"status", "help", "cache", "scene"}:
            steps.append(
                RequestPlanStep(
                    step_id=f"answer_{intent.status_query or 'status'}",
                    layer="answer",
                    operation="answer",
                    outputs=["operator_response"],
                    memory_reads=["knowledge.delivery_target", "episodic.last_result"],
                )
            )
        else:
            depends = [steps[-1].step_id] if steps else []
            steps.append(
                RequestPlanStep(
                    step_id="answer_query",
                    layer="answer",
                    operation="answer",
                    outputs=["operator_response"],
                    depends_on=depends,
                    memory_reads=["active_claims.ranked_scene_doors"]
                    if active_claims_summary is not None
                    else [],
                )
            )

    if not steps:
        steps.append(
            RequestPlanStep(
                step_id="refuse",
                layer="control",
                operation="refuse",
                outputs=["operator_response"],
            )
        )

    return RequestPlan(
        request_id=request_id,
        original_utterance=utterance,
        objective_type=objective_type,
        objective_summary=intent.reason or intent.canonical_instruction or intent.intent_type,
        steps=steps,
        preservation_signals=_signals(utterance),
        expected_response=expected_response,
    )
