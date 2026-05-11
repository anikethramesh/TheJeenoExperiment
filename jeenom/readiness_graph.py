from __future__ import annotations

from .capability_registry import CapabilityRegistry
from .schemas import (
    ReadinessGraph,
    ReadinessNode,
    RequestPlan,
    RequestPlanStep,
    StationActiveClaims,
)


def _status_for_primitive(
    step: RequestPlanStep,
    registry: CapabilityRegistry,
) -> tuple[str, str]:
    handle = step.required_handle
    if handle is None:
        if step.operation in {"answer", "read", "update", "reset", "refuse", "select"}:
            return "executable", "No primitive handle required for this plan step."
        return "needs_clarification", "Plan step is missing a required primitive handle."

    spec = registry.lookup(handle)
    if spec is None:
        return "missing_skills", f"Primitive '{handle}' is not present in the registry."
    if spec.implementation_status == "implemented":
        return "executable", spec.description
    if spec.implementation_status == "synthesizable" or spec.safe_to_synthesize:
        return "synthesizable", spec.description
    if spec.implementation_status in {"planned", "missing"}:
        return "missing_skills", spec.description
    return "unsupported", spec.description


def _claims_status(
    step: RequestPlanStep,
    *,
    active_claims: StationActiveClaims | None,
    claims_valid: bool,
) -> tuple[str | None, str | None]:
    if not step.scene_fingerprint_required:
        return None, None
    if active_claims is None:
        return "stale_claims", "Step needs ActiveClaims, but none are available yet."
    if not claims_valid:
        return "stale_claims", "ActiveClaims exist but do not match the current scene."
    return None, None


def evaluate_request_plan(
    plan: RequestPlan,
    *,
    registry: CapabilityRegistry,
    active_claims: StationActiveClaims | None = None,
    claims_valid: bool = False,
) -> ReadinessGraph:
    nodes: list[ReadinessNode] = []
    statuses_by_step: dict[str, str] = {}

    for step in plan.steps:
        blocking_dependencies = [
            dep
            for dep in step.depends_on
            if statuses_by_step.get(dep) not in {None, "executable"}
        ]
        if blocking_dependencies:
            status = "blocked_by_dependency"
            reason = "Blocked by non-executable dependency."
        else:
            claims_status, claims_reason = _claims_status(
                step,
                active_claims=active_claims,
                claims_valid=claims_valid,
            )
            if claims_status is not None:
                status = claims_status
                reason = claims_reason or ""
            else:
                status, reason = _status_for_primitive(step, registry)

        statuses_by_step[step.step_id] = status
        nodes.append(
            ReadinessNode(
                step_id=step.step_id,
                status=status,
                layer=step.layer,
                operation=step.operation,
                required_handle=step.required_handle,
                reason=reason,
                blocking_dependencies=blocking_dependencies,
            )
        )

    graph_status = "executable"
    blocking = next((node for node in nodes if node.status != "executable"), None)
    if blocking is not None:
        graph_status = blocking.status

    next_action = _next_action(plan, graph_status, blocking)
    explanation = (
        "RequestPlan is ready."
        if blocking is None
        else f"{blocking.step_id}: {blocking.reason}"
    )
    return ReadinessGraph(
        request_id=plan.request_id,
        nodes=nodes,
        graph_status=graph_status,
        next_action=next_action,
        blocking_step_id=blocking.step_id if blocking is not None else None,
        explanation=explanation,
    )


def _next_action(
    plan: RequestPlan,
    graph_status: str,
    blocking: ReadinessNode | None,
) -> str:
    if blocking is None:
        if plan.expected_response == "execute_task":
            return "execute_task"
        if plan.expected_response == "update_memory":
            return "update_memory"
        if plan.expected_response == "answer_query":
            return "answer_query"
        if plan.expected_response == "refuse":
            return "refuse"
        return "answer_query"
    if graph_status == "needs_clarification":
        return "ask_clarification"
    if graph_status == "synthesizable":
        return "propose_synthesis"
    if graph_status == "stale_claims":
        return "refresh_claims"
    return "refuse"
