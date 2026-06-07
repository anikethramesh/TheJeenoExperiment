"""Phase 10B probe: side-effect tickets leave OperatorStationSession.

Ticket minting is execution authority, not presentation logic. The station may
still orchestrate a turn, but it should not directly construct task, motor, or
memory tickets inline.
"""
from __future__ import annotations

import ast
from typing import Any

from harness import ROOT, emit_result, make_session


TICKET_CONSTRUCTORS = {"ExecutionTicket", "RawMotorTicket", "MemoryWriteTicket"}


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _call_hits(tree: ast.AST, names: set[str]) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called: str | None = None
        if isinstance(node.func, ast.Name):
            called = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called = node.func.attr
        if called in names:
            hits.append((node.lineno, called))
    return hits


def _function_call_names(tree: ast.AST, name: str) -> list[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != name:
            continue
        calls: list[str] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
        return calls
    return []


def _sample_task_plan_and_graph():
    from jeenom.schemas import ReadinessGraph, RequestPlan, RequestPlanStep

    plan = RequestPlan(
        request_id="phase10-side-effect-task",
        original_utterance="go to the red door",
        objective_type="task",
        objective_summary="Execute a task.",
        expected_response="execute_task",
        steps=[
            RequestPlanStep(
                step_id="execute_task",
                layer="spine",
                operation="execute",
            )
        ],
    )
    graph = ReadinessGraph(
        request_id=plan.request_id,
        graph_status="executable",
        next_action="execute_task",
        explanation="Unit probe executable task.",
    )
    return plan, graph


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        import jeenom.side_effect_authority as side_effect_authority
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        side_effect_authority = None  # type: ignore[assignment]
        details["side_effect_authority_import_error"] = f"{type(exc).__name__}: {exc}"

    authority_cls = (
        getattr(side_effect_authority, "SideEffectAuthority", None)
        if side_effect_authority is not None
        else None
    )
    metrics["side_effect_authority_module_exists"] = side_effect_authority is not None
    metrics["side_effect_authority_class_exists"] = authority_cls is not None

    if side_effect_authority is not None:
        source = _source("jeenom/side_effect_authority.py")
        metrics["side_effect_authority_has_no_operator_station_dependency"] = (
            "operator_station" not in source
        )
    else:
        metrics["side_effect_authority_has_no_operator_station_dependency"] = False

    station_tree = ast.parse(_source("jeenom/operator_station.py"))
    constructor_hits = _call_hits(station_tree, TICKET_CONSTRUCTORS)
    details["station_direct_ticket_constructor_hits"] = constructor_hits
    metrics["station_does_not_construct_side_effect_tickets"] = not constructor_hits

    execution_calls = _function_call_names(station_tree, "_execution_ticket_from_plan")
    raw_motor_calls = _function_call_names(station_tree, "_raw_motor_ticket_from_plan")
    memory_calls = _function_call_names(station_tree, "_memory_write_ticket_for_payload")
    details["execution_ticket_from_plan_calls"] = sorted(set(execution_calls))
    details["raw_motor_ticket_from_plan_calls"] = sorted(set(raw_motor_calls))
    details["memory_write_ticket_for_payload_calls"] = sorted(set(memory_calls))
    metrics["station_execution_ticket_delegates_to_authority"] = (
        "issue_execution_ticket" in execution_calls
    )
    metrics["station_raw_motor_ticket_delegates_to_authority"] = (
        "issue_raw_motor_ticket" in raw_motor_calls
    )
    metrics["station_memory_write_ticket_delegates_to_authority"] = (
        "issue_memory_write_ticket" in memory_calls
    )

    if authority_cls is not None:
        session = make_session()
        metrics["operator_station_has_side_effect_authority"] = isinstance(
            getattr(session, "side_effect_authority", None),
            authority_cls,
        )
        try:
            from jeenom.schemas import ExecutionTicket

            plan, graph = _sample_task_plan_and_graph()
            ticket = authority_cls(source_name="Phase10Probe").issue_execution_ticket(
                instruction="go to the red door",
                task_type="go_to_object",
                params={"object_type": "door", "color": "red"},
                request_plan=plan,
                readiness_graph=graph,
                source="probe",
            )
            metrics["side_effect_authority_issues_typed_execution_ticket"] = isinstance(
                ticket,
                ExecutionTicket,
            )
            metrics["side_effect_authority_preserves_plan_graph_ids"] = (
                ticket.request_id == plan.request_id
                and ticket.request_plan is plan
                and ticket.readiness_graph is graph
            )
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["side_effect_authority_behavior_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            metrics["side_effect_authority_issues_typed_execution_ticket"] = False
            metrics["side_effect_authority_preserves_plan_graph_ids"] = False
    else:
        metrics["operator_station_has_side_effect_authority"] = False
        metrics["side_effect_authority_issues_typed_execution_ticket"] = False
        metrics["side_effect_authority_preserves_plan_graph_ids"] = False

    metrics["phase10_side_effect_authority_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase10_side_effect_authority_holds")


if __name__ == "__main__":
    raise SystemExit(main())
