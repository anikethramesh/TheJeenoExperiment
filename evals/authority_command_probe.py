"""Phase 10 probe: command/result authority leaves OperatorStationSession.

The operator station may remain the user-facing facade, but it should not be the
place where cortical envelopes, approved commands, and command results are
manufactured inline. That authority boundary needs a small typed service.
"""
from __future__ import annotations

import ast
from typing import Any

from harness import ROOT, ast_call_names, ast_function_call_names, ast_source, emit_result, make_session


TRACE_CONSTRUCTORS = {"CorticalEnvelope", "ApprovedCommand", "CommandResult"}


def _sample_plan_and_graph():
    from jeenom.schemas import ReadinessGraph, RequestPlan, RequestPlanStep

    plan = RequestPlan(
        request_id="phase10-command-authority",
        original_utterance="turn right twice",
        objective_type="motor",
        objective_summary="Execute a raw motor command.",
        expected_response="execute_motor",
        steps=[
            RequestPlanStep(
                step_id="execute_motor",
                layer="spine",
                operation="execute",
            )
        ],
    )
    graph = ReadinessGraph(
        request_id=plan.request_id,
        graph_status="executable",
        next_action="execute_motor",
        explanation="Unit probe executable motor command.",
    )
    return plan, graph


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        import jeenom.command_authority as command_authority
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        command_authority = None  # type: ignore[assignment]
        details["command_authority_import_error"] = f"{type(exc).__name__}: {exc}"

    command_authority_cls = (
        getattr(command_authority, "CommandAuthority", None)
        if command_authority is not None
        else None
    )
    metrics["command_authority_module_exists"] = command_authority is not None
    metrics["command_authority_class_exists"] = command_authority_cls is not None

    if command_authority is not None:
        source = ast_source("jeenom/command_authority.py")
        metrics["command_authority_has_no_operator_station_dependency"] = (
            "operator_station" not in source
        )
    else:
        metrics["command_authority_has_no_operator_station_dependency"] = False

    station_tree = ast.parse(ast_source("jeenom/operator_station.py"))
    record_calls = ast_function_call_names(station_tree, "_record_command_result")
    pending_calls = ast_function_call_names(station_tree, "_pending_clarification_trace")
    details["record_command_result_calls"] = sorted(set(record_calls))
    details["pending_clarification_trace_calls"] = sorted(set(pending_calls))

    metrics["record_result_delegates_to_command_authority"] = (
        "record_result" in record_calls
    )
    metrics["pending_clarification_trace_delegates_to_command_authority"] = (
        "pending_clarification_trace" in pending_calls
    )
    metrics["station_record_result_does_not_construct_trace_objects"] = not (
        TRACE_CONSTRUCTORS & set(record_calls)
    )
    metrics["station_pending_trace_does_not_construct_envelopes"] = (
        "CorticalEnvelope" not in pending_calls
    )

    if command_authority_cls is not None:
        session = make_session()
        metrics["operator_station_has_command_authority"] = isinstance(
            getattr(session, "command_authority", None),
            command_authority_cls,
        )
        try:
            from jeenom.schemas import CommandResult

            plan, graph = _sample_plan_and_graph()
            result = command_authority_cls(station_name="Phase10Probe").record_result(
                "turn right twice",
                "MOTOR COMPLETE\nsteps=2",
                intent=None,
                plan=plan,
                graph=graph,
                tickets=[],
                compiler_name="smoke",
            )
            metrics["command_authority_builds_typed_command_result"] = isinstance(
                result,
                CommandResult,
            )
            metrics["command_authority_preserves_plan_graph_ids"] = (
                result.envelope is not None
                and result.command is not None
                and result.envelope.request_plan is plan
                and result.envelope.readiness_graph is graph
                and result.command.request_id == plan.request_id
            )
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["command_authority_behavior_error"] = f"{type(exc).__name__}: {exc}"
            metrics["command_authority_builds_typed_command_result"] = False
            metrics["command_authority_preserves_plan_graph_ids"] = False
    else:
        metrics["operator_station_has_command_authority"] = False
        metrics["command_authority_builds_typed_command_result"] = False
        metrics["command_authority_preserves_plan_graph_ids"] = False

    metrics["phase10_command_authority_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase10_command_authority_holds")


if __name__ == "__main__":
    raise SystemExit(main())
