"""Phase 9D probe: static structural schema enforcement.

This intentionally fails on the current loose-router implementation. It checks
for the architectural shape Phase 9D is supposed to enforce before runtime
behavior is changed.
"""
from __future__ import annotations

import ast
import inspect
from typing import Any

from harness import ROOT, ast_call_names, ast_function_call_names, ast_source, emit_result


REQUIRED_SCHEMA_TYPES = [
    "CorticalEnvelope",
    "ApprovedCommand",
    "ExecutionTicket",
    "MemoryWriteTicket",
    "RawMotorTicket",
    "MissionExecutionPlan",
    "CommandResult",
]


def main() -> int:
    import jeenom.schemas as schemas
    from jeenom import operator_station

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    missing = [name for name in REQUIRED_SCHEMA_TYPES if not hasattr(schemas, name)]
    metrics["required_phase9d_schema_types_exist"] = not missing
    details["missing_schema_types"] = missing

    metrics["loose_operator_command_removed"] = not hasattr(operator_station, "OperatorCommand")

    run_task_sig = inspect.signature(operator_station.OperatorStationSession.run_task)
    run_task_params = list(run_task_sig.parameters.values())
    second = run_task_params[1] if len(run_task_params) > 1 else None
    metrics["run_task_no_longer_accepts_instruction_string"] = not (
        second is not None
        and second.name == "instruction"
        and second.annotation in {str, "str"}
    )
    details["run_task_signature"] = str(run_task_sig)

    apply_sig = inspect.signature(operator_station.OperatorStationSession.apply_knowledge_update)
    metrics["memory_update_api_no_longer_accepts_payload_dict"] = "payload" not in apply_sig.parameters
    details["apply_knowledge_update_signature"] = str(apply_sig)

    station_tree = ast.parse(ast_source("jeenom/operator_station.py"))
    direct_run_task_calls = [
        (node.lineno, ast.unparse(node.func))
        for node in ast.walk(station_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_task"
    ]
    metrics["station_has_no_direct_run_task_calls"] = not direct_run_task_calls
    details["direct_run_task_calls"] = direct_run_task_calls

    resume_calls = (
        ast_function_call_names(station_tree, "resume_pending_clarification")
        + ast_function_call_names(station_tree, "resume_candidate_clarification")
    )
    metrics["clarification_resume_has_no_direct_execution_or_memory_write"] = not (
        "run_task" in resume_calls or "apply_knowledge_update" in resume_calls
    )
    details["clarification_resume_calls"] = sorted(set(resume_calls))

    llm_prompt = ast_source("jeenom/llm_compiler.py")
    schema_text = ast_source("jeenom/schemas.py")
    metrics["motor_language_no_longer_describes_planning_bypass"] = (
        "bypass all task planning" not in llm_prompt
        and "bypasses task planning" not in schema_text
    )

    metrics["phase9d_static_architecture_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase9d_static_architecture_holds")


if __name__ == "__main__":
    raise SystemExit(main())
