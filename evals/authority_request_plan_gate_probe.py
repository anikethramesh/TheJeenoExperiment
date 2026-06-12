"""Phase 9 probe: executable task paths must be readiness-gated.

Raw motor commands may be direct low-level controls. Task-like station paths
must record a meaningful RequestPlan and ReadinessGraph before execution or
refusal.
"""
from __future__ import annotations

from harness import emit_result, first_line, has_meaningful_plan, make_session, patched_env_builder


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, str | None] = {}

    with patched_env_builder():
        task = make_session()
        task_response = task.handle_utterance("go to the red door")
        metrics["task_path_records_request_plan_and_graph"] = has_meaningful_plan(task)
        metrics["task_path_still_completes"] = "RUN COMPLETE" in task_response
        details["task_response"] = first_line(task_response)
        details["task_graph_status"] = (
            task.last_readiness_graph.graph_status
            if task.last_readiness_graph is not None
            else None
        )

        unsupported = make_session()
        unsupported_response = unsupported.handle_utterance("pick up the red key")
        metrics["unsupported_task_records_blocking_plan"] = has_meaningful_plan(
            unsupported
        ) and unsupported.last_readiness_graph.graph_status in {
            "missing_skills",
            "unsupported",
            "blocked_by_dependency",
        }
        metrics["unsupported_task_does_not_execute"] = (
            "RUN COMPLETE" not in unsupported_response
            and "MOTOR COMPLETE" not in unsupported_response
        )
        details["unsupported_response"] = first_line(unsupported_response)
        details["unsupported_graph_status"] = (
            unsupported.last_readiness_graph.graph_status
            if unsupported.last_readiness_graph is not None
            else None
        )

        motor = make_session()
        motor_response = motor.handle_utterance("turn right twice")
        metrics["explicit_raw_motor_command_is_allowed"] = "MOTOR COMPLETE" in motor_response
        details["motor_response"] = first_line(motor_response)

    metrics["request_plan_gate_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="request_plan_gate_holds")


if __name__ == "__main__":
    raise SystemExit(main())
