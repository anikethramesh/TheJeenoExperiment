"""Phase 9D probe: raw motor execution is typed and explicitly authorized."""
from __future__ import annotations

from typing import Any

from harness import emit_result, first_line, is_motor_execution, make_session, patched_env_builder


def main() -> int:
    import jeenom.schemas as schemas

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    ticket_cls = getattr(schemas, "RawMotorTicket", None)
    metrics["raw_motor_ticket_schema_exists"] = ticket_cls is not None

    with patched_env_builder():
        explicit = make_session()
        explicit_response = explicit.handle_utterance("turn right twice")
        metrics["explicit_low_level_motor_still_executes"] = is_motor_execution(explicit_response)
        details["explicit_motor_response"] = first_line(explicit_response)

        raw_ticket = getattr(explicit, "last_raw_motor_ticket", None)
        metrics["motor_execution_records_raw_motor_ticket"] = ticket_cls is not None and isinstance(raw_ticket, ticket_cls)
        metrics["motor_execution_has_plan_and_graph"] = (
            explicit.last_request_plan is not None
            and explicit.last_readiness_graph is not None
            and explicit.last_readiness_graph.graph_status == "executable"
        )

        task_like = make_session()
        task_like_response = task_like.handle_utterance("pick up the red key")
        metrics["task_like_request_does_not_motor_execute"] = not is_motor_execution(task_like_response)
        metrics["task_like_request_gets_no_raw_motor_ticket"] = getattr(task_like, "last_raw_motor_ticket", None) is None
        details["task_like_response"] = first_line(task_like_response)

    metrics["raw_motor_ticket_gate_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="raw_motor_ticket_gate_holds")


if __name__ == "__main__":
    raise SystemExit(main())
