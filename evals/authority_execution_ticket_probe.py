"""Phase 9D probe: task execution requires an ExecutionTicket."""
from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import patch

from harness import emit_result, make_session


def _fake_episode(**kwargs: Any) -> dict[str, Any]:
    return {
        "final_state": {"task_complete": True},
        "runtime_llm_calls_during_render": 0,
        "cache_miss_during_render": 0,
        "task": {"instruction": kwargs.get("instruction"), "task_type": "probe"},
        "loop_records": [],
        "_render_adapter": None,
    }


def main() -> int:
    import jeenom.schemas as schemas
    from jeenom.operator_station import OperatorStationSession

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    ticket_cls = getattr(schemas, "ExecutionTicket", None)
    metrics["execution_ticket_schema_exists"] = ticket_cls is not None

    sig = inspect.signature(OperatorStationSession.run_task)
    details["run_task_signature"] = str(sig)
    params = list(sig.parameters.values())
    second = params[1] if len(params) > 1 else None
    metrics["run_task_signature_requires_ticket"] = (
        second is not None
        and second.name in {"ticket", "execution_ticket"}
        and ticket_cls is not None
    )

    session = make_session()
    with patch("jeenom.run_demo.run_episode", side_effect=_fake_episode) as fake:
        try:
            result = session.run_task("go to the red door")
        except (TypeError, ValueError, RuntimeError) as exc:
            metrics["raw_string_task_execution_is_rejected"] = True
            details["raw_string_rejection"] = type(exc).__name__
        else:
            metrics["raw_string_task_execution_is_rejected"] = False
            details["raw_string_result_type"] = type(result).__name__
            details["raw_string_run_episode_called"] = fake.called

    active_plan = getattr(session, "last_request_plan", None)
    active_graph = getattr(session, "last_readiness_graph", None)
    metrics["execution_requires_active_plan_and_graph"] = active_plan is not None and active_graph is not None

    metrics["execution_ticket_gate_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="execution_ticket_gate_holds")


if __name__ == "__main__":
    raise SystemExit(main())
