"""Phase 9D probe: durable memory writes require typed authority."""
from __future__ import annotations

from typing import Any

from harness import emit_result, make_session


class RaisingCompiler:
    def compile_operator_intent(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("phase9d forced planning failure")


def main() -> int:
    import jeenom.schemas as schemas

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    ticket_cls = getattr(schemas, "MemoryWriteTicket", None)
    metrics["memory_write_ticket_schema_exists"] = ticket_cls is not None

    session = make_session()
    before = dict(session.memory.knowledge)
    try:
        response = session.apply_knowledge_update(
            {
                "target_color": "red",
                "target_type": "door",
                "delivery_target": {"color": "red", "object_type": "door"},
            }
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        metrics["raw_payload_memory_write_is_rejected"] = True
        details["raw_payload_rejection"] = type(exc).__name__
    else:
        after = dict(session.memory.knowledge)
        metrics["raw_payload_memory_write_is_rejected"] = after == before
        details["raw_payload_response"] = response.splitlines()[0] if response else ""
        details["knowledge_after_raw_payload"] = after

    teaching = make_session()
    teaching.compiler = RaisingCompiler()
    teach_response = teaching.teach_concept("badplan", "go to the red door")
    taught = teaching.knowledge_base.recall("badplan") is not None
    metrics["concept_teach_does_not_swallow_planning_failure"] = not taught
    details["concept_teach_response"] = teach_response.splitlines()[0] if teach_response else ""
    details["concept_written_after_planning_failure"] = taught

    metrics["memory_write_gate_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="memory_write_gate_holds")


if __name__ == "__main__":
    raise SystemExit(main())
