"""Phase 9D probe: station turns must produce typed command results.

The current station returns strings and keeps loose command routing. Phase 9D
requires a result/envelope trace for every user-visible command path.
"""
from __future__ import annotations

from typing import Any

from harness import emit_result, first_line, make_session, patched_env_builder


REQUIRED_TRACE_ATTRS = [
    "last_cortical_envelope",
    "last_command_result",
]


def _has_trace_attrs(session: Any) -> bool:
    return all(hasattr(session, attr) and getattr(session, attr) is not None for attr in REQUIRED_TRACE_ATTRS)


def main() -> int:
    import jeenom.schemas as schemas

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    command_result_cls = getattr(schemas, "CommandResult", None)
    metrics["command_result_schema_exists"] = command_result_cls is not None

    cases = {
        "status": "status",
        "scene_query": "what do you see",
        "knowledge_update": "the red door is the delivery target",
        "task": "go to the red door",
        "raw_motor": "turn right",
    }

    with patched_env_builder():
        for label, utterance in cases.items():
            session = make_session()
            response = session.handle_utterance(utterance)
            details[f"{label}_response"] = first_line(response)
            details[f"{label}_response_type"] = type(response).__name__
            metrics[f"{label}_returns_command_result"] = (
                command_result_cls is not None and isinstance(response, command_result_cls)
            )
            metrics[f"{label}_records_envelope_and_result_trace"] = _has_trace_attrs(session)

    metrics["all_user_visible_paths_are_schema_enforced"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="all_user_visible_paths_are_schema_enforced")


if __name__ == "__main__":
    raise SystemExit(main())
