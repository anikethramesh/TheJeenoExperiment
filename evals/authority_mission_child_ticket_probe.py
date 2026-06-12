"""Phase 9D probe: missions execute child tickets, not raw task strings."""
from __future__ import annotations

import ast
from typing import Any

from harness import ROOT, emit_result


def _mission_calls() -> list[tuple[int, str]]:
    tree = ast.parse((ROOT / "jeenom/operator_station.py").read_text(encoding="utf-8"))
    calls: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_mission":
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute):
                        calls.append((child.lineno, child.func.attr))
                    elif isinstance(child.func, ast.Name):
                        calls.append((child.lineno, child.func.id))
    return calls


def main() -> int:
    import jeenom.schemas as schemas

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    metrics["mission_execution_plan_schema_exists"] = hasattr(schemas, "MissionExecutionPlan")

    calls = _mission_calls()
    details["mission_calls"] = calls
    metrics["mission_does_not_call_run_task_directly"] = not any(name == "run_task" for _, name in calls)
    metrics["mission_builds_or_uses_child_tickets"] = any(
        name in {"build_execution_ticket", "execute_ticket", "_run_task_with_ticket"}
        for _, name in calls
    )

    source = (ROOT / "jeenom/operator_station.py").read_text(encoding="utf-8")
    metrics["mission_does_not_loop_over_raw_task_strings_as_authority"] = (
        "for step_idx, step_utterance in enumerate(steps)" not in source
        or "self.run_task(step_utterance)" not in source
    )

    metrics["mission_child_ticket_gate_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="mission_child_ticket_gate_holds")


if __name__ == "__main__":
    raise SystemExit(main())
