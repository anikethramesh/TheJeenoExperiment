"""Phase 9D probe: clarification resume must re-enter readiness."""
from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass
from typing import Any

from harness import ROOT, emit_result


def _function_calls(tree: ast.AST, function_name: str) -> list[tuple[int, str]]:
    calls: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute):
                        calls.append((child.lineno, child.func.attr))
                    elif isinstance(child.func, ast.Name):
                        calls.append((child.lineno, child.func.id))
    return calls


def main() -> int:
    from jeenom.operator_station import PendingClarification

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    pending_fields = [field.name for field in fields(PendingClarification)] if is_dataclass(PendingClarification) else []
    details["pending_clarification_fields"] = pending_fields
    metrics["pending_clarification_stores_plan_or_envelope"] = any(
        name in pending_fields
        for name in {
            "request_plan",
            "readiness_graph",
            "cortical_envelope",
            "pending_envelope",
        }
    )

    tree = ast.parse((ROOT / "jeenom/operator_station.py").read_text(encoding="utf-8"))
    pending_calls = _function_calls(tree, "resume_pending_clarification")
    candidate_calls = _function_calls(tree, "resume_candidate_clarification")
    all_calls = pending_calls + candidate_calls
    details["resume_calls"] = all_calls
    direct_unsafe_calls = [
        item for item in all_calls if item[1] in {"run_task", "apply_knowledge_update"}
    ]
    metrics["clarification_resume_has_no_direct_execution_or_memory_write"] = not direct_unsafe_calls
    details["direct_unsafe_resume_calls"] = direct_unsafe_calls

    metrics["clarification_resume_reevaluates_readiness"] = any(
        item[1] == "evaluate_request_plan" for item in all_calls
    )

    metrics["clarification_resume_gate_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="clarification_resume_gate_holds")


if __name__ == "__main__":
    raise SystemExit(main())
