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


SUBSTRATE_FILES = [
    "jeenom/schemas.py",
    "jeenom/readiness_graph.py",
    "jeenom/command_authority.py",
    "jeenom/side_effect_authority.py",
    "jeenom/intent_verifier.py",
    "jeenom/cortex.py",
    "jeenom/cortex_session.py",
    "jeenom/sense.py",
    "jeenom/spine.py",
    "jeenom/memory.py",
    "jeenom/planning_semantics.py",
    "jeenom/primitive_library.py",
    "jeenom/primitive_synthesizer.py",
    "jeenom/capability_registry.py",
]

DOMAIN_MODULE_PREFIXES = ("minigrid",)  # external package or internal jeenom/minigrid_*.py


def _substrate_domain_violations(root) -> list[tuple[str, int, str]]:
    """Return (file, lineno, import_string) for every substrate→domain import."""
    violations = []
    for rel_path in SUBSTRATE_FILES:
        path = root / rel_path
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for prefix in DOMAIN_MODULE_PREFIXES:
                            if alias.name.startswith(prefix):
                                violations.append((rel_path, node.lineno, alias.name))
                else:  # ImportFrom
                    module = node.module or ""
                    # Relative import from a domain module within jeenom/
                    if node.level and any(
                        module.startswith(f"minigrid_{p}") or module == f"minigrid_{p[:-1]}"
                        or module.startswith(f"minigrid")
                        for p in ("",)
                    ):
                        for prefix in DOMAIN_MODULE_PREFIXES:
                            if module.startswith(prefix):
                                violations.append((rel_path, node.lineno, f".{module}"))
                    # Absolute external import
                    for prefix in DOMAIN_MODULE_PREFIXES:
                        if module.startswith(prefix):
                            violations.append((rel_path, node.lineno, module))
    return violations


def _substrate_door_literals(root) -> list[tuple[str, int, str]]:
    """Return (file, lineno, line) for hardcoded 'door' validator literals in substrate."""
    hits = []
    for rel_path in SUBSTRATE_FILES:
        path = root / rel_path
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if (
                '"door"' in stripped
                and any(op in stripped for op in ("!= ", "== ", 'enum": [', "= (\"door"))
            ):
                hits.append((rel_path, i, stripped))
    return hits


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

    # ── Op 1+2: import partition discipline ─────────────────────────────────
    from harness import ROOT as _ROOT
    import ast as _ast

    domain_violations = _substrate_domain_violations(_ROOT)
    metrics["substrate_files_have_no_domain_imports"] = not domain_violations
    details["substrate_domain_violations"] = [
        f"{f}:{ln} — {imp}" for f, ln, imp in domain_violations
    ]

    door_literals = _substrate_door_literals(_ROOT)
    metrics["substrate_files_have_no_hardcoded_door_literals"] = not door_literals
    details["substrate_door_literals"] = [
        f"{f}:{ln} — {src}" for f, ln, src in door_literals
    ]

    # Vocabulary gate: once registered, unregistered object_type is rejected
    from jeenom.schemas import (
        register_domain_vocabulary,
        clear_registered_vocabulary,
        _validate_object_type,
        SchemaValidationError,
    )
    try:
        clear_registered_vocabulary()
        register_domain_vocabulary(("door",))
        rejected = False
        try:
            _validate_object_type("key", "test")
        except SchemaValidationError:
            rejected = True
        clear_registered_vocabulary()
        metrics["vocabulary_gate_rejects_unregistered_type"] = rejected
    except Exception as exc:
        metrics["vocabulary_gate_rejects_unregistered_type"] = False
        details["vocabulary_gate_error"] = str(exc)

    metrics["phase9d_static_architecture_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase9d_static_architecture_holds")


if __name__ == "__main__":
    raise SystemExit(main())
