"""Phase 10F probe: top-level turn routing is delegated out of the station."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from harness import emit_result, make_session


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        from jeenom.schemas import ApprovedCommand
        from jeenom.turn_orchestrator import TurnOrchestrator
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        ApprovedCommand = None  # type: ignore[assignment]
        TurnOrchestrator = None  # type: ignore[assignment]
        details["turn_orchestrator_import_error"] = f"{type(exc).__name__}: {exc}"

    metrics["turn_orchestrator_module_exists"] = TurnOrchestrator is not None

    try:
        session = make_session()
        orchestrator = getattr(session, "turn_orchestrator", None)
        metrics["station_has_turn_orchestrator"] = (
            TurnOrchestrator is not None
            and isinstance(orchestrator, TurnOrchestrator)
        )
        if ApprovedCommand is not None:
            message = session.execute_command(
                ApprovedCommand(
                    kind="clarification",
                    utterance="unit",
                    payload={"message": "orchestrated"},
                )
            )
            metrics["execute_command_compatibility_preserved"] = message == "orchestrated"
        else:
            metrics["execute_command_compatibility_preserved"] = False
        result = session.handle_utterance("help")
        metrics["handle_utterance_still_records_result"] = (
            isinstance(result.message, str)
            and bool(result.message)
            and session.last_command_result is result
        )
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        details["station_turn_orchestrator_error"] = f"{type(exc).__name__}: {exc}"
        metrics["station_has_turn_orchestrator"] = False
        metrics["execute_command_compatibility_preserved"] = False
        metrics["handle_utterance_still_records_result"] = False

    station_source = (ROOT / "jeenom" / "operator_station.py").read_text()
    orchestrator_source = (
        (ROOT / "jeenom" / "turn_orchestrator.py").read_text()
        if (ROOT / "jeenom" / "turn_orchestrator.py").exists()
        else ""
    )
    station_forbidden = [
        "def _handle_utterance_text(",
        "def handle_pending_clarification(",
    ]
    remaining = [needle for needle in station_forbidden if needle in station_source]
    details["remaining_station_turn_methods"] = remaining
    metrics["station_top_level_turn_methods_extracted"] = not remaining
    metrics["orchestrator_owns_turn_methods"] = all(
        needle in orchestrator_source
        for needle in (
            "def handle_utterance_text(",
            "def execute_command(",
            "def handle_pending_clarification(",
        )
    )

    # ── Op 3a: god-method extraction invariants ──────────────────────────────
    import ast as _ast
    from jeenom import operator_station as _os_mod

    # 3a-1: command_from_operator_intent must not exist on OperatorStationSession
    metrics["op3a_command_from_operator_intent_removed"] = not hasattr(
        _os_mod.OperatorStationSession, "command_from_operator_intent"
    )

    # 3a-2: station AST has no direct calls to build_request_plan / evaluate_request_plan
    station_tree = _ast.parse(station_source)
    planner_calls = [
        node.lineno
        for node in _ast.walk(station_tree)
        if isinstance(node, _ast.Call)
        and isinstance(node.func, _ast.Attribute)
        and node.func.attr in {"build_request_plan", "evaluate_request_plan"}
    ]
    metrics["op3a_station_has_no_direct_planner_calls"] = not planner_calls
    details["op3a_station_planner_calls"] = planner_calls

    # 3a-3: TurnOrchestrator has a dispatch method
    from jeenom.turn_orchestrator import TurnOrchestrator as _TO
    metrics["op3a_turn_orchestrator_has_dispatch"] = callable(
        getattr(_TO, "dispatch", None)
    )

    # 3a-4: TurnOrchestrator holds cortex_session (dataclass field check)
    import dataclasses as _dc
    to_field_names = {f.name for f in _dc.fields(_TO)} if _dc.is_dataclass(_TO) else set()
    metrics["op3a_turn_orchestrator_holds_cortex_session"] = (
        "cortex_session" in to_field_names
    )
    details["op3a_turn_orchestrator_fields"] = sorted(to_field_names)

    metrics["phase10_turn_orchestrator_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase10_turn_orchestrator_holds")


if __name__ == "__main__":
    raise SystemExit(main())
