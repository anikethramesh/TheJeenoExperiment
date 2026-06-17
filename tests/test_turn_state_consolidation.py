"""Phase 13A.2.4 — typed per-turn state consolidation.

The per-turn trace (19 `last_*` / `current_*` / `active_steering_directive` fields) is one
concept and must live on a typed `TurnState`, not as loose session attributes. The session
surfaces each field as a delegating property so the public `session.last_*` read API the
eval suite depends on is preserved, while initialization is now guaranteed (the bug class
behind the `active_steering_directive` `AttributeError`).

Red-bar first: imports `jeenom.turn_state.TurnState` and asserts the session routes every
per-turn field through it.
"""
from __future__ import annotations

import ast
from pathlib import Path

from jeenom.operator_station import OperatorStationSession
from jeenom.turn_state import TURN_STATE_FIELDS, TurnState

STATION = Path(__file__).resolve().parents[1] / "jeenom" / "operator_station.py"

EXPECTED_FIELDS = {
    "last_result", "active_steering_directive", "last_mission_execution_plan",
    "last_execution_ticket", "last_memory_write_ticket", "last_raw_motor_ticket",
    "last_sense_ticket", "last_operator_intent", "last_cortical_envelope",
    "last_approved_command", "last_command_result", "current_environment_identity",
    "last_environment_invalidation_reason", "last_request_plan", "last_readiness_graph",
    "last_plan_reuse_verdict", "last_arbitration_trace", "last_operational_mismatches",
    "last_repair_events",
}


def _session() -> OperatorStationSession:
    return OperatorStationSession(compiler_name="smoke_test", render_mode="none")


def test_turn_state_holds_the_full_per_turn_trace():
    assert set(TURN_STATE_FIELDS) == EXPECTED_FIELDS
    fresh = TurnState()
    # all default to empty so reads never raise before the first turn
    for name in EXPECTED_FIELDS:
        value = getattr(fresh, name)
        assert value is None or value == []


def test_session_surfaces_each_field_as_a_delegating_property():
    for name in EXPECTED_FIELDS:
        attr = vars(OperatorStationSession).get(name)
        assert isinstance(attr, property), f"{name} is not a delegating property on the session"


def test_session_reads_and_delegates_without_attribute_errors():
    session = _session()
    try:
        # initialization guaranteed: every field readable on a fresh session
        for name in EXPECTED_FIELDS:
            getattr(session, name)  # must not raise
        # delegation both ways
        session.turn_state.last_request_plan = "SENTINEL"
        assert session.last_request_plan == "SENTINEL"
        session.last_readiness_graph = "GRAPH"
        assert session.turn_state.last_readiness_graph == "GRAPH"
    finally:
        session.close()


def test_init_does_not_bare_assign_per_turn_fields():
    """Anti-drift: __init__ builds a TurnState and stops hand-initializing the 19 fields."""
    tree = ast.parse(STATION.read_text(encoding="utf-8"), filename="operator_station.py")
    init = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    bare = []
    for node in ast.walk(init):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr in EXPECTED_FIELDS
                ):
                    bare.append((target.attr, node.lineno))
    assert not bare, f"__init__ still bare-initializes per-turn fields (use TurnState): {bare}"
    assert "self.turn_state" in STATION.read_text(encoding="utf-8")
