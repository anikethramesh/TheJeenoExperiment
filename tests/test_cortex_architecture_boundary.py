"""Architecture invariant: operator_station must not call the planning pipeline directly.

The cortex boundary requires that build_request_plan and evaluate_request_plan are
only ever called through CortexSession.  The station dispatches; it does not plan.

If either of these tests starts failing, someone has bypassed CortexSession and
re-introduced a planning call directly into the station.  Fix the regression by
routing through self.cortex_session.plan() or self.cortex_session.evaluate().
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_STATION_PATH = Path(__file__).parent.parent / "jeenom" / "operator_station.py"
_PLANNING_NAMES = {"build_request_plan", "evaluate_request_plan"}


def _load_source() -> str:
    return _STATION_PATH.read_text(encoding="utf-8")


def _call_names_in_source(source: str) -> set[str]:
    """Return the set of function names called directly in the source tree."""
    tree = ast.parse(source)
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    return called


def _import_names_in_source(source: str) -> set[str]:
    """Return the set of names imported at module level."""
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    return imported


class TestCortexArchitectureBoundary:
    def test_station_does_not_import_planning_functions(self):
        """operator_station.py must not import build_request_plan or evaluate_request_plan."""
        source = _load_source()
        imported = _import_names_in_source(source)
        violations = _PLANNING_NAMES & imported
        assert not violations, (
            f"operator_station.py imports planning functions directly: {violations}. "
            "Route through CortexSession instead."
        )

    def test_station_does_not_call_planning_functions_by_name(self):
        """operator_station.py must not call build_request_plan or evaluate_request_plan directly."""
        source = _load_source()
        calls = _call_names_in_source(source)
        violations = _PLANNING_NAMES & calls
        assert not violations, (
            f"operator_station.py calls planning functions directly: {violations}. "
            "Route through self.cortex_session.plan() or self.cortex_session.evaluate()."
        )

    def test_cortex_session_exists_and_is_importable(self):
        """CortexSession must be importable from jeenom.cortex_session."""
        from jeenom.cortex_session import CortexSession
        assert hasattr(CortexSession, "plan")
        assert hasattr(CortexSession, "evaluate")

    def test_station_exposes_cortex_session(self):
        """OperatorStationSession must expose .cortex_session as a CortexSession instance."""
        from evals.harness import make_session
        from jeenom.cortex_session import CortexSession

        session = make_session()
        assert hasattr(session, "cortex_session")
        assert isinstance(session.cortex_session, CortexSession)

    def test_pending_state_owned_by_turn_orchestrator(self):
        """Pending state must live on TurnOrchestrator, not as direct station fields."""
        from evals.harness import make_session
        from jeenom.turn_orchestrator import (
            PendingClarification,
            PendingPrimitiveDefinition,
            PendingSynthesisProposal,
            TurnOrchestrator,
        )

        session = make_session()
        to = session.turn_orchestrator
        assert isinstance(to, TurnOrchestrator)
        assert hasattr(to, "pending_clarification")
        assert hasattr(to, "pending_synthesis_proposal")
        assert hasattr(to, "pending_primitive_definition")
        # Station exposes them via properties (not direct attributes)
        assert "pending_clarification" not in session.__dict__
        assert "pending_synthesis_proposal" not in session.__dict__
        assert "pending_primitive_definition" not in session.__dict__

    def test_pending_dataclasses_not_defined_in_station(self):
        """PendingClarification, PendingSynthesisProposal, PendingPrimitiveDefinition
        must be defined in turn_orchestrator, not operator_station."""
        import inspect
        import jeenom.operator_station as station_module
        import jeenom.turn_orchestrator as orch_module

        for name in ("PendingClarification", "PendingSynthesisProposal", "PendingPrimitiveDefinition"):
            # Must exist in turn_orchestrator
            assert hasattr(orch_module, name), f"{name} missing from turn_orchestrator"
            # Must be defined there (not just re-exported from station)
            cls = getattr(orch_module, name)
            assert inspect.getmodule(cls) is orch_module, (
                f"{name} is defined in {inspect.getmodule(cls).__name__}, not turn_orchestrator"
            )
            # Station may re-import them but must not define them
            if hasattr(station_module, name):
                station_cls = getattr(station_module, name)
                assert inspect.getmodule(station_cls) is orch_module, (
                    f"{name} is defined in station module — must come from turn_orchestrator"
                )
