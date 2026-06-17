"""Phase 13A.2.2 — capability-handle grammar consolidation.

The substrate-neutral control plane must build capability handles through the one
home that already exists (`PlanningSemantics.capability_handle`, fed by the context's
`capability_handles` templates), not by hand-rolling `f"grounding.all_doors.ranked.
{metric}.agent"` strings. This proves the home reproduces the exact legacy strings
(byte-identical, so cache keys / capability lookups don't shift) and red-bars on any
handle f-string left in `operator_station.py`.

Scope: `llm_compiler.py` (the MiniGrid compiler profile, no PlanningSemantics handle)
and its prompt examples stay with the Phase 14 door-leak work — out of scope here.
"""
from __future__ import annotations

import ast
from pathlib import Path

from jeenom.minigrid_operational_context import MiniGridOperationalContext
from jeenom.planning_semantics import PlanningSemantics

STATION = Path(__file__).resolve().parents[1] / "jeenom" / "operator_station.py"
HANDLE_PREFIXES = ("grounding.", "claims.", "task.", "sensing.", "action.")


def _semantics() -> PlanningSemantics:
    return PlanningSemantics(MiniGridOperationalContext.default())


def test_capability_handle_reproduces_legacy_handles_byte_for_byte():
    ps = _semantics()
    assert ps.capability_handle("ranked", metric="manhattan") == "grounding.all_doors.ranked.manhattan.agent"
    assert ps.capability_handle("ranked", metric="euclidean") == "grounding.all_doors.ranked.euclidean.agent"
    assert ps.capability_handle("filter_threshold", metric="euclidean") == "claims.filter.threshold.euclidean"
    assert ps.capability_handle("closest", metric="manhattan") == "grounding.closest_door.manhattan.agent"


def test_capability_handle_does_not_gate_on_supported_metric():
    # Unlike ranked_handle, the composer must still produce a handle for a synthesized
    # metric the registry has not blessed yet (Phase 10I user-defined metrics).
    ps = _semantics()
    assert ps.capability_handle("ranked", metric="convenient_distance") == (
        "grounding.all_doors.ranked.convenient_distance.agent"
    )


def test_station_builds_no_handle_fstrings():
    """Anti-drift: every parameterized handle in the station goes through the composer."""
    tree = ast.parse(STATION.read_text(encoding="utf-8"), filename="operator_station.py")
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        head = node.values[0] if node.values else None
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            if head.value.startswith(HANDLE_PREFIXES):
                offenders.append(node.lineno)
    assert not offenders, f"hand-built handle f-strings remain in operator_station: {offenders}"
