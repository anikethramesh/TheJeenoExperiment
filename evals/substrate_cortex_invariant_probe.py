"""Op 5 probe: substrate cortex invariant.

Verifies that the legacy Readiness class has been deleted and that MissionCortex
remains one-directional (no circular dependency back into Cortex/CortexSession).

Metrics:
  op5_readiness_class_deleted          — from jeenom.cortex import Readiness → ImportError
  op5_cortex_no_readiness_attr         — Cortex instance has no .readiness attribute
  op5_readiness_not_in_public_api      — "Readiness" absent from jeenom.__all__
  op5_onboard_no_readiness_check       — Cortex.onboard_task body has no self.readiness.check call (AST)
  op5_mission_cortex_one_directional   — mission_cortex.py does not import Cortex, CortexSession, or TurnOrchestrator
"""
from __future__ import annotations

import ast
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JEENOM = ROOT / "jeenom"


def _ast_has_attribute_call(source: str, obj_attr: str, method: str) -> bool:
    """Return True if source contains a call like self.<obj_attr>.<method>(...)."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != method:
            continue
        value = func.value
        if isinstance(value, ast.Attribute) and value.attr == obj_attr:
            return True
    return False


def main() -> int:
    metrics: dict[str, bool] = {}

    # 1. Readiness class should no longer be importable from jeenom.cortex
    try:
        from jeenom.cortex import Readiness  # noqa: F401
        metrics["op5_readiness_class_deleted"] = False
    except ImportError:
        metrics["op5_readiness_class_deleted"] = True

    # 2. Cortex instance should have no .readiness attribute
    try:
        from jeenom.cortex import Cortex
        from unittest.mock import MagicMock
        mock_memory = MagicMock()
        mock_compiler = MagicMock()
        cortex = Cortex(mock_memory, mock_compiler)
        metrics["op5_cortex_no_readiness_attr"] = not hasattr(cortex, "readiness")
    except Exception:
        metrics["op5_cortex_no_readiness_attr"] = False

    # 3. Readiness should not appear in jeenom's public API
    try:
        import jeenom
        metrics["op5_readiness_not_in_public_api"] = "Readiness" not in getattr(jeenom, "__all__", [])
    except Exception:
        metrics["op5_readiness_not_in_public_api"] = False

    # 4. Cortex.onboard_task should not call self.readiness.check (AST check)
    cortex_src = (JEENOM / "cortex.py").read_text()
    metrics["op5_onboard_no_readiness_check"] = not _ast_has_attribute_call(
        cortex_src, "readiness", "check"
    )

    # 5. MissionCortex must be one-directional — no import of Cortex, CortexSession,
    #    or TurnOrchestrator from mission_cortex.py
    mc_src = (JEENOM / "mission_cortex.py").read_text()
    tree = ast.parse(mc_src)
    forbidden_names = {"Cortex", "CortexSession", "TurnOrchestrator"}
    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name.split(".")[0])
    metrics["op5_mission_cortex_one_directional"] = not bool(
        forbidden_names & imported_names
    )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
