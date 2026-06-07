"""Phase 10C probe: MiniGrid HOW starts leaving OperatorStationSession."""
from __future__ import annotations

import ast
from typing import Any

from harness import ROOT, emit_result, make_session


FORBIDDEN_STATION_CALLS = {
    "MiniGridAdapter",
    "build_env",
    "run_episode",
    "run_motor_sequence",
    "prewarm_jit_cache",
}
FORBIDDEN_STATION_IMPORT_NAMES = {
    "MiniGridAdapter",
    "MiniGridSense",
    "MiniGridSpine",
    "ensure_custom_minigrid_envs_registered",
}


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def _call_hits(tree: ast.AST, names: set[str]) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called: str | None = None
        if isinstance(node.func, ast.Name):
            called = node.func.id
        elif isinstance(node.func, ast.Attribute):
            called = node.func.attr
        if called in names:
            hits.append((node.lineno, called))
    return hits


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        import jeenom.substrate_adapter as substrate_adapter
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        substrate_adapter = None  # type: ignore[assignment]
        details["substrate_adapter_import_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import jeenom.minigrid_substrate_adapter as minigrid_substrate_adapter
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        minigrid_substrate_adapter = None  # type: ignore[assignment]
        details["minigrid_substrate_adapter_import_error"] = f"{type(exc).__name__}: {exc}"

    substrate_cls = (
        getattr(substrate_adapter, "SubstrateAdapter", None)
        if substrate_adapter is not None
        else None
    )
    minigrid_cls = (
        getattr(minigrid_substrate_adapter, "MiniGridSubstrateAdapter", None)
        if minigrid_substrate_adapter is not None
        else None
    )
    metrics["substrate_adapter_module_exists"] = substrate_adapter is not None
    metrics["substrate_adapter_type_exists"] = substrate_cls is not None
    metrics["minigrid_substrate_adapter_module_exists"] = minigrid_substrate_adapter is not None
    metrics["minigrid_substrate_adapter_type_exists"] = minigrid_cls is not None

    station_tree = ast.parse(_source("jeenom/operator_station.py"))
    imports = _imports(station_tree)
    call_hits = _call_hits(station_tree, FORBIDDEN_STATION_CALLS)
    details["forbidden_station_imports"] = sorted(imports & FORBIDDEN_STATION_IMPORT_NAMES)
    details["forbidden_station_calls"] = call_hits
    metrics["station_has_no_direct_minigrid_how_imports"] = not (
        imports & FORBIDDEN_STATION_IMPORT_NAMES
    )
    metrics["station_has_no_direct_minigrid_runtime_calls"] = not call_hits

    if minigrid_cls is not None:
        session = make_session()
        metrics["operator_station_has_substrate_adapter"] = isinstance(
            getattr(session, "substrate", None),
            minigrid_cls,
        )
        try:
            action_names = session.substrate.known_action_names()
            metrics["substrate_exposes_motor_manifest"] = "turn_right" in action_names
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["known_action_names_error"] = f"{type(exc).__name__}: {exc}"
            metrics["substrate_exposes_motor_manifest"] = False
    else:
        metrics["operator_station_has_substrate_adapter"] = False
        metrics["substrate_exposes_motor_manifest"] = False

    metrics["phase10_substrate_adapter_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase10_substrate_adapter_holds")


if __name__ == "__main__":
    raise SystemExit(main())
