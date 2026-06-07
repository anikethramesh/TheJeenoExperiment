"""Phase 9E probe: block boundaries are enforceable.

This probe checks for the first Phase 9E gate:
canonical blocks must stop using scattered memory pockets as authority.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from harness import ROOT, emit_result, make_session


ALLOWED_MEMORY_MUTATION_FILES = {
    "jeenom/memory.py",
    "jeenom/representation.py",
}


def _py_files() -> list[Path]:
    return sorted((ROOT / "jeenom").glob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _is_memory_attr(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "memory"
    )


def _external_memory_stores() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in _py_files():
        rel = _rel(path)
        if rel in ALLOWED_MEMORY_MUTATION_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets: list[ast.AST]
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                else:
                    targets = [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript) and (
                        _is_memory_attr(target.value, "knowledge")
                        or _is_memory_attr(target.value, "episodic_memory")
                    ):
                        hits.append((rel, target.lineno, ast.unparse(target)))
                    if _is_memory_attr(target, "scene_model"):
                        hits.append((rel, target.lineno, ast.unparse(target)))
    return hits


def _external_update_knowledge_calls() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in _py_files():
        rel = _rel(path)
        if rel in ALLOWED_MEMORY_MUTATION_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update_knowledge"
            ):
                hits.append((rel, node.lineno, ast.unparse(node.func)))
    return hits


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        from jeenom.representation import RepresentationStore
    except Exception as exc:  # pragma: no cover - probe detail
        RepresentationStore = None  # type: ignore[assignment]
        details["representation_import_error"] = f"{type(exc).__name__}: {exc}"

    metrics["representation_store_exists"] = RepresentationStore is not None

    if RepresentationStore is not None:
        session = make_session()
        metrics["operator_station_has_representation_store"] = isinstance(
            getattr(session, "representation", None),
            RepresentationStore,
        )
        metrics["active_claims_is_property"] = isinstance(
            getattr(type(session), "active_claims", None),
            property,
        )
    else:
        metrics["operator_station_has_representation_store"] = False
        metrics["active_claims_is_property"] = False

    memory_stores = _external_memory_stores()
    update_knowledge_calls = _external_update_knowledge_calls()
    metrics["no_external_memory_dict_or_scene_stores"] = not memory_stores
    metrics["no_external_durable_update_knowledge_calls"] = not update_knowledge_calls
    details["external_memory_stores"] = memory_stores
    details["external_update_knowledge_calls"] = update_knowledge_calls

    metrics["phase9e_block_boundary_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase9e_block_boundary_holds")


if __name__ == "__main__":
    raise SystemExit(main())
