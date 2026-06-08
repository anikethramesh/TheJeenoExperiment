"""Shared helpers for JEENOM eval probes."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.llm_compiler import CompilerBackend, SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession


def build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def patched_env_builder():
    return patch("jeenom.run_demo.build_env", side_effect=build_env)


def make_session(
    *,
    memory_root: Path | None = None,
    env_id: str = "MiniGrid-GoToDoor-8x8-v0",
    seed: int = 42,
    render_mode: str = "none",
    compiler: CompilerBackend | None = None,
    compiler_name: str = "smoke",
    max_loops: int | None = None,
    **kwargs: Any,
) -> OperatorStationSession:
    params: dict[str, Any] = {
        "compiler": compiler or SmokeTestCompiler(),
        "compiler_name": compiler_name,
        "env_id": env_id,
        "seed": seed,
        "render_mode": render_mode,
        "memory_root": memory_root or Path(tempfile.mkdtemp()),
    }
    if max_loops is not None:
        params["max_loops"] = max_loops
    params.update(kwargs)
    return OperatorStationSession(**params)


def first_line(response: str) -> str:
    return response.splitlines()[0] if response else ""


def is_motor_execution(response: str) -> bool:
    return "MOTOR COMPLETE" in response or "MOTOR SEQUENCE" in response


def has_meaningful_plan(session: Any) -> bool:
    plan = session.last_request_plan
    graph = session.last_readiness_graph
    return (
        plan is not None
        and graph is not None
        and bool(getattr(plan, "steps", []))
        and getattr(graph, "graph_status", None) is not None
    )


def make_grounding_plan(
    *,
    operation: str,
    metric: str | None,
    order: str | None = None,
    ordinal: int | None = None,
    distance_value: int | None = None,
    comparison: str | None = None,
    required_capabilities: list[str] | None = None,
    answer_fields: list[str] | None = None,
) -> dict:
    primitive_handle = None
    if metric is not None:
        primitive_handle = f"grounding.all_doors.ranked.{metric}.agent"
    return {
        "object_type": "door",
        "operation": operation,
        "primitive_handle": primitive_handle,
        "metric": metric,
        "reference": "agent" if metric else None,
        "order": order,
        "ordinal": ordinal,
        "color": None,
        "exclude_colors": [],
        "distance_value": distance_value,
        "comparison": comparison,
        "tie_policy": "clarify",
        "answer_fields": answer_fields or [],
        "required_capabilities": required_capabilities or (
            [primitive_handle] if primitive_handle else []
        ),
        "preserved_constraints": [],
    }


def ast_source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def ast_call_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                names.append(node.func.attr)
            elif isinstance(node.func, ast.Name):
                names.append(node.func.id)
    return names


def ast_function_call_names(tree: ast.AST, name: str) -> list[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast_call_names(node)
    return []


def emit_result(
    metrics: dict[str, bool],
    details: dict[str, Any] | None = None,
    *,
    pass_metric: str | None = None,
) -> int:
    ok = bool(metrics.get(pass_metric)) if pass_metric is not None else all(metrics.values())
    print(json.dumps({"metrics": metrics, "details": details or {}}, sort_keys=True))
    return 0 if ok else 1
