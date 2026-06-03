"""Shared helpers for JEENOM eval probes."""
from __future__ import annotations

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


def emit_result(
    metrics: dict[str, bool],
    details: dict[str, Any] | None = None,
    *,
    pass_metric: str | None = None,
) -> int:
    ok = bool(metrics.get(pass_metric)) if pass_metric is not None else all(metrics.values())
    print(json.dumps({"metrics": metrics, "details": details or {}}, sort_keys=True))
    return 0 if ok else 1
