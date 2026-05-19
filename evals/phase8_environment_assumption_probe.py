"""Phase 8.2 probe: explicit environment assumptions in RequestPlan readiness."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.readiness_graph import evaluate_request_plan


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session(*, env_id: str = "MiniGrid-GoToDoor-8x8-v0", seed: int = 42):
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id=env_id,
        seed=seed,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )


def main() -> int:
    metrics: dict[str, bool] = {}

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session = _make_session(seed=42)
        session.handle_utterance("which door is closest by manhattan distance")
        session.handle_utterance("which door is closest by manhattan distance")
        plan = session.last_request_plan
        identity = session.current_environment_identity

        metrics["plan_has_environment_assumptions"] = (
            plan is not None and len(plan.environment_assumptions) > 0
        )
        metrics["steps_reference_assumptions"] = (
            plan is not None
            and any(step.environment_assumption_ids for step in plan.steps)
        )

        same_graph = evaluate_request_plan(
            plan,
            registry=session.capability_registry,
            environment_identity=identity,
        )
        metrics["same_environment_assumptions_pass"] = (
            same_graph.violated_assumption_ids == []
            and same_graph.graph_status != "environment_assumption_failed"
        )

        changed_seed = _make_session(seed=43)
        changed_seed.handle_utterance("what doors do you see?")
        seed_graph = evaluate_request_plan(
            plan,
            registry=session.capability_registry,
            environment_identity=changed_seed.current_environment_identity,
        )
        metrics["changed_seed_reported_as_diagnostic"] = (
            "env.seed" in seed_graph.diagnostic_assumption_ids
            and seed_graph.graph_status != "environment_assumption_failed"
        )

        changed_env = _make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=42)
        changed_env.handle_utterance("what doors do you see?")
        env_graph = evaluate_request_plan(
            plan,
            registry=session.capability_registry,
            environment_identity=changed_env.current_environment_identity,
        )
        metrics["changed_required_env_reports_specific_assumption"] = (
            "env.env_id" in env_graph.violated_assumption_ids
            or "env.grid_size" in env_graph.violated_assumption_ids
        )
        metrics["required_assumption_uses_environment_assumption_failed"] = (
            env_graph.graph_status == "environment_assumption_failed"
        )

        golden = _make_session(seed=42)
        golden_response = golden.handle_utterance("go to the red door")
        metrics["runtime_llm_calls_during_render_zero"] = (
            "RUN COMPLETE" in golden_response
            and golden.last_result is not None
            and golden.last_result["runtime_llm_calls_during_render"] == 0
        )
        metrics["cache_miss_during_render_zero"] = (
            golden.last_result is not None
            and golden.last_result["cache_miss_during_render"] == 0
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
