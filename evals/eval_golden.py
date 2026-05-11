"""Cross-phase golden regression eval.

Verifies that the golden path ("go to the red door") passes through ALL
architectural layers without regressions:
- Intent compilation → CapabilityMatcher → task execution
- Prewarm, 0 LLM calls during render, 0 cache misses
- No synthesis proposals, no arbitration traces, no clarification pending.

This eval should pass at every phase. If it breaks, the architecture is regressed.
"""
from __future__ import annotations

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


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session() -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )


def main() -> int:
    checks: dict[str, bool] = {}

    print("CROSS-PHASE GOLDEN REGRESSION EVAL\n")

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session = _make_session()
        result = session.handle_utterance("go to the red door")

    # 1. Task completed
    checks["result_is_string"] = isinstance(result, str) and len(result) > 0
    checks["task_complete"] = (
        session.last_result is not None
        and session.last_result["final_state"]["task_complete"] is True
    )
    checks["runtime_llm_calls_zero"] = (
        session.last_result is not None
        and session.last_result["runtime_llm_calls_during_render"] == 0
    )
    checks["cache_miss_zero"] = (
        session.last_result is not None
        and session.last_result["cache_miss_during_render"] == 0
    )

    # 2. No side-effects
    checks["no_pending_clarification"] = session.pending_clarification is None
    checks["no_pending_synthesis"] = session.pending_synthesis_proposal is None
    checks["no_arbitration_trace"] = session.last_arbitration_trace is None
    checks["no_active_claims"] = session.active_claims is None

    # 3. Memory updated
    checks["scene_model_populated"] = session.memory.scene_model is not None
    checks["last_result_recorded"] = session.last_result is not None

    # 4. Run second query to verify session still functional
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        scene_response = session.handle_utterance("what do you see")
    checks["second_query_responds"] = isinstance(scene_response, str) and len(scene_response) > 0

    # 5. Run another task to verify re-execution
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        second_task = session.handle_utterance("go to the red door")
    checks["second_task_completes"] = (
        session.last_result is not None
        and session.last_result["final_state"]["task_complete"] is True
    )

    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    n_pass = sum(checks.values())
    print(f"\n{n_pass}/{len(checks)} passed")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
