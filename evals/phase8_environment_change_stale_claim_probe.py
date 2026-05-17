"""Phase 8.1 probe: environment identity invalidates stale active claims."""
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


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session(*, seed: int = 42) -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=seed,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )


def main() -> int:
    metrics: dict[str, bool] = {}

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session = _make_session()
        session.handle_utterance("which door is closest by manhattan distance")
        metrics["environment_identity_recorded"] = (
            session.current_environment_identity is not None
            and session.active_claims is not None
            and session.active_claims.environment_fingerprint
            == session.current_environment_identity.fingerprint()
        )
        metrics["claims_valid_before_change"] = (
            session.active_claims is not None
            and session._claims_valid_for_current_environment()
        )

        session.seed = 43
        stale_response = session.handle_utterance("next closest door")
        metrics["claims_stale_after_env_change"] = (
            "Scene has changed" in stale_response
            and session.active_claims is None
            and session.memory.scene_model is None
        )
        metrics["stale_claim_blocked_execution"] = (
            "source=active_claims" not in stale_response
            and session.last_result is None
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
