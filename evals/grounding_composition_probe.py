"""Phase 7.7 - Grounding Result Composition probe.

This eval verifies that the station composes operator answers and tasks from:
- the registered ranked-door grounding primitive handle
- SceneModel
- StationActiveClaims

It intentionally does not require live LLM access. The point is to prove the
composition substrate and runtime guardrails, not provider behavior.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from pprint import pprint
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession


ENV_ID = "MiniGrid-GoToDoor-16x16-v0"
SEED = 8
RANKED_HANDLE = "grounding.all_doors.ranked.manhattan.agent"


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session() -> OperatorStationSession:
    return _make_session_for(seed=SEED)


def _make_session_for(*, seed: int) -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id=ENV_ID,
        seed=seed,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
        max_loops=512,
    )


def _run_session_step(session: OperatorStationSession, utterance: str) -> str:
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        return session.handle_utterance(utterance)


def _task_guardrail_checks(session: OperatorStationSession, prefix: str) -> dict[str, bool]:
    result = session.last_result or {}
    return {
        f"{prefix}_task_complete": bool(result.get("final_state", {}).get("task_complete")),
        f"{prefix}_runtime_llm_calls_zero": result.get("runtime_llm_calls_during_render") == 0,
        f"{prefix}_cache_miss_zero": result.get("cache_miss_during_render") == 0,
    }


def main() -> int:
    checks: dict[str, bool] = {}

    print("GROUNDING COMPOSITION PROBE")
    print()
    print("CONFIG")
    pprint(
        {
            "env_id": ENV_ID,
            "seed": SEED,
            "compiler": "smoke",
            "ranked_handle": RANKED_HANDLE,
        }
    )
    print()

    # 1. Compose closest+farthest answer from ranked claims.
    s1 = _make_session()
    response = _run_session_step(s1, "which door is closest and which is farthest")
    print("CASE closest_and_farthest_answer")
    print(response)
    print()
    checks["closest_farthest_answer"] = "GROUNDING ANSWER" in response
    checks["closest_answer_has_purple"] = "closest=purple door@(4,0) distance=5" in response
    checks["farthest_answer_has_blue_red_tie"] = (
        "blue door@(12,3) distance=8" in response
        and "red door@(10,7) distance=8" in response
        and "tie=" in response
    )
    checks["claims_written_from_ranked_handle"] = (
        s1.active_claims is not None
        and s1.active_claims.last_grounding_query.get("primitive") == RANKED_HANDLE
    )
    checks["answer_did_not_execute"] = s1.last_result is None

    # 2. Farthest task must clarify on tie, then execute only after operator chooses.
    s2 = _make_session()
    clarify = _run_session_step(s2, "go to the farthest door")
    print("CASE farthest_task_tie")
    print(clarify)
    print()
    checks["farthest_task_clarifies_on_tie"] = (
        "CLARIFY" in clarify and "multiple farthest" in clarify
    )
    checks["farthest_tie_did_not_execute"] = s2.last_result is None
    resumed = _run_session_step(s2, "red")
    print("CASE farthest_task_after_red_answer")
    print(resumed)
    print()
    checks["farthest_red_answer_runs"] = "RUN COMPLETE" in resumed
    checks["farthest_red_instruction"] = (
        s2.last_result is not None
        and s2.last_result["task"]["instruction"] == "go to the red door"
    )
    checks.update(_task_guardrail_checks(s2, "farthest_red"))

    # 3. Second closest composes ranked[1] into a go_to_object task.
    s3 = _make_session()
    second = _run_session_step(s3, "go to the second closest door")
    print("CASE second_closest_task")
    print(second)
    print()
    checks["second_closest_runs"] = "RUN COMPLETE" in second
    checks["second_closest_is_yellow"] = (
        s3.last_result is not None
        and s3.last_result["task"]["instruction"] == "go to the yellow door"
    )
    checks.update(_task_guardrail_checks(s3, "second_closest"))

    # 4. Distance reference composes from ranked claims.
    s4 = _make_session()
    distance = _run_session_step(s4, "go to the door with a distance of 7")
    print("CASE distance_reference_task")
    print(distance)
    print()
    checks["distance_7_runs"] = "RUN COMPLETE" in distance
    checks["distance_7_is_yellow"] = (
        s4.last_result is not None
        and s4.last_result["task"]["instruction"] == "go to the yellow door"
    )
    checks.update(_task_guardrail_checks(s4, "distance_7"))

    # 5. A ranked display creates active claims; color reference uses those claims.
    s5 = _make_session()
    ranked = _run_session_step(s5, "rank all the doors by manhattan distance")
    color_ref = _run_session_step(s5, "go to the red one")
    print("CASE ranked_display_then_color_reference")
    print(ranked)
    print(color_ref)
    print()
    checks["ranked_display_returns_list"] = "DOORS RANKED BY MANHATTAN DISTANCE" in ranked
    checks["red_one_runs_from_claims"] = "RUN COMPLETE" in color_ref
    checks["red_one_instruction"] = (
        s5.last_result is not None
        and s5.last_result["task"]["instruction"] == "go to the red door"
    )
    checks.update(_task_guardrail_checks(s5, "red_one"))

    # 6. Seed 12 has a second-farthest tie. Do not degrade it to farthest.
    s6 = _make_session_for(seed=12)
    second_farthest = _run_session_step(s6, "can you navigate to the second farthest door")
    print("CASE second_farthest_tie_does_not_degrade")
    print(second_farthest)
    print()
    checks["second_farthest_clarifies_tie"] = (
        "CLARIFY" in second_farthest
        and "ordinal falls inside a distance tie" in second_farthest
    )
    checks["second_farthest_options_are_rank_tie"] = (
        "green door@(9,0)" in second_farthest
        and "yellow door@(11,2)" in second_farthest
    )
    checks["second_farthest_did_not_execute"] = s6.last_result is None

    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
