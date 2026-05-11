"""Consolidated Phase 7.6 eval — Grounding composition and clarification.

Covers:
- Phase 7.6:  Operator clarification loop for grounding (missing distance metric).
- Phase 7.7:  Grounding result composition from ranked claims (closest, farthest,
              ordinal, distance reference, ties, color reference).
- Phase 7.75: Semantic query plan (ordinal, filter, answer fields).

Migrated from: operator_clarification_probe.py, grounding_composition_probe.py,
               operator_query_plan_probe.py.

NOTE: operator_clarification_probe.py required a live LLM by default. This
consolidated version uses the SmokeTestCompiler like the other probes. The
live-LLM clarification can be run separately via --allow-fallback on the
original probe if needed.
"""
from __future__ import annotations

import os
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

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import LLMCompiler, SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.operator_station import OperatorStationSession


ENV_ID = "MiniGrid-GoToDoor-16x16-v0"
SEED = 8
RANKED_HANDLE = "grounding.all_doors.ranked.manhattan.agent"


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session(*, seed: int = SEED) -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id=ENV_ID,
        seed=seed,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
        max_loops=512,
    )


def _run(fn):
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        return fn()


def _task_guardrail_checks(session: OperatorStationSession, prefix: str) -> dict[str, bool]:
    result = session.last_result or {}
    return {
        f"{prefix}_task_complete": bool(result.get("final_state", {}).get("task_complete")),
        f"{prefix}_runtime_llm_calls_zero": result.get("runtime_llm_calls_during_render") == 0,
        f"{prefix}_cache_miss_zero": result.get("cache_miss_during_render") == 0,
    }


# ── Phase 7.7: Grounding Composition ────────────────────────────────────────

def _check_grounding_composition(checks: dict[str, bool]) -> None:
    # 1. Compose closest+farthest answer from ranked claims
    s1 = _make_session()
    response = _run(lambda: s1.handle_utterance("which door is closest and which is farthest"))
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

    # 2. Farthest task must clarify on tie, then execute after choice
    s2 = _make_session()
    clarify = _run(lambda: s2.handle_utterance("go to the farthest door"))
    checks["farthest_task_clarifies_on_tie"] = (
        "CLARIFY" in clarify and "multiple farthest" in clarify
    )
    checks["farthest_tie_did_not_execute"] = s2.last_result is None
    resumed = _run(lambda: s2.handle_utterance("red"))
    checks["farthest_red_answer_runs"] = "RUN COMPLETE" in resumed
    checks["farthest_red_instruction"] = (
        s2.last_result is not None
        and s2.last_result["task"]["instruction"] == "go to the red door"
    )
    checks.update(_task_guardrail_checks(s2, "farthest_red"))

    # 3. Second closest composes ranked[1] into go_to_object task
    s3 = _make_session()
    second = _run(lambda: s3.handle_utterance("go to the second closest door"))
    checks["second_closest_runs"] = "RUN COMPLETE" in second
    checks["second_closest_is_yellow"] = (
        s3.last_result is not None
        and s3.last_result["task"]["instruction"] == "go to the yellow door"
    )
    checks.update(_task_guardrail_checks(s3, "second_closest"))

    # 4. Distance reference composes from ranked claims
    s4 = _make_session()
    distance = _run(lambda: s4.handle_utterance("go to the door with a distance of 7"))
    checks["distance_7_runs"] = "RUN COMPLETE" in distance
    checks["distance_7_is_yellow"] = (
        s4.last_result is not None
        and s4.last_result["task"]["instruction"] == "go to the yellow door"
    )
    checks.update(_task_guardrail_checks(s4, "distance_7"))

    # 5. Ranked display then color reference
    s5 = _make_session()
    ranked = _run(lambda: s5.handle_utterance("rank all the doors by manhattan distance"))
    color_ref = _run(lambda: s5.handle_utterance("go to the red one"))
    checks["ranked_display_returns_list"] = "DOORS RANKED BY MANHATTAN DISTANCE" in ranked
    checks["red_one_runs_from_claims"] = "RUN COMPLETE" in color_ref
    checks.update(_task_guardrail_checks(s5, "red_one"))

    # 6. Second farthest tie does not degrade
    s6 = _make_session(seed=12)
    second_farthest = _run(lambda: s6.handle_utterance("can you navigate to the second farthest door"))
    checks["second_farthest_clarifies_tie"] = (
        "CLARIFY" in second_farthest
        and "ordinal falls inside a distance tie" in second_farthest
    )
    checks["second_farthest_did_not_execute"] = s6.last_result is None


# ── Phase 7.6: Clarification Loop ───────────────────────────────────────────

def _check_clarification(checks: dict[str, bool]) -> None:
    # SmokeTestCompiler handles "go to the closest door" → needs_clarification
    s = OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    _run(lambda: s.handle_utterance("go to the red door"))  # warm scene
    s.last_result = None

    first_response = _run(lambda: s.handle_utterance("go to the closest door"))
    checks["clarify_station_returned_clarify"] = first_response.startswith("CLARIFY")
    checks["clarify_mentions_distance_metric"] = "distance metric" in first_response.lower()
    checks["clarify_lists_manhattan"] = "manhattan" in first_response.lower()
    checks["clarify_pending_created"] = s.pending_clarification is not None
    checks["clarify_pending_type"] = (
        s.pending_clarification is not None
        and s.pending_clarification.clarification_type == "target_selector_missing_field"
    )
    checks["clarify_pending_missing_field"] = (
        s.pending_clarification is not None
        and s.pending_clarification.missing_field == "distance_metric"
    )
    checks["clarify_no_result_before_answer"] = s.last_result is None

    # Answer with "manhattan" → task executes
    answer_response = _run(lambda: s.handle_utterance("manhattan"))
    checks["clarify_answer_cleared_pending"] = s.pending_clarification is None
    checks["clarify_answer_runs"] = (
        isinstance(answer_response, str) and "RUN COMPLETE" in answer_response
    )


# ── Phase 7.75: Semantic Query Plan ─────────────────────────────────────────

def _check_query_plan(checks: dict[str, bool]) -> None:
    # Use SmokeTestCompiler which has query plan support
    compiler = SmokeTestCompiler()
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))

    # 1. Second farthest task
    intent_sf = compiler.compile_operator_intent(
        "can you navigate to the second farthest door", memory=memory
    )
    plan_sf = intent_sf.grounding_query_plan or {}
    checks["qplan_second_farthest_has_plan"] = intent_sf.grounding_query_plan is not None
    checks["qplan_second_farthest_operation"] = plan_sf.get("operation") == "select"
    checks["qplan_second_farthest_order"] = plan_sf.get("order") == "descending"
    checks["qplan_second_farthest_ordinal"] = plan_sf.get("ordinal") == 2

    # 2. Red distance answer
    intent_rd = compiler.compile_operator_intent(
        "how far is the red door", memory=memory
    )
    plan_rd = intent_rd.grounding_query_plan or {}
    checks["qplan_red_distance_has_plan"] = intent_rd.grounding_query_plan is not None
    checks["qplan_red_distance_operation"] = plan_rd.get("operation") == "answer"
    checks["qplan_red_distance_color"] = plan_rd.get("color") == "red"
    checks["qplan_red_distance_answer_field"] = (
        "distance" in (plan_rd.get("answer_fields") or [])
    )

    # 3. Closest and farthest answer
    intent_cf = compiler.compile_operator_intent(
        "which door is closest and which is farthest?", memory=memory
    )
    plan_cf = intent_cf.grounding_query_plan or {}
    checks["qplan_closest_farthest_has_plan"] = intent_cf.grounding_query_plan is not None
    fields = set(plan_cf.get("answer_fields") or [])
    checks["qplan_closest_farthest_fields"] = {"closest", "farthest"}.issubset(fields)


def main() -> int:
    checks: dict[str, bool] = {}

    print("CONSOLIDATED EVAL: PHASE 7.6 (Clarification + Grounding + QueryPlan)\n")

    print("── Phase 7.6: Clarification Loop ──")
    _check_clarification(checks)

    print("── Phase 7.7: Grounding Composition ──")
    _check_grounding_composition(checks)

    print("── Phase 7.75: Semantic Query Plan ──")
    _check_query_plan(checks)

    print("\nCHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    n_pass = sum(checks.values())
    print(f"\n{n_pass}/{len(checks)} passed")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
