from __future__ import annotations

import argparse
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

from jeenom.llm_compiler import LLMCompiler
from jeenom.run_demo import run_episode


def build_test_llm_transport():
    def transport(request):
        method = request["method_name"]
        payload = request["user_payload"]

        if method == "compile_task":
            instruction = payload["instruction"]
            return {
                "instruction": instruction,
                "task_type": "go_to_object",
                "params": {
                    "color": "red",
                    "object_type": "door",
                    "target_location": None,
                },
                "source": "llm_compiler",
            }

        if method == "compile_procedure":
            return {
                "task_type": payload["task_request"]["task_type"],
                "steps": ["locate_object", "navigate_to_object", "verify_adjacent", "done"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "Reusable high-level task recipe.",
            }

        if method == "compile_sense_plan":
            needs = set(payload["evidence_frame"]["needs"])
            context = dict(payload["execution_context"]["params"])
            context.update(payload["evidence_frame"]["context"])
            primitives: list[str] = []
            if needs & {"object_location", "agent_pose", "adjacency_to_target", "occupancy_grid"}:
                primitives.extend(["parse_grid_objects", "build_occupancy_grid"])
            if "object_location" in needs:
                primitives.append("find_object_by_color_type")
            if "agent_pose" in needs:
                primitives.append("get_agent_pose")
            if "adjacency_to_target" in needs:
                primitives.append("check_adjacency")

            required_inputs = ["observation"]
            if context.get("color") is not None:
                required_inputs.append("color")
            if context.get("object_type") is not None:
                required_inputs.append("object_type")

            return {
                "primitives": primitives,
                "required_inputs": required_inputs,
                "produces": ["world_sample", "operational_evidence", "percepts"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "Reusable sensing template.",
            }

        if method == "compile_skill_plan":
            skill = payload["execution_contract"]["skill"]
            if skill == "navigate_to_object":
                primitives = ["plan_grid_path", "execute_next_path_action"]
                required_inputs = ["agent_pose", "target_location", "occupancy_grid", "direction"]
            elif skill == "done":
                primitives = ["done"]
                required_inputs = ["adjacency_to_target"]
            else:
                primitives = [skill]
                required_inputs = ["execution_contract"]

            return {
                "primitives": primitives,
                "required_inputs": required_inputs,
                "produces": ["execution_report", "execution_context"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "Reusable skill template.",
            }

        if method == "compile_memory_updates":
            return {"updates": []}

        raise AssertionError(f"Unexpected method: {method}")

    return transport


def _run_golden(render_mode: str, headless_human: bool):
    compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
    kwargs = {
        "instruction": "go to the red door",
        "compiler_name": "llm",
        "compiler": compiler,
        "env_id": "MiniGrid-GoToDoor-8x8-v0",
        "seed": 42,
        "max_loops": 64,
        "render_mode": render_mode,
        "memory_root": Path(tempfile.mkdtemp()),
        "use_cache": True,
        "prewarm": True,
    }

    if headless_human and render_mode == "human":
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            return run_episode(**kwargs)

    return run_episode(**kwargs)


def _check_result(result):
    final_records = [
        record for record in result["loop_records"] if record["skill_plan"] is not None
    ]
    final_record = final_records[-1] if final_records else None
    checks = {
        "jit_prewarm": result["jit_prewarm"] is True,
        "task_complete": result["final_state"]["task_complete"] is True,
        "runtime_llm_calls_during_render_zero": result["runtime_llm_calls_during_render"] == 0,
        "cache_miss_during_render_zero": result["cache_miss_during_render"] == 0,
        "final_skill_plan_done": bool(final_record)
        and final_record["skill_plan"] == ["done"],
        "final_report_succeeded": bool(final_record)
        and final_record["report"]["status"] == "succeeded",
    }
    return checks, final_record


def _print_summary(result, checks, final_record, show_loops: bool) -> None:
    print("PHASE 3.5 GOLDEN EVAL REPORT")
    print()
    print("CONFIG")
    pprint(
        {
            "instruction": result["task"]["instruction"],
            "compiler_backend": result["compiler_backend"],
            "env_id": "MiniGrid-GoToDoor-8x8-v0",
            "seed": 42,
            "render_mode": "human",
            "cache_enabled": True,
            "prewarm_enabled": True,
        }
    )
    print()
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    print()
    print("KEY RESULTS")
    pprint(
        {
            "task_complete": result["final_state"]["task_complete"],
            "runtime_llm_calls_during_render": result["runtime_llm_calls_during_render"],
            "cache_miss_during_render": result["cache_miss_during_render"],
            "jit_prewarm": result["jit_prewarm"],
            "loop_count": len(result["loop_records"]),
            "final_skill_plan": final_record["skill_plan"] if final_record else None,
            "final_report": final_record["report"] if final_record else None,
            "plan_cache": {
                "hits": result["plan_cache"]["hits"],
                "misses": result["plan_cache"]["misses"],
                "llm_calls_saved": result["plan_cache"]["llm_calls_saved"],
                "entries": len(result["plan_cache"]["entries"]),
            },
        }
    )

    if show_loops:
        print()
        print("LOOP TIMELINE")
        for record in result["loop_records"]:
            pprint(
                {
                    "loop": record["loop"],
                    "evidence_needs": record["evidence_needs"],
                    "sense_plan_cache": record["sense_plan_cache"],
                    "skill_plan": record["skill_plan"],
                    "skill_plan_cache": record["skill_plan_cache"],
                    "action": record["action"],
                    "report_status": record["report"]["status"]
                    if record["report"] is not None
                    else None,
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the Phase 3.5 golden eval report.")
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Open the real MiniGrid human render window instead of running headless.",
    )
    parser.add_argument(
        "--show-loops",
        action="store_true",
        help="Print a compact loop-by-loop timeline.",
    )
    args = parser.parse_args()

    result = _run_golden(render_mode="human", headless_human=not args.show_window)
    checks, final_record = _check_result(result)
    _print_summary(result, checks, final_record, show_loops=args.show_loops)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
