"""Phase 7.75 - Operator semantic query-plan probe.

Checks whether the operator-intent compiler emits typed GroundingQueryPlan objects.
By default this expects a live OpenRouter-backed LLM. Use --allow-fallback only to
verify eval wiring; fallback mode does not prove live LLM planning.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import LLMCompiler
from jeenom.memory import OperationalMemory


CASES = [
    {
        "name": "second_farthest_task",
        "utterance": "can you navigate to the second farthest door",
        "expect": {
            "intent_type": "task_instruction",
            "operation": "select",
            "order": "descending",
            "ordinal": 2,
            "primitive_handle": "grounding.all_doors.ranked.manhattan.agent",
            "tie_policy": "clarify",
        },
    },
    {
        "name": "red_distance_answer",
        "utterance": "how far is the red door",
        "expect": {
            "intent_type": "status_query",
            "operation": "answer",
            "color": "red",
            "answer_field": "distance",
        },
    },
    {
        "name": "green_exists_answer",
        "utterance": "is there a green door",
        "expect": {
            "intent_type": "status_query",
            "operation": "answer",
            "color": "green",
            "answer_field": "exists",
        },
    },
    {
        "name": "closest_and_farthest_answer",
        "utterance": "which door is closest and which is farthest?",
        "expect": {
            "intent_type": "status_query",
            "operation": "answer",
            "answer_fields": ["closest", "farthest"],
        },
    },
    {
        "name": "closest_and_second_closest_answer",
        "utterance": "which door is the closest and second closest from you",
        "expect": {
            "intent_type": "status_query",
            "operation": "answer",
            "answer_fields": ["closest", "second_closest"],
        },
    },
    {
        "name": "go_to_that_claim_reference",
        "utterance": "go to that",
        "active_claims_summary": {
            "last_grounded_target": "purple door @ distance 6",
            "ranked_doors": ["grey@1", "purple@6"],
            "last_rank": 1,
        },
        "expect": {
            "intent_type": "task_instruction",
            "operation": "select",
            "primitive_handle": "grounding.claims.last_grounded_target",
            "answer_field": "target",
        },
    },
]


def _check_case(intent, expect: dict) -> dict[str, bool]:
    plan = intent.grounding_query_plan or {}
    checks = {
        "has_grounding_query_plan": intent.grounding_query_plan is not None,
        "intent_type": intent.intent_type == expect.get("intent_type"),
    }
    for key in ("operation", "order", "ordinal", "primitive_handle", "tie_policy", "color"):
        if key in expect:
            checks[key] = plan.get(key) == expect[key]
    if "answer_field" in expect:
        checks[f"answer_field_{expect['answer_field']}"] = (
            expect["answer_field"] in (plan.get("answer_fields") or [])
        )
    if "answer_fields" in expect:
        fields = set(plan.get("answer_fields") or [])
        checks["answer_fields"] = set(expect["answer_fields"]).issubset(fields)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[c["name"] for c in CASES] + ["all"], default="all")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow fallback compiler. This does not prove live LLM query planning.",
    )
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    compiler = LLMCompiler(api_key=api_key)
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
    manifest = CapabilityRegistry.minigrid_default().compact_summary()
    selected = CASES if args.case == "all" else [c for c in CASES if c["name"] == args.case]

    print("OPERATOR QUERY PLAN PROBE")
    print()
    print("CONFIG")
    pprint(
        {
            "case": args.case,
            "allow_fallback": args.allow_fallback,
            "openrouter_api_key_visible": bool(api_key),
        }
    )
    print()

    all_checks: dict[str, bool] = {}
    for case in selected:
        before = len(compiler.call_history)
        intent = compiler.compile_operator_intent(
            case["utterance"],
            memory=memory,
            capability_manifest=manifest,
            active_claims_summary=case.get("active_claims_summary"),
        )
        recent_calls = compiler.call_history[before:]
        used_live = any(
            call["method_name"] == "compile_operator_intent"
            and call["backend"] == "llm_compiler"
            and not call["used_fallback"]
            for call in recent_calls
        )

        print(f"CASE {case['name']}")
        print("UTTERANCE")
        print(case["utterance"])
        print("INTENT")
        pprint(
            {
                "intent_type": intent.intent_type,
                "capability_status": intent.capability_status,
                "required_capabilities": intent.required_capabilities,
                "grounding_query_plan": intent.grounding_query_plan,
                "reason": intent.reason,
            }
        )
        print("COMPILER LOGS")
        pprint(compiler.logs[-5:])
        print()

        checks = _check_case(intent, case["expect"])
        checks["used_live_llm"] = used_live or args.allow_fallback
        for name, passed in checks.items():
            all_checks[f"{case['name']}:{name}"] = passed

    print("CHECKS")
    for name, passed in all_checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    return 0 if all(all_checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
