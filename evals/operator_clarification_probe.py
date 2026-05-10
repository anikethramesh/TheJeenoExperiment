from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from pprint import pprint
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.llm_compiler import LLMCompiler
from jeenom.operator_station import OperatorStationSession


CASES = {
    "closest_query": {
        "utterance": "I see. What is the closest door to you",
        "answer": "manhattan",
        "answer_kind": "grounded_target",
    },
    "closest_task": {
        "utterance": "go to the closest door",
        "answer": "manhattan",
        "answer_kind": "run_complete",
    },
}


def _final_record(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    records = [
        record
        for record in result.get("loop_records", [])
        if record.get("skill_plan") is not None
    ]
    return records[-1] if records else None


def _run_case(
    name: str,
    config: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    compiler = LLMCompiler()
    session = OperatorStationSession(
        compiler_name="llm",
        compiler=compiler,
        env_id=args.env_id,
        seed=args.seed,
        render_mode="none",
        max_loops=args.max_loops,
        memory_root=Path(tempfile.mkdtemp()),
        verbose=args.verbose,
    )

    first_response = session.handle_utterance(config["utterance"])
    pending = session.pending_clarification
    last_result_before_answer = session.last_result
    used_fallback = "smoke_test_compiler" in compiler.active_backend

    answer_response = None
    if args.answer and pending is not None:
        answer_response = session.handle_utterance(config["answer"])

    final_record = _final_record(session.last_result)
    checks = {
        "used_live_llm": not used_fallback,
        "station_returned_clarify": first_response.startswith("CLARIFY"),
        "clarify_mentions_closest": "closest" in first_response.lower(),
        "clarify_mentions_distance_metric": "distance metric" in first_response.lower(),
        "clarify_lists_manhattan": "manhattan" in first_response.lower(),
        "pending_clarification_created": pending is not None,
        "pending_type_missing_field": bool(pending)
        and pending.clarification_type == "target_selector_missing_field",
        "pending_missing_field_distance_metric": bool(pending)
        and pending.missing_field == "distance_metric",
        "pending_supports_manhattan": bool(pending)
        and "manhattan" in pending.supported_values,
        "no_result_before_answer": last_result_before_answer is None,
    }
    if args.answer:
        if config["answer_kind"] == "grounded_target":
            checks.update(
                {
                    "answer_cleared_pending": session.pending_clarification is None,
                    "answer_returned_grounded_target": isinstance(answer_response, str)
                    and answer_response.startswith("GROUNDED TARGET"),
                    "answer_has_target": isinstance(answer_response, str)
                    and "target=" in answer_response,
                    "answer_has_distance": isinstance(answer_response, str)
                    and "distance=" in answer_response,
                }
            )
        elif config["answer_kind"] == "run_complete":
            checks.update(
                {
                    "answer_cleared_pending": session.pending_clarification is None,
                    "answer_returned_run_complete": isinstance(answer_response, str)
                    and answer_response.startswith("RUN COMPLETE"),
                    "task_complete": bool(session.last_result)
                    and session.last_result["final_state"]["task_complete"] is True,
                    "runtime_llm_calls_during_render_zero": bool(session.last_result)
                    and session.last_result["runtime_llm_calls_during_render"] == 0,
                    "cache_miss_during_render_zero": bool(session.last_result)
                    and session.last_result["cache_miss_during_render"] == 0,
                    "final_skill_plan_done": bool(final_record)
                    and final_record["skill_plan"] == ["done"],
                }
            )

    return {
        "case": name,
        "utterance": config["utterance"],
        "answer": config["answer"] if args.answer else None,
        "compiler": {
            "active_backend": compiler.active_backend,
            "logs": compiler.logs[-5:],
        },
        "first_response": first_response,
        "pending_after_first_response": None
        if pending is None
        else {
            "clarification_type": pending.clarification_type,
            "missing_field": pending.missing_field,
            "supported_values": pending.supported_values,
            "resume_kind": pending.resume_kind,
            "partial_selector": pending.partial_selector,
        },
        "answer_response": answer_response,
        "last_result_summary": None
        if session.last_result is None
        else {
            "task_complete": session.last_result["final_state"]["task_complete"],
            "runtime_llm_calls_during_render": session.last_result[
                "runtime_llm_calls_during_render"
            ],
            "cache_miss_during_render": session.last_result["cache_miss_during_render"],
            "final_skill_plan": final_record["skill_plan"] if final_record else None,
        },
        "checks": checks,
    }


def _print_report(args: argparse.Namespace, results: list[dict[str, Any]]) -> None:
    print("OPERATOR CLARIFICATION STATION PROBE")
    print()
    print("CONFIG")
    pprint(
        {
            "case": args.case,
            "answer_enabled": args.answer,
            "openrouter_api_key_visible": bool(os.environ.get("OPENROUTER_API_KEY")),
            "allow_fallback": args.allow_fallback,
            "env_id": args.env_id,
            "seed": args.seed,
            "max_loops": args.max_loops,
        }
    )
    for result in results:
        print()
        print(f"CASE {result['case']}")
        print()
        print("UTTERANCE")
        print(result["utterance"])
        print()
        print("COMPILER")
        pprint(result["compiler"])
        print()
        print("FIRST RESPONSE")
        print(result["first_response"])
        print()
        print("PENDING")
        pprint(result["pending_after_first_response"])
        if args.answer:
            print()
            print("ANSWER")
            print(result["answer"])
            print()
            print("ANSWER RESPONSE")
            print(result["answer_response"])
            print()
            print("LAST RESULT")
            pprint(result["last_result_summary"])
        print()
        print("CHECKS")
        for name, passed in result["checks"].items():
            print(f"{'PASS' if passed else 'FAIL'} {name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether the live operator station turns a live LLM "
            "needs_clarification intent into an operator-facing CLARIFY question."
        )
    )
    parser.add_argument(
        "--case",
        choices=["all", *CASES],
        default="all",
        help="Clarification case to run.",
    )
    parser.add_argument(
        "--no-answer",
        dest="answer",
        action="store_false",
        help="Only check that the station asks a clarification question.",
    )
    parser.set_defaults(answer=True)
    parser.add_argument("--env-id", default="MiniGrid-GoToDoor-16x16-v0")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--max-loops", type=int, default=512)
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow smoke-test fallback for script sanity checks. Do not use as live LLM proof.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    selected_cases = CASES if args.case == "all" else {args.case: CASES[args.case]}
    results = [
        _run_case(name, config, args)
        for name, config in selected_cases.items()
    ]
    _print_report(args, results)
    def _result_passed(result: dict[str, Any]) -> bool:
        checks = dict(result["checks"])
        if args.allow_fallback:
            checks.pop("used_live_llm", None)
        return all(checks.values())

    return 0 if all(_result_passed(result) for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
