from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from pprint import pprint
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import LLMCompiler
from jeenom.memory import OperationalMemory


DEFAULT_UTTERANCE = "I see. What is the closest door to you"


def _selector_value(intent_dict: dict[str, Any], key: str) -> Any:
    selector = intent_dict.get("target_selector")
    if not isinstance(selector, dict):
        return None
    return selector.get(key)


def _run_probe(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, bool], LLMCompiler]:
    compiler = LLMCompiler()
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
    registry = CapabilityRegistry.minigrid_default()

    intent = compiler.compile_operator_intent(
        args.utterance,
        memory=memory,
        capability_manifest=registry.compact_summary(),
    )
    intent_dict = asdict(intent)
    used_fallback = "smoke_test_compiler" in compiler.active_backend

    checks = {
        "used_live_llm": not used_fallback,
        "intent_type": intent.intent_type == args.expect_intent_type,
        "capability_status": intent.capability_status == args.expect_capability_status,
        "status_query": intent.status_query == args.expect_status_query,
        "selector_object_type": _selector_value(intent_dict, "object_type") == "door",
        "selector_relation": _selector_value(intent_dict, "relation") == "closest",
        "selector_distance_metric_missing": _selector_value(intent_dict, "distance_metric") is None,
        "selector_distance_reference_missing": _selector_value(intent_dict, "distance_reference") is None,
    }
    if args.allow_fallback:
        checks["used_live_llm"] = True
    return intent_dict, checks, compiler


def _print_report(
    args: argparse.Namespace,
    intent: dict[str, Any],
    checks: dict[str, bool],
    compiler: LLMCompiler,
) -> None:
    print("OPERATOR INTENT CLARIFICATION PROBE")
    print()
    print("CONFIG")
    pprint(
        {
            "utterance": args.utterance,
            "openrouter_api_key_visible": bool(os.environ.get("OPENROUTER_API_KEY")),
            "allow_fallback": args.allow_fallback,
            "expected": {
                "intent_type": args.expect_intent_type,
                "capability_status": args.expect_capability_status,
                "status_query": args.expect_status_query,
                "selector": {
                    "object_type": "door",
                    "relation": "closest",
                    "distance_metric": None,
                    "distance_reference": None,
                },
            },
        }
    )
    print()
    print("COMPILER")
    pprint(
        {
            "active_backend": compiler.active_backend,
            "logs": compiler.logs[-5:],
        }
    )
    print()
    print("INTENT")
    pprint(intent)
    print()
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether the live operator-intent LLM emits a clarification-ready "
            "OperatorIntent for an underspecified grounding query."
        )
    )
    parser.add_argument(
        "--utterance",
        default=DEFAULT_UTTERANCE,
        help="Operator utterance to probe.",
    )
    parser.add_argument(
        "--expect-intent-type",
        default="status_query",
        help="Expected OperatorIntent.intent_type.",
    )
    parser.add_argument(
        "--expect-capability-status",
        default="needs_clarification",
        help="Expected OperatorIntent.capability_status.",
    )
    parser.add_argument(
        "--expect-status-query",
        default="ground_target",
        help="Expected OperatorIntent.status_query.",
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow smoke-test fallback for local script sanity checks. Do not use as a live LLM proof.",
    )
    args = parser.parse_args()

    intent, checks, compiler = _run_probe(args)
    _print_report(args, intent, checks, compiler)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
