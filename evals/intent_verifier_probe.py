"""Phase 7.595 — Proactive Intent Signal Verification probe.

Verifies that:
- IntentVerifier has no substrate imports.
- SUPERLATIVE signals detected: farthest, furthest, most distant.
- CARDINALITY signals detected: all doors, sort by distance, distance of all.
- ORDINAL signals detected: second closest, third nearest.
- Signals inject the correct required_capabilities handles.
- No double-injection when LLM already declared the handle.
- Normal queries (closest, red door) produce no signals.
- Full session: "go to the farthest door" → missing_skills, no task executed.
- Full session: "distance of all the doors" → missing_skills.
- Full session: "sort the doors by distance" → missing_skills.
- Full session: "second closest door" → missing_skills.
- Golden path "go to the red door" still works after IntentVerifier added.
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

from jeenom.intent_verifier import (
    IntentVerifier,
    SIGNAL_SUPERLATIVE,
    SIGNAL_CARDINALITY,
    SIGNAL_ORDINAL,
    default_verifier,
)
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import OperatorIntent


def _make_session() -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )


def _run(fn):
    def fake(env_id, render_mode):
        return FullyObsWrapper(gym.make(env_id))
    with patch("jeenom.run_demo.build_env", side_effect=fake):
        return fn()


def main() -> int:
    checks: dict[str, bool] = {}
    verifier = IntentVerifier()

    # ── 1. No substrate imports ────────────────────────────────────────────
    import ast
    import jeenom.intent_verifier as iv_mod
    iv_source = Path(iv_mod.__file__).read_text()
    iv_tree = ast.parse(iv_source)
    imported = {
        node.module.split(".")[0]
        for node in ast.walk(iv_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(iv_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    checks["no_minigrid_import"] = "minigrid" not in imported
    checks["no_gymnasium_import"] = "gymnasium" not in imported

    # ── 2. SUPERLATIVE signals ─────────────────────────────────────────────
    base_intent = OperatorIntent(
        intent_type="task_instruction",
        task_type="go_to_object",
        target={"color": None, "object_type": "door"},
        required_capabilities=[],
        confidence=0.9,
    )

    superlative_cases = [
        ("farthest door", "farthest"),
        ("go to the furthest door", "furthest"),
        ("navigate to the most distant door", "most_distant"),
    ]
    for utterance, label in superlative_cases:
        enriched, result = verifier.enrich(utterance, base_intent)
        checks[f"superlative_signal_{label}"] = (
            any(s.signal_type == SIGNAL_SUPERLATIVE for s in result.signals)
        )
        checks[f"superlative_injects_handle_{label}"] = (
            "grounding.farthest_door.manhattan.agent" in enriched.required_capabilities
        )

    # ── 3. CARDINALITY signals ─────────────────────────────────────────────
    cardinality_cases = [
        "what is the distance of all the doors from you",
        "can you sort the doors by distance",
        "rank all the doors by manhattan distance",
        "tell me all doors in descending order",
        "list the doors by distance",
        "distance of all doors",
    ]
    for utterance in cardinality_cases:
        enriched, result = verifier.enrich(utterance, base_intent)
        key = utterance[:30].replace(" ", "_")
        checks[f"cardinality_{key}"] = (
            any(s.signal_type == SIGNAL_CARDINALITY for s in result.signals)
            and "grounding.ranked_doors.manhattan.agent" in enriched.required_capabilities
        )

    # ── 4. ORDINAL signals ─────────────────────────────────────────────────
    ordinal_cases = [
        ("go to the second closest door", "grounding.nth_closest_door.manhattan.agent", SIGNAL_ORDINAL),
        ("go to the third nearest door", "grounding.nth_closest_door.manhattan.agent", SIGNAL_ORDINAL),
        # "second farthest" — superlative fires first (farthest in term list), same handle
        ("go to the second farthest door", "grounding.farthest_door.manhattan.agent", None),
    ]
    for utterance, expected_handle, expected_signal in ordinal_cases:
        enriched, result = verifier.enrich(utterance, base_intent)
        signal_ok = (
            expected_signal is None
            or any(s.signal_type == expected_signal for s in result.signals)
        )
        checks[f"ordinal_{utterance[:25].replace(' ','_')}"] = (
            signal_ok and expected_handle in enriched.required_capabilities
        )

    # ── 5. No double-injection ─────────────────────────────────────────────
    already_declared = OperatorIntent(
        intent_type="status_query",
        status_query="ground_target",
        required_capabilities=["grounding.farthest_door.manhattan.agent"],
        confidence=0.9,
    )
    enriched_dup, result_dup = verifier.enrich("go to the farthest door", already_declared)
    checks["no_double_injection"] = (
        enriched_dup.required_capabilities.count("grounding.farthest_door.manhattan.agent") == 1
    )
    checks["no_injection_when_already_declared"] = result_dup.injected_handles == []

    # ── 6. Normal queries produce no signals ──────────────────────────────
    normal_cases = [
        "go to the red door",
        "which door is closest by manhattan distance",
        "go to the purple door",
        "what do you see around you",
    ]
    for utterance in normal_cases:
        _, result = verifier.enrich(utterance, base_intent)
        checks[f"no_signal_{utterance[:20].replace(' ','_')}"] = not result.has_signals

    # ── 7. Full session: farthest door → missing_skills, no task ──────────
    session = _make_session()
    _run(lambda: session.handle_utterance("go to the red door"))
    task_ran = [False]
    original_run_task = session.run_task

    def tracking_run_task(instruction):
        task_ran[0] = True
        return original_run_task(instruction)

    session.run_task = tracking_run_task
    result_far = _run(lambda: session.handle_utterance("go to the farthest door"))
    checks["farthest_returns_missing_skills"] = "MISSING" in result_far.upper()
    checks["farthest_does_not_execute_task"] = not task_ran[0]

    print("FARTHEST DOOR response")
    print(result_far)
    print()

    # ── 8. Full session: "distance of all the doors" → missing_skills ──────
    session2 = _make_session()
    _run(lambda: session2.handle_utterance("go to the red door"))
    result_all = _run(lambda: session2.handle_utterance("what is the distance of all the doors from you"))
    checks["all_doors_returns_missing_skills"] = "MISSING" in result_all.upper()

    print("ALL DOORS DISTANCE response")
    print(result_all)
    print()

    # ── 9. Full session: "sort doors by distance" → missing_skills ─────────
    session3 = _make_session()
    _run(lambda: session3.handle_utterance("go to the red door"))
    result_sort = _run(lambda: session3.handle_utterance("can you sort the doors by distance"))
    checks["sort_doors_returns_missing_skills"] = "MISSING" in result_sort.upper()

    print("SORT DOORS response")
    print(result_sort)
    print()

    # ── 10. Full session: "second closest" → missing_skills ───────────────
    session4 = _make_session()
    _run(lambda: session4.handle_utterance("go to the red door"))
    result_ord = _run(lambda: session4.handle_utterance("go to the second closest door"))
    checks["second_closest_returns_missing_skills"] = "MISSING" in result_ord.upper()

    print("SECOND CLOSEST response")
    print(result_ord)
    print()

    # ── 11. Golden path still works ────────────────────────────────────────
    session5 = _make_session()
    result_golden = _run(lambda: session5.handle_utterance("go to the red door"))
    checks["golden_path_still_works"] = "task_complete=True" in result_golden

    # ── Summary ────────────────────────────────────────────────────────────
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
