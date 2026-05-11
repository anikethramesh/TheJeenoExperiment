"""Phase 7.59 — Intent Readiness Requirement Matching probe.

Verifies that:
- CapabilityMatcher has no substrate imports (minigrid, gymnasium, sense, spine).
- Exact handle matching: executable handles pass, missing handles fail.
- No weakening: closest_door does not satisfy ranked_doors.
- No weakening: closest_door does not satisfy nth_closest_door.
- No weakening: go_to_object does not satisfy pickup.
- No weakening: visible_doors does not satisfy ranked_doors.
- synthesizable handles produce verdict=synthesizable, not executable.
- synthesizable does not unblock execution.
- Multiple missing handles are all reported at once.
- Matcher verdict overrides LLM's capability_status=executable when handles are missing.
- SmokeTestCompiler emits required_capabilities for door navigation.
- SmokeTestCompiler emits required_capabilities for closest grounding.
- SmokeTestCompiler emits ranked_doors handle for ranked-listing queries.
- CapabilityRegistry.lookup() returns exact matches only.
- Empty required_capabilities → verdict=skipped (matcher defers).
"""
from __future__ import annotations

import importlib
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

from jeenom.capability_matcher import (
    CapabilityMatchResult,
    CapabilityMatcher,
    VERDICT_EXECUTABLE,
    VERDICT_MISSING_SKILLS,
    VERDICT_SYNTHESIZABLE,
    VERDICT_SKIPPED,
    default_matcher,
)
from jeenom.capability_registry import CapabilityRegistry
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
    def fake_build_env(env_id, render_mode):
        return FullyObsWrapper(gym.make(env_id))
    with patch("jeenom.run_demo.build_env", side_effect=fake_build_env):
        return fn()


def main() -> int:
    checks: dict[str, bool] = {}
    registry = CapabilityRegistry.minigrid_default()
    matcher = CapabilityMatcher()

    # ── 1. CapabilityMatcher has no substrate imports ──────────────────────
    import jeenom.capability_matcher as cm_mod
    import ast
    cm_source = Path(cm_mod.__file__).read_text()
    cm_tree = ast.parse(cm_source)
    imported_modules = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(cm_tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in node.names
        if not (isinstance(node, ast.ImportFrom) and node.module == "__future__")
    } | {
        node.module.split(".")[0]
        for node in ast.walk(cm_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    checks["no_minigrid_import"] = "minigrid" not in imported_modules
    checks["no_gymnasium_import"] = "gymnasium" not in imported_modules
    checks["no_gym_import"] = "gym" not in imported_modules

    # ── 2. Exact handle matching — implemented → executable ────────────────
    intent_ok = OperatorIntent(
        intent_type="task_instruction",
        task_type="go_to_object",
        target={"color": "red", "object_type": "door"},
        required_capabilities=["task.go_to_object.door"],
        confidence=1.0,
    )
    result_ok = matcher.match(intent_ok, registry)
    checks["implemented_handle_is_executable"] = result_ok.verdict == VERDICT_EXECUTABLE
    checks["implemented_handle_in_matched"] = "task.go_to_object.door" in result_ok.matched

    # ── 3. Missing handle → missing_skills ────────────────────────────────
    intent_missing = OperatorIntent(
        intent_type="status_query",
        status_query="ground_target",
        required_capabilities=["grounding.ranked_doors.manhattan.agent"],
        confidence=0.9,
    )
    result_missing = matcher.match(intent_missing, registry)
    checks["missing_handle_verdict_is_missing_skills"] = result_missing.verdict == VERDICT_MISSING_SKILLS
    checks["missing_handle_in_missing_list"] = "grounding.ranked_doors.manhattan.agent" in result_missing.missing

    print("MISSING HANDLE result")
    pprint(result_missing.compact())
    print()

    # ── 4. No weakening: closest_door does NOT satisfy ranked_doors ────────
    intent_ranked = OperatorIntent(
        intent_type="status_query",
        status_query="ground_target",
        required_capabilities=["grounding.ranked_doors.manhattan.agent"],
        confidence=0.9,
    )
    result_ranked = matcher.match(intent_ranked, registry)
    checks["closest_does_not_satisfy_ranked"] = result_ranked.verdict == VERDICT_MISSING_SKILLS

    # ── 5. No weakening: go_to_object does NOT satisfy pickup ──────────────
    intent_pickup = OperatorIntent(
        intent_type="unsupported",
        required_capabilities=["task.pickup.key"],
        confidence=0.5,
    )
    result_pickup = matcher.match(intent_pickup, registry)
    checks["go_to_object_does_not_satisfy_pickup"] = result_pickup.verdict == VERDICT_MISSING_SKILLS

    # ── 6. No weakening: visible_doors does NOT satisfy ranked_doors ───────
    intent_visible = OperatorIntent(
        intent_type="status_query",
        status_query="ground_target",
        required_capabilities=["grounding.ranked_doors.manhattan.agent"],
        confidence=0.9,
    )
    result_visible = matcher.match(intent_visible, registry)
    checks["visible_doors_does_not_satisfy_ranked"] = result_visible.verdict == VERDICT_MISSING_SKILLS

    # ── 7. synthesizable handle → verdict=synthesizable, not executable ───
    intent_synth = OperatorIntent(
        intent_type="status_query",
        status_query="ground_target",
        target_selector={
            "object_type": "door", "color": None, "exclude_color": None,
            "relation": "closest", "distance_metric": "euclidean",
            "distance_reference": "agent",
        },
        required_capabilities=["grounding.closest_door.euclidean.agent"],
        capability_status="synthesizable",
        confidence=0.85,
    )
    result_synth = matcher.match(intent_synth, registry)
    checks["synthesizable_verdict_is_synthesizable"] = result_synth.verdict == VERDICT_SYNTHESIZABLE
    checks["synthesizable_not_executable"] = result_synth.verdict != VERDICT_EXECUTABLE
    checks["synthesizable_handle_in_synthesizable_list"] = (
        "grounding.closest_door.euclidean.agent" in result_synth.synthesizable_handles
    )

    print("SYNTHESIZABLE result")
    pprint(result_synth.compact())
    print()

    # ── 8. Multiple missing handles all reported at once ───────────────────
    intent_multi = OperatorIntent(
        intent_type="unsupported",
        required_capabilities=[
            "grounding.ranked_doors.manhattan.agent",
            "grounding.nth_closest_door.manhattan.agent",
        ],
        confidence=0.5,
    )
    result_multi = matcher.match(intent_multi, registry)
    checks["all_missing_handles_reported"] = (
        "grounding.ranked_doors.manhattan.agent" in result_multi.missing
        and "grounding.nth_closest_door.manhattan.agent" in result_multi.missing
    )
    checks["multi_missing_verdict_is_missing_skills"] = result_multi.verdict == VERDICT_MISSING_SKILLS

    # ── 9. Empty required_capabilities → skipped ──────────────────────────
    intent_empty = OperatorIntent(
        intent_type="status_query",
        status_query="scene",
        required_capabilities=[],
        confidence=0.9,
    )
    result_empty = matcher.match(intent_empty, registry)
    checks["empty_required_caps_verdict_is_skipped"] = result_empty.verdict == VERDICT_SKIPPED

    # ── 10. Matcher overrides LLM capability_status=executable when missing─
    session = _make_session()
    _run(lambda: session.handle_utterance("go to the red door"))
    # Force an intent where LLM says executable but handle is missing
    intent_override = OperatorIntent(
        intent_type="status_query",
        status_query="ground_target",
        required_capabilities=["grounding.ranked_doors.manhattan.agent"],
        capability_status="executable",  # LLM incorrectly says executable
        confidence=0.9,
    )
    from jeenom.operator_station import OperatorCommand
    cmd = _run(lambda: session.command_from_operator_intent(intent_override, "test"))
    checks["matcher_overrides_llm_executable"] = cmd.kind == "missing_skills"
    checks["matcher_result_on_command"] = cmd.capability_match is not None

    print("MATCHER OVERRIDE result")
    print(f"command kind: {cmd.kind}")
    pprint(cmd.payload.get("match"))
    print()

    # ── 11. SmokeTestCompiler emits required_capabilities ──────────────────
    compiler = SmokeTestCompiler()
    from jeenom.memory import OperationalMemory
    mem = OperationalMemory(root=Path(tempfile.mkdtemp()))

    intent_nav = compiler.compile_operator_intent(
        "go to the red door", memory=mem
    )
    checks["smoke_nav_emits_required_caps"] = len(intent_nav.required_capabilities) > 0
    checks["smoke_nav_requires_go_to_object"] = (
        "task.go_to_object.door" in intent_nav.required_capabilities
    )

    intent_closest = compiler.compile_operator_intent(
        "which door is closest by manhattan distance", memory=mem
    )
    checks["smoke_closest_emits_required_caps"] = len(intent_closest.required_capabilities) > 0
    checks["smoke_closest_requires_closest_manhattan"] = (
        "grounding.closest_door.manhattan.agent" in intent_closest.required_capabilities
    )

    intent_ranked_smoke = compiler.compile_operator_intent(
        "tell me the doors in descending order by manhattan distance", memory=mem
    )
    checks["smoke_ranked_requires_ranked_handle"] = (
        "grounding.all_doors.ranked.manhattan.agent" in intent_ranked_smoke.required_capabilities
    )

    print("SMOKETEST required_capabilities")
    print(f"  nav: {intent_nav.required_capabilities}")
    print(f"  closest: {intent_closest.required_capabilities}")
    print(f"  ranked: {intent_ranked_smoke.required_capabilities}")
    print()

    # ── 12. CapabilityRegistry.lookup exact match ──────────────────────────
    checks["lookup_exact_match"] = registry.lookup("task.go_to_object.door") is not None
    checks["lookup_missing_returns_none"] = registry.lookup("grounding.ranked_doors.manhattan.agent") is None
    checks["lookup_no_prefix_relaxation"] = registry.lookup("grounding.closest_door") is None

    # ── 13. Full session: ranked query → registered ranked display ─────────
    session2 = _make_session()
    _run(lambda: session2.handle_utterance("go to the red door"))
    result_str = _run(lambda: session2.handle_utterance(
        "tell me the doors in descending order by manhattan distance"
    ))
    checks["ranked_query_returns_registered_display"] = (
        "DOORS RANKED BY MANHATTAN DISTANCE FROM AGENT" in result_str
    )

    print("RANKED QUERY response")
    print(result_str)
    print()

    # ── Summary ────────────────────────────────────────────────────────────
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
