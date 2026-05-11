"""Consolidated Phase 7.59 eval — Intent readiness pipeline.

Covers:
- Phase 7.59:  CapabilityMatcher exact handle matching, no-weakening, synthesizable.
- Phase 7.595: IntentVerifier semantic signal extraction, handle injection.
- Phase 7.596: CapabilityArbitrator gap arbitration, ArbitrationDecision schema.

Migrated from: capability_matcher_probe.py, intent_verifier_probe.py,
               capability_arbitrator_probe.py.
"""
from __future__ import annotations

import ast
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
    CapabilityMatcher,
    VERDICT_EXECUTABLE,
    VERDICT_MISSING_SKILLS,
    VERDICT_SYNTHESIZABLE,
    VERDICT_SKIPPED,
)
from jeenom.capability_registry import CapabilityRegistry
from jeenom.capability_arbitrator import (
    LLMArbitrator,
    SmokeTestArbitrator,
    build_arbitrator,
    default_arbitrator,
)
from jeenom.intent_verifier import (
    IntentVerifier,
    SIGNAL_SUPERLATIVE,
    SIGNAL_CARDINALITY,
    SIGNAL_ORDINAL,
)
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import (
    ARBITRATION_DECISION_TYPES,
    ArbitrationDecision,
    ArbitrationTrace,
    OperatorIntent,
    SceneModel,
    SceneObject,
    SchemaValidationError,
    _migrate_exclude_color,
)


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


def _run(fn):
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        return fn()


# ── Phase 7.59: CapabilityMatcher ────────────────────────────────────────────

def _check_matcher(checks: dict[str, bool]) -> None:
    registry = CapabilityRegistry.minigrid_default()
    matcher = CapabilityMatcher()

    # 1. No substrate imports
    import jeenom.capability_matcher as cm_mod
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
    checks["matcher_no_minigrid_import"] = "minigrid" not in imported_modules
    checks["matcher_no_gymnasium_import"] = "gymnasium" not in imported_modules

    # 2. Implemented handle → executable
    intent_ok = OperatorIntent(
        intent_type="task_instruction",
        task_type="go_to_object",
        target={"color": "red", "object_type": "door"},
        required_capabilities=["task.go_to_object.door"],
        confidence=1.0,
    )
    result_ok = matcher.match(intent_ok, registry)
    checks["implemented_handle_is_executable"] = result_ok.verdict == VERDICT_EXECUTABLE

    # 3. Missing handle → missing_skills
    intent_missing = OperatorIntent(
        intent_type="status_query",
        status_query="ground_target",
        required_capabilities=["grounding.ranked_doors.manhattan.agent"],
        confidence=0.9,
    )
    result_missing = matcher.match(intent_missing, registry)
    checks["missing_handle_verdict_is_missing_skills"] = result_missing.verdict == VERDICT_MISSING_SKILLS

    # 4. No weakening: closest ≠ ranked
    checks["closest_does_not_satisfy_ranked"] = result_missing.verdict == VERDICT_MISSING_SKILLS

    # 5. No weakening: go_to_object ≠ pickup
    intent_pickup = OperatorIntent(
        intent_type="unsupported",
        required_capabilities=["task.pickup.key"],
        confidence=0.5,
    )
    result_pickup = matcher.match(intent_pickup, registry)
    checks["go_to_object_does_not_satisfy_pickup"] = result_pickup.verdict == VERDICT_MISSING_SKILLS

    # 6. Synthesizable → synthesizable
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

    # 7. Multiple missing handles reported
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

    # 8. Empty required_capabilities → skipped
    intent_empty = OperatorIntent(
        intent_type="status_query",
        status_query="scene",
        required_capabilities=[],
        confidence=0.9,
    )
    result_empty = matcher.match(intent_empty, registry)
    checks["empty_required_caps_verdict_is_skipped"] = result_empty.verdict == VERDICT_SKIPPED

    # 9. Matcher overrides LLM capability_status
    session = _make_session()
    _run(lambda: session.handle_utterance("go to the red door"))
    intent_override = OperatorIntent(
        intent_type="status_query",
        status_query="ground_target",
        required_capabilities=["grounding.ranked_doors.manhattan.agent"],
        capability_status="executable",
        confidence=0.9,
    )
    cmd = _run(lambda: session.command_from_operator_intent(intent_override, "test"))
    checks["matcher_overrides_llm_executable"] = cmd.kind == "missing_skills"

    # 10. Lookup exact match
    checks["lookup_exact_match"] = registry.lookup("task.go_to_object.door") is not None
    checks["lookup_missing_returns_none"] = registry.lookup("grounding.ranked_doors.manhattan.agent") is None
    checks["lookup_no_prefix_relaxation"] = registry.lookup("grounding.closest_door") is None


# ── Phase 7.595: IntentVerifier ──────────────────────────────────────────────

def _check_verifier(checks: dict[str, bool]) -> None:
    verifier = IntentVerifier()

    # 1. No substrate imports
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
    checks["verifier_no_minigrid_import"] = "minigrid" not in imported
    checks["verifier_no_gymnasium_import"] = "gymnasium" not in imported

    base_intent = OperatorIntent(
        intent_type="task_instruction",
        task_type="go_to_object",
        target={"color": None, "object_type": "door"},
        required_capabilities=[],
        confidence=0.9,
    )

    # 2. Superlative signals
    for utterance, label in [
        ("farthest door", "farthest"),
        ("go to the furthest door", "furthest"),
    ]:
        enriched, result = verifier.enrich(utterance, base_intent)
        checks[f"superlative_signal_{label}"] = (
            any(s.signal_type == SIGNAL_SUPERLATIVE for s in result.signals)
        )

    # 3. Cardinality signals
    for utterance in [
        "what is the distance of all the doors from you",
        "can you sort the doors by distance",
        "rank all the doors by manhattan distance",
    ]:
        enriched, result = verifier.enrich(utterance, base_intent)
        key = utterance[:30].replace(" ", "_")
        checks[f"cardinality_{key}"] = (
            any(s.signal_type == SIGNAL_CARDINALITY for s in result.signals)
        )

    # 4. Ordinal signals
    enriched, result = verifier.enrich("go to the second closest door", base_intent)
    checks["ordinal_second_closest"] = any(s.signal_type == SIGNAL_ORDINAL for s in result.signals)

    # 5. No double-injection
    already_declared = OperatorIntent(
        intent_type="status_query",
        status_query="ground_target",
        required_capabilities=["grounding.all_doors.ranked.manhattan.agent"],
        confidence=0.9,
    )
    enriched_dup, result_dup = verifier.enrich("go to the farthest door", already_declared)
    checks["no_double_injection"] = (
        enriched_dup.required_capabilities.count("grounding.all_doors.ranked.manhattan.agent") == 1
    )

    # 6. Normal queries produce no signals
    for utterance in ["go to the red door", "what do you see around you"]:
        _, result = verifier.enrich(utterance, base_intent)
        checks[f"no_signal_{utterance[:20].replace(' ','_')}"] = not result.has_signals

    # 7. Full session: farthest → completes successfully
    session = _make_session()
    _run(lambda: session.handle_utterance("go to the red door"))
    result_far = _run(lambda: session.handle_utterance("go to the farthest door"))
    checks["farthest_completes_successfully"] = "task_complete=True" in result_far

    # 8. Golden path still works
    session5 = _make_session()
    result_golden = _run(lambda: session5.handle_utterance("go to the red door"))
    checks["verifier_golden_path_still_works"] = "task_complete=True" in result_golden


# ── Phase 7.596: CapabilityArbitrator ────────────────────────────────────────

def _check_arbitrator(checks: dict[str, bool]) -> None:
    # 1. No substrate imports
    import jeenom.capability_arbitrator as arb_mod
    arb_source = Path(arb_mod.__file__).read_text()
    arb_tree = ast.parse(arb_source)
    arb_imported = {
        node.module.split(".")[0]
        for node in ast.walk(arb_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name.split(".")[0]
        for node in ast.walk(arb_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    checks["arbitrator_no_minigrid_import"] = "minigrid" not in arb_imported
    checks["arbitrator_no_gymnasium_import"] = "gymnasium" not in arb_imported

    # 2. SmokeTestArbitrator: refuse on missing handles
    smoke = SmokeTestArbitrator()
    decision = smoke.arbitrate(
        utterance="go to the farthest door",
        intent_type="status_query",
        required_capabilities=["grounding.farthest_door.manhattan.agent"],
        missing_handles=["grounding.farthest_door.manhattan.agent"],
        synthesizable_handles=[],
        available_handles=["grounding.closest_door.manhattan.agent"],
    )
    checks["smoke_missing_gives_refuse"] = decision.decision_type == "refuse"
    checks["smoke_refuse_not_safe"] = not decision.safe_to_execute

    # 3. SmokeTestArbitrator: synthesize on synthesizable
    decision_synth = smoke.arbitrate(
        utterance="go to the closest door using euclidean distance",
        intent_type="task_instruction",
        required_capabilities=["grounding.closest_door.euclidean.agent"],
        missing_handles=[],
        synthesizable_handles=["grounding.closest_door.euclidean.agent"],
        available_handles=["grounding.closest_door.manhattan.agent"],
    )
    checks["smoke_synthesizable_gives_synthesize"] = decision_synth.decision_type == "synthesize"

    # 4. ArbitrationDecision schema validation
    try:
        ArbitrationDecision(
            decision_type="refuse",
            safe_to_execute=True,
            reason="test",
        )
        checks["schema_refuses_safe_refuse"] = False
    except SchemaValidationError:
        checks["schema_refuses_safe_refuse"] = True

    valid = ArbitrationDecision(
        decision_type="substitute",
        safe_to_execute=True,
        reason="semantically equivalent",
        suggested_handle="grounding.closest_door.manhattan.agent",
    )
    checks["schema_allows_safe_substitute"] = valid.safe_to_execute

    # 5. LLMArbitrator falls back without API key
    llm_arb = LLMArbitrator(api_key=None)
    checks["llm_arb_has_fallback_reason"] = bool(llm_arb._fallback_reason)

    # 6. build_arbitrator
    checks["build_smoke_is_smoke"] = isinstance(build_arbitrator("smoke"), SmokeTestArbitrator)
    checks["build_llm_is_llm"] = isinstance(build_arbitrator("llm"), LLMArbitrator)

    # 7. Session: euclidean door → synthesize trace
    session = _make_session()
    _run(lambda: session.handle_utterance("go to the red door"))
    session.last_arbitration_trace = None
    _run(lambda: session.handle_utterance("go to the closest door using euclidean distance"))
    checks["euclidean_sets_trace"] = session.last_arbitration_trace is not None

    # 8. Golden path produces NO arbitration trace
    session2 = _make_session()
    session2.last_arbitration_trace = None
    _run(lambda: session2.handle_utterance("go to the red door"))
    checks["golden_path_no_trace"] = session2.last_arbitration_trace is None

    # 9. exclude_colors multi-exclusion
    intent = SmokeTestCompiler().compile_operator_intent(
        "go to a door which is not purple or yellow",
        memory=None,
    )
    if intent.target_selector is not None:
        exclude = intent.target_selector.get("exclude_colors", [])
        checks["multi_exclusion_is_list"] = isinstance(exclude, list)
        checks["multi_exclusion_has_purple"] = "purple" in exclude
        checks["multi_exclusion_has_yellow"] = "yellow" in exclude
    else:
        checks["multi_exclusion_is_list"] = False
        checks["multi_exclusion_has_purple"] = False
        checks["multi_exclusion_has_yellow"] = False

    # 10. exclude_color migration
    old_selector = {"object_type": "door", "color": None, "exclude_color": "yellow"}
    _migrate_exclude_color(old_selector)
    checks["migration_adds_exclude_colors"] = "exclude_colors" in old_selector
    checks["migration_value_is_list"] = old_selector.get("exclude_colors") == ["yellow"]
    checks["migration_removes_exclude_color"] = "exclude_color" not in old_selector

    # 11. SceneModel.find() with exclude_colors
    scene = SceneModel(
        agent_x=1, agent_y=1, agent_dir=0,
        grid_width=8, grid_height=8,
        objects=[
            SceneObject(object_type="door", color="red", x=3, y=3),
            SceneObject(object_type="door", color="purple", x=4, y=4),
            SceneObject(object_type="door", color="yellow", x=5, y=5),
            SceneObject(object_type="door", color="blue", x=6, y=6),
        ],
        source="test",
    )
    filtered = scene.find(object_type="door", exclude_colors=["purple", "yellow"])
    filtered_colors = {o.color for o in filtered}
    checks["find_excludes_purple"] = "purple" not in filtered_colors
    checks["find_keeps_red"] = "red" in filtered_colors


def main() -> int:
    checks: dict[str, bool] = {}

    print("CONSOLIDATED EVAL: PHASE 7.59 (Matcher + Verifier + Arbitrator)\n")

    print("── Phase 7.59: CapabilityMatcher ──")
    _check_matcher(checks)

    print("── Phase 7.595: IntentVerifier ──")
    _check_verifier(checks)

    print("── Phase 7.596: CapabilityArbitrator ──")
    _check_arbitrator(checks)

    print("\nCHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    n_pass = sum(checks.values())
    print(f"\n{n_pass}/{len(checks)} passed")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
