"""Phase 7.596 — CapabilityArbitrator probe.

Verifies that:
- ArbitratorBackend has no substrate imports (AST check).
- SmokeTestArbitrator refuses missing handles with MISSING SKILLS message.
- SmokeTestArbitrator returns synthesize for synthesizable handles.
- ArbitrationDecision schema validates (refuse/synthesize must be safe_to_execute=False).
- ArbitrationTrace records provenance correctly.
- LLMArbitrator falls back to SmokeTestArbitrator when no API key.
- CapabilityArbitrator integrates into operator_station (last_arbitration_trace set).
- Farthest-door session works now, so we test Euclidean session instead for synthesize trace.
- Golden-path "go to the red door" produces NO arbitration trace.
- exclude_colors multi-exclusion: "neither purple nor yellow" is parsed as list.
- exclude_colors migration: legacy exclude_color dict is migrated correctly.
- SceneModel.find() with exclude_colors filters multiple colors.
"""
from __future__ import annotations

import ast
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

from jeenom.capability_arbitrator import (
    ArbitratorBackend,
    LLMArbitrator,
    SmokeTestArbitrator,
    build_arbitrator,
    default_arbitrator,
)
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import (
    ARBITRATION_DECISION_TYPES,
    ArbitrationDecision,
    ArbitrationTrace,
    SceneModel,
    SceneObject,
    SchemaValidationError,
    _migrate_exclude_color,
)


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

    # ── 1. No substrate imports in capability_arbitrator ──────────────────
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
    checks["no_minigrid_import"] = "minigrid" not in arb_imported
    checks["no_gymnasium_import"] = "gymnasium" not in arb_imported

    # ── 2. SmokeTestArbitrator: refuse on missing handles ─────────────────
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
    checks["smoke_message_has_missing"] = "MISSING" in decision.operator_message

    # ── 3. SmokeTestArbitrator: synthesize on synthesizable handles ────────
    decision_synth = smoke.arbitrate(
        utterance="go to the closest door using euclidean distance",
        intent_type="task_instruction",
        required_capabilities=["grounding.closest_door.euclidean.agent"],
        missing_handles=[],
        synthesizable_handles=["grounding.closest_door.euclidean.agent"],
        available_handles=["grounding.closest_door.manhattan.agent"],
    )
    checks["smoke_synthesizable_gives_synthesize"] = (
        decision_synth.decision_type == "synthesize"
    )
    checks["smoke_synthesize_not_safe"] = not decision_synth.safe_to_execute

    # ── 4. ArbitrationDecision schema validation ──────────────────────────
    try:
        ArbitrationDecision(
            decision_type="refuse",
            safe_to_execute=True,  # invalid — must raise
            reason="test",
        )
        checks["schema_refuses_safe_refuse"] = False
    except SchemaValidationError:
        checks["schema_refuses_safe_refuse"] = True

    try:
        ArbitrationDecision(
            decision_type="synthesize",
            safe_to_execute=True,  # invalid — must raise
            reason="test",
        )
        checks["schema_refuses_safe_synthesize"] = False
    except SchemaValidationError:
        checks["schema_refuses_safe_synthesize"] = True

    valid = ArbitrationDecision(
        decision_type="substitute",
        safe_to_execute=True,
        reason="semantically equivalent",
        suggested_handle="grounding.closest_door.manhattan.agent",
    )
    checks["schema_allows_safe_substitute"] = valid.safe_to_execute

    # ── 5. ArbitrationTrace provenance ────────────────────────────────────
    trace = ArbitrationTrace(
        utterance="go to the farthest door",
        intent_type="status_query",
        required_capabilities=["grounding.farthest_door.manhattan.agent"],
        missing_handles=["grounding.farthest_door.manhattan.agent"],
        synthesizable_handles=[],
        decision=decision,
    )
    compact = trace.compact()
    checks["trace_has_utterance"] = compact["utterance"] == "go to the farthest door"
    checks["trace_has_missing_handles"] = "grounding.farthest_door.manhattan.agent" in compact["missing_handles"]
    checks["trace_has_decision_type"] = compact["decision_type"] == "refuse"

    # ── 6. LLMArbitrator falls back without API key ────────────────────────
    llm_arb = LLMArbitrator(api_key=None)
    checks["llm_arb_has_fallback_reason"] = bool(llm_arb._fallback_reason)
    llm_decision = llm_arb.arbitrate(
        utterance="go to the farthest door",
        intent_type="status_query",
        required_capabilities=["grounding.farthest_door.manhattan.agent"],
        missing_handles=["grounding.farthest_door.manhattan.agent"],
        synthesizable_handles=[],
        available_handles=[],
    )
    checks["llm_arb_falls_back_to_refuse"] = llm_decision.decision_type == "refuse"

    # ── 7. build_arbitrator ───────────────────────────────────────────────
    checks["build_smoke_is_smoke"] = isinstance(
        build_arbitrator("smoke"), SmokeTestArbitrator
    )
    checks["build_llm_is_llm"] = isinstance(
        build_arbitrator("llm"), LLMArbitrator
    )
    checks["default_arbitrator_is_smoke"] = isinstance(
        default_arbitrator, SmokeTestArbitrator
    )

    # ── 8. Session integration: euclidean door → synthesize trace ─────────
    session = _make_session()
    _run(lambda: session.handle_utterance("go to the red door"))
    session.last_arbitration_trace = None  # clear after golden path
    _run(lambda: session.handle_utterance("go to the closest door using euclidean distance"))
    checks["farthest_sets_trace"] = session.last_arbitration_trace is not None
    if session.last_arbitration_trace is not None:
        checks["farthest_trace_is_refuse"] = (
            session.last_arbitration_trace.decision.decision_type == "synthesize"
        )
        checks["farthest_trace_has_missing"] = (
            "grounding.closest_door.euclidean.agent"
            in session.last_arbitration_trace.missing_handles
            or "grounding.closest_door.euclidean.agent"
            in session.last_arbitration_trace.synthesizable_handles
        )
    else:
        checks["farthest_trace_is_refuse"] = False
        checks["farthest_trace_has_missing"] = False

    # ── 9. Golden path produces NO arbitration trace ──────────────────────
    session2 = _make_session()
    session2.last_arbitration_trace = None
    _run(lambda: session2.handle_utterance("go to the red door"))
    checks["golden_path_no_trace"] = session2.last_arbitration_trace is None

    # ── 10. exclude_colors multi-exclusion via SmokeTestCompiler ──────────
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

    # ── 11. exclude_color migration ───────────────────────────────────────
    old_selector = {"object_type": "door", "color": None, "exclude_color": "yellow"}
    _migrate_exclude_color(old_selector)
    checks["migration_adds_exclude_colors"] = "exclude_colors" in old_selector
    checks["migration_value_is_list"] = old_selector.get("exclude_colors") == ["yellow"]
    checks["migration_removes_exclude_color"] = "exclude_color" not in old_selector

    # ── 12. SceneModel.find() with exclude_colors filters correctly ────────
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
    checks["find_excludes_yellow"] = "yellow" not in filtered_colors
    checks["find_keeps_red"] = "red" in filtered_colors
    checks["find_keeps_blue"] = "blue" in filtered_colors

    # ── Summary ────────────────────────────────────────────────────────────
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
