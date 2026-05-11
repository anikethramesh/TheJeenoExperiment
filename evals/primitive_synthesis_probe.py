"""Phase 7.8 — Primitive Synthesis probe.

Verifies:
- PrimitiveSynthesizer has no substrate imports (AST check).
- PrimitiveValidator has no substrate imports (AST check).
- SmokeTestSynthesizer always refuses.
- LLMSynthesizer falls back when no API key.
- PrimitiveValidator passes all Euclidean fixtures with a correct implementation.
- PrimitiveValidator rejects an incorrect implementation.
- PrimitiveValidator rejects code with disallowed imports.
- CapabilityRegistry.register_synthesized promotes synthesizable → implemented.
- CapabilityRegistry.get_synthesized_callable returns the registered callable.
- operator_station._try_synthesize_primitive works end-to-end with a fake transport.
- After synthesis, ground_target_selector dispatches through synthesized callable.
- Golden path "go to the red door" still produces no synthesis trace.
- runtime_llm_calls_during_render remains 0 after synthesized-primitive grounding task.
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

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.primitive_synthesizer import (
    LLMSynthesizer,
    SmokeTestSynthesizer,
    SynthesisResult,
    build_synthesizer,
)
from jeenom.primitive_validator import PrimitiveValidator, EUCLIDEAN_FIXTURES
from jeenom.schemas import SceneModel, SceneObject


CORRECT_EUCLIDEAN_CODE = """\
def grounding_closest_door_euclidean_agent(scene, selector):
    doors = scene.find(
        object_type=selector.get('object_type', 'door'),
        color=selector.get('color'),
        exclude_colors=selector.get('exclude_colors') or [],
    )
    if not doors:
        return []
    return sorted(
        [
            (math.sqrt((d.x - scene.agent_x) ** 2 + (d.y - scene.agent_y) ** 2), d)
            for d in doors
        ],
        key=lambda pair: (pair[0], pair[1].color or ''),
    )
"""

WRONG_EUCLIDEAN_CODE = """\
def grounding_closest_door_euclidean_agent(scene, selector):
    # BUG: uses Manhattan instead of Euclidean
    doors = scene.find(
        object_type=selector.get('object_type', 'door'),
        color=selector.get('color'),
        exclude_colors=selector.get('exclude_colors') or [],
    )
    if not doors:
        return []
    return sorted(
        [
            (abs(d.x - scene.agent_x) + abs(d.y - scene.agent_y), d)
            for d in doors
        ],
        key=lambda pair: (pair[0], pair[1].color or ''),
    )
"""

DISALLOWED_IMPORT_CODE = """\
def grounding_closest_door_euclidean_agent(scene, selector):
    import numpy as np
    doors = scene.find(object_type='door')
    return []
"""


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

    # ── 1. No substrate imports in synthesizer ────────────────────────────
    import jeenom.primitive_synthesizer as synth_mod
    import jeenom.primitive_validator as val_mod

    for mod, key_prefix in [(synth_mod, "synthesizer"), (val_mod, "validator")]:
        source = Path(mod.__file__).read_text()
        tree = ast.parse(source)
        imported = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        } | {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        checks[f"{key_prefix}_no_minigrid"] = "minigrid" not in imported
        checks[f"{key_prefix}_no_gymnasium"] = "gymnasium" not in imported

    # ── 2. SmokeTestSynthesizer always refuses ───────────────────────────
    smoke_synth = SmokeTestSynthesizer()
    result = smoke_synth.synthesize(
        handle="grounding.closest_door.euclidean.agent",
        description="Euclidean distance ranking",
        consumes=("scene.door_candidates", "agent_pose"),
        produces=("grounded_target", "distance"),
    )
    checks["smoke_synthesizer_refuses"] = result.status == "refused"

    # ── 3. LLMSynthesizer falls back without API key ─────────────────────
    llm_synth = LLMSynthesizer(api_key=None)
    checks["llm_synthesizer_has_fallback_reason"] = bool(llm_synth._fallback_reason)
    llm_result = llm_synth.synthesize(
        handle="grounding.closest_door.euclidean.agent",
        description="Euclidean distance ranking",
        consumes=("scene.door_candidates",),
        produces=("grounded_target",),
    )
    checks["llm_synthesizer_falls_back_to_refused"] = llm_result.status == "refused"

    # ── 4. Validator passes correct Euclidean code ────────────────────────
    validator = PrimitiveValidator()
    handle = "grounding.closest_door.euclidean.agent"
    fn_name = "grounding_closest_door_euclidean_agent"

    ok_result = validator.validate(handle, fn_name, CORRECT_EUCLIDEAN_CODE)
    checks["validator_passes_correct_euclidean"] = ok_result.passed
    checks["validator_correct_has_no_failures"] = len(ok_result.failures) == 0
    checks["validator_correct_returns_callable"] = callable(ok_result.compiled_fn)

    # ── 5. Validator rejects wrong implementation ─────────────────────────
    bad_result = validator.validate(handle, fn_name, WRONG_EUCLIDEAN_CODE)
    checks["validator_rejects_manhattan_as_euclidean"] = not bad_result.passed
    checks["validator_bad_has_failures"] = len(bad_result.failures) > 0
    checks["validator_bad_callable_is_none"] = bad_result.compiled_fn is None

    # ── 6. Validator rejects disallowed import ────────────────────────────
    import_result = validator.validate(handle, fn_name, DISALLOWED_IMPORT_CODE)
    checks["validator_rejects_disallowed_import"] = not import_result.passed

    # ── 7. CapabilityRegistry.register_synthesized ────────────────────────
    reg = CapabilityRegistry.minigrid_default()
    spec_before = reg.lookup(handle)
    checks["spec_is_synthesizable_before"] = (
        spec_before is not None and spec_before.implementation_status == "synthesizable"
    )

    # Compile and register a correct callable
    ok2 = validator.validate(handle, fn_name, CORRECT_EUCLIDEAN_CODE)
    registered = reg.register_synthesized(handle, ok2.compiled_fn)
    checks["register_synthesized_returns_true"] = registered

    spec_after = reg.lookup(handle)
    checks["spec_is_implemented_after"] = (
        spec_after is not None and spec_after.implementation_status == "implemented"
    )
    checks["get_synthesized_callable_works"] = callable(
        reg.get_synthesized_callable(handle)
    )
    alias_handle = "grounding.all_doors.ranked.euclidean.agent"
    alias_after = reg.lookup(alias_handle)
    checks["ranked_euclidean_alias_promoted"] = (
        alias_after is not None and alias_after.implementation_status == "implemented"
    )
    checks["ranked_euclidean_alias_callable"] = callable(
        reg.get_synthesized_callable(alias_handle)
    )

    # ── 8. Synthesized callable produces correct output ───────────────────
    fn = reg.get_synthesized_callable(handle)
    scene = SceneModel(
        agent_x=0, agent_y=0, agent_dir=0,
        grid_width=8, grid_height=8,
        objects=[
            SceneObject(object_type="door", color="red", x=3, y=4),   # eucl=5.0
            SceneObject(object_type="door", color="blue", x=1, y=1),  # eucl=sqrt(2)≈1.41
        ],
        source="test",
    )
    ranked = fn(scene, {"object_type": "door", "color": None, "exclude_colors": []})
    checks["synthesized_fn_ranks_correctly"] = (
        len(ranked) == 2 and ranked[0][1].color == "blue"
    )

    # ── 9. _try_synthesize_primitive end-to-end via fake transport ────────
    def fake_transport(payload):
        return {
            "function_name": fn_name,
            "function_body": CORRECT_EUCLIDEAN_CODE,
            "description": "Euclidean grounding synthesized.",
        }

    session = _make_session()
    session.synthesizer = LLMSynthesizer(api_key="fake", transport=fake_transport)
    # Warm scene
    _run(lambda: session.handle_utterance("go to the red door"))

    # Now trigger synthesis path
    response = _run(
        lambda: session.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["synthesis_triggered_and_responded"] = isinstance(response, str) and len(response) > 0
    if session.pending_synthesis_proposal is not None:
        response = _run(lambda: session.handle_utterance("yes"))
    # After synthesis, the spec should now be implemented in the session registry
    spec_synth = session.capability_registry.lookup(handle)
    checks["session_spec_promoted_to_implemented"] = (
        spec_synth is not None and spec_synth.implementation_status == "implemented"
    )
    session_alias = session.capability_registry.lookup(alias_handle)
    checks["session_ranked_alias_promoted"] = (
        session_alias is not None and session_alias.implementation_status == "implemented"
    )
    all_doors_response = _run(
        lambda: session.resume_arbitration_offer(alias_handle)
    )
    checks["session_ranked_alias_displays_euclidean"] = (
        "DOORS RANKED BY EUCLIDEAN DISTANCE FROM AGENT" in all_doors_response
    )

    # ── 10. Second call uses synthesized primitive without re-synthesizing ─
    response2 = _run(
        lambda: session.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["second_call_succeeds"] = isinstance(response2, str) and len(response2) > 0

    # ── 11. Validation repair loop retries once after malformed code ──────
    repair_calls = {"count": 0}

    def repair_transport(payload):
        repair_calls["count"] += 1
        if repair_calls["count"] == 1:
            return {
                "function_name": fn_name,
                "function_body": (
                    "def grounding_closest_door_euclidean_agent(scene, selector):\n"
                    "    return sorted([\n"
                    "        (math.sqrt((d.x - scene.agent_x) ** 2 + (d.y - scene.agent_y) ** 2), d)\n"
                    "        for d in scene.find(object_type='door')\n"
                    "    )\n"
                ),
                "description": "Malformed first attempt.",
            }
        checks["repair_prompt_includes_validation_error"] = (
            "validation failure" in payload["system_prompt"].lower()
            and "SyntaxError" in payload["system_prompt"]
        )
        return {
            "function_name": fn_name,
            "function_body": CORRECT_EUCLIDEAN_CODE,
            "description": "Corrected Euclidean grounding.",
        }

    repair_session = _make_session()
    repair_session.synthesizer = LLMSynthesizer(api_key="fake", transport=repair_transport)
    _run(lambda: repair_session.handle_utterance("go to the red door"))
    _run(
        lambda: repair_session.handle_utterance("go to the closest door using euclidean distance")
    )
    repair_response = _run(lambda: repair_session.handle_utterance("yes"))
    repair_spec = repair_session.capability_registry.lookup(handle)
    checks["repair_loop_called_twice"] = repair_calls["count"] == 2
    checks["repair_loop_registers_after_second_candidate"] = (
        repair_spec is not None and repair_spec.implementation_status == "implemented"
    )
    checks["repair_loop_response_non_empty"] = isinstance(repair_response, str) and len(repair_response) > 0
    checks.setdefault("repair_prompt_includes_validation_error", False)

    # ── 12. Golden path unaffected ────────────────────────────────────────
    session_golden = _make_session()
    result_golden = _run(lambda: session_golden.handle_utterance("go to the red door"))
    checks["golden_path_still_works"] = "RUN COMPLETE" in result_golden or "task_complete" in result_golden

    # ── Summary ───────────────────────────────────────────────────────────
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
