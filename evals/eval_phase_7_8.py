"""Consolidated Phase 7.8 eval — Primitive synthesis and collaborative composition.

Covers:
- Phase 7.8: Validated grounding primitive synthesis (synthesizer, validator,
             registry promotion, repair loop).
- Phase 7.9: Collaborative synthesis (proposal → approval → synthesis → re-route),
             primitive composition (body-only normalization, novel metrics, dynamic
             registry), claims-filter synthesis pipeline.

Migrated from: primitive_synthesis_probe.py, collaborative_synthesis_probe.py,
               primitive_composition_probe.py, claims_filter_synthesis_probe.py.
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

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.primitive_synthesizer import (
    LLMSynthesizer,
    SmokeTestSynthesizer,
    SynthesisResult,
    _handle_to_function_name,
)
from jeenom.primitive_validator import PrimitiveValidator, EUCLIDEAN_FIXTURES
from jeenom.schemas import OperatorIntent, SceneModel, SceneObject


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

EUCLIDEAN_BODY_ONLY = """\
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

CHEBYSHEV_CODE = """\
def grounding_closest_door_chebyshev_agent(scene, selector):
    doors = scene.find(
        object_type=selector.get('object_type', 'door'),
        color=selector.get('color'),
        exclude_colors=selector.get('exclude_colors') or [],
    )
    if not doors:
        return []
    return sorted(
        [
            (max(abs(d.x - scene.agent_x), abs(d.y - scene.agent_y)), d)
            for d in doors
        ],
        key=lambda pair: (pair[0], pair[1].color or ''),
    )
"""

WRONG_RETURN_TYPE_CODE = """\
def grounding_bad_return_agent(scene, selector):
    doors = scene.find(object_type='door')
    return {"doors": doors}
"""

DISALLOWED_IMPORT_CODE = """\
def grounding_closest_door_euclidean_agent(scene, selector):
    import numpy as np
    doors = scene.find(object_type='door')
    return []
"""


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session(*, env_id="MiniGrid-GoToDoor-8x8-v0", seed=42, max_loops=64) -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id=env_id,
        seed=seed,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
        max_loops=max_loops,
    )


def _run(fn):
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        return fn()


# ── Phase 7.8: Synthesis + Validation ────────────────────────────────────────

def _check_synthesis(checks: dict[str, bool]) -> None:
    validator = PrimitiveValidator()
    handle = "grounding.closest_door.euclidean.agent"
    fn_name = "grounding_closest_door_euclidean_agent"

    # 1. No substrate imports
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

    # 2. SmokeTestSynthesizer always refuses
    smoke_synth = SmokeTestSynthesizer()
    result = smoke_synth.synthesize(
        handle=handle,
        description="Euclidean distance ranking",
        consumes=("scene.door_candidates", "agent_pose"),
        produces=("grounded_target", "distance"),
    )
    checks["smoke_synthesizer_refuses"] = result.status == "refused"

    # 3. LLMSynthesizer falls back without API key
    llm_synth = LLMSynthesizer(api_key=None)
    checks["llm_synthesizer_has_fallback_reason"] = bool(llm_synth._fallback_reason)

    # 4. Validator passes correct Euclidean code
    ok_result = validator.validate(handle, fn_name, CORRECT_EUCLIDEAN_CODE)
    checks["validator_passes_correct_euclidean"] = ok_result.passed
    checks["validator_correct_has_no_failures"] = len(ok_result.failures) == 0

    # 5. Validator rejects wrong implementation
    bad_result = validator.validate(handle, fn_name, WRONG_EUCLIDEAN_CODE)
    checks["validator_rejects_manhattan_as_euclidean"] = not bad_result.passed

    # 6. Validator rejects disallowed import
    import_result = validator.validate(handle, fn_name, DISALLOWED_IMPORT_CODE)
    checks["validator_rejects_disallowed_import"] = not import_result.passed

    # 7. Registry register_synthesized promotes synthesizable → implemented
    reg = CapabilityRegistry.minigrid_default()
    spec_before = reg.lookup(handle)
    checks["spec_is_synthesizable_before"] = (
        spec_before is not None and spec_before.implementation_status == "synthesizable"
    )
    ok2 = validator.validate(handle, fn_name, CORRECT_EUCLIDEAN_CODE)
    registered = reg.register_synthesized(handle, ok2.compiled_fn)
    checks["register_synthesized_returns_true"] = registered
    spec_after = reg.lookup(handle)
    checks["spec_is_implemented_after"] = (
        spec_after is not None and spec_after.implementation_status == "implemented"
    )

    # 8. Body-only normalization
    def body_only_transport(payload):
        return {
            "function_name": fn_name,
            "function_body": EUCLIDEAN_BODY_ONLY,
            "description": "Euclidean grounding (body only).",
        }

    synth = LLMSynthesizer(api_key="fake", transport=body_only_transport)
    result = synth.synthesize(
        handle=handle,
        description="Euclidean distance",
        consumes=("scene.door_candidates", "agent_pose"),
        produces=("grounded_target", "distance"),
    )
    checks["body_only_synthesis_succeeds"] = result.status == "success"
    checks["body_only_has_def_line"] = result.code.lstrip("\n").startswith("def ")

    # 9. Wrong return type rejected
    bad_return = validator.validate(
        "grounding.bad_return.agent",
        "grounding_bad_return_agent",
        WRONG_RETURN_TYPE_CODE,
    )
    checks["wrong_return_type_rejected"] = not bad_return.passed

    # 10. Chebyshev (novel metric) validates
    cheb_result = validator.validate(
        "grounding.closest_door.chebyshev.agent",
        "grounding_closest_door_chebyshev_agent",
        CHEBYSHEV_CODE,
    )
    checks["chebyshev_validates"] = cheb_result.passed


# ── Phase 7.9: Collaborative Composition ────────────────────────────────────

def _check_collaborative(checks: dict[str, bool]) -> None:
    def _fake_transport(payload):
        return {
            "function_name": "grounding_closest_door_euclidean_agent",
            "function_body": CORRECT_EUCLIDEAN_CODE,
            "description": "Euclidean grounding synthesized.",
        }

    # 1. Synthesizable utterance → proposal
    session = _make_session()
    _run(lambda: session.handle_utterance("go to the red door"))
    response = _run(
        lambda: session.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["proposal_returned"] = "SYNTHESIS PROPOSAL" in response
    checks["proposal_mentions_handle"] = "grounding.closest_door.euclidean.agent" in response
    checks["pending_proposal_set"] = session.pending_synthesis_proposal is not None

    # 2. "no" → pending cleared
    no_response = _run(lambda: session.handle_utterance("no"))
    checks["no_clears_pending"] = session.pending_synthesis_proposal is None

    # 3. Proposal fires again; "yes" → synthesis runs
    session2 = _make_session()
    session2.synthesizer = LLMSynthesizer(api_key="fake", transport=_fake_transport)
    _run(lambda: session2.handle_utterance("go to the red door"))
    _run(
        lambda: session2.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["second_session_proposal_set"] = session2.pending_synthesis_proposal is not None
    yes_response = _run(lambda: session2.handle_utterance("yes"))
    checks["yes_clears_pending"] = session2.pending_synthesis_proposal is None
    checks["yes_synthesized_primitive"] = (
        session2.capability_registry.lookup(
            "grounding.closest_door.euclidean.agent"
        ).implementation_status == "implemented"
    )

    # 4. Second call uses synthesized primitive (no re-proposal)
    response2 = _run(
        lambda: session2.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["second_call_no_proposal"] = session2.pending_synthesis_proposal is None
    checks["second_call_succeeds"] = isinstance(response2, str) and len(response2) > 0

    # 5. Reset clears pending
    session4 = _make_session()
    _run(lambda: session4.handle_utterance("go to the red door"))
    _run(
        lambda: session4.handle_utterance("go to the closest door using euclidean distance")
    )
    session4.reset()
    checks["reset_clears_proposal"] = session4.pending_synthesis_proposal is None

    # 6. Golden path unaffected
    session_golden = _make_session()
    golden = _run(lambda: session_golden.handle_utterance("go to the red door"))
    checks["golden_path_no_proposal"] = session_golden.pending_synthesis_proposal is None
    checks["golden_path_still_works"] = (
        "RUN COMPLETE" in golden or "task_complete" in golden
    )


# ── Phase 7.9: Claims-Filter Synthesis ──────────────────────────────────────

def _check_claims_filter(checks: dict[str, bool]) -> None:
    class ThresholdSynthesizer:
        def synthesize(self, handle, description, consumes, produces,
                       existing_example=None, previous_code=None, validation_error=None):
            function_name = handle.replace(".", "_")
            code = (
                f"def {function_name}(entries, condition):\n"
                "    threshold = float(condition.get('threshold', 0))\n"
                "    comparison = condition.get('comparison', 'above')\n"
                "    if comparison == 'above':\n"
                "        return [e for e in entries if e.distance > threshold]\n"
                "    if comparison == 'at_least':\n"
                "        return [e for e in entries if e.distance >= threshold]\n"
                "    if comparison == 'below':\n"
                "        return [e for e in entries if e.distance < threshold]\n"
                "    if comparison in ('at_most', 'within'):\n"
                "        return [e for e in entries if e.distance <= threshold]\n"
                "    return []\n"
            )
            return SynthesisResult(
                handle=handle,
                function_name=function_name,
                code=code,
                status="success",
            )

    session = _make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8, max_loops=512)
    session.synthesizer = ThresholdSynthesizer()

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        ranked = session.handle_utterance("rank all the doors by manhattan distance")
        checks["ranked_claims_written"] = session.active_claims is not None

        proposal = session.command_from_operator_intent(
            OperatorIntent(
                intent_type="unsupported",
                capability_status="synthesizable",
                confidence=1.0,
                reason="Probe arbitration gap.",
                required_capabilities=["claims.filter.threshold.manhattan"],
            ),
            "go to the door where manhattan distance is above 7",
        )
        proposal_text = session.execute_command(proposal)
        checks["cf_proposal_created"] = proposal.kind == "synthesis_proposal"
        checks["cf_proposal_mentions_handle"] = (
            "claims.filter.threshold.manhattan" in proposal_text
        )
        checks["cf_proposal_preserves_condition"] = (
            session.pending_synthesis_proposal is not None
            and session.pending_synthesis_proposal.proposed_condition
            == {"threshold": 7.0, "comparison": "above", "metric": "manhattan"}
        )

        clarified = session.handle_utterance("yes")
        checks["cf_filter_registered"] = (
            session.capability_registry.lookup(
                "claims.filter.threshold.manhattan"
            ).implementation_status == "implemented"
        )
        checks["cf_multiple_matches_clarified"] = (
            "DOORS WITH MANHATTAN DISTANCE ABOVE 7.0" in clarified
            and "blue door@(12,3)" in clarified
            and "red door@(10,7)" in clarified
        )

        result = session.handle_utterance("red")
        checks["cf_selected_target_ran"] = "RUN COMPLETE" in result
        checks["cf_runtime_llm_zero"] = (
            session.last_result is not None
            and session.last_result["runtime_llm_calls_during_render"] == 0
        )
        checks["cf_cache_miss_zero"] = (
            session.last_result is not None
            and session.last_result["cache_miss_during_render"] == 0
        )


def main() -> int:
    checks: dict[str, bool] = {}

    print("CONSOLIDATED EVAL: PHASE 7.8 (Synthesis + Collaborative + Claims-Filter)\n")

    print("── Phase 7.8: Synthesis + Validation ──")
    _check_synthesis(checks)

    print("── Phase 7.9: Collaborative Composition ──")
    _check_collaborative(checks)

    print("── Phase 7.9: Claims-Filter Synthesis ──")
    _check_claims_filter(checks)

    print("\nCHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    n_pass = sum(checks.values())
    print(f"\n{n_pass}/{len(checks)} passed")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
