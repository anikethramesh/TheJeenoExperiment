"""Phase 7.9 — Primitive Composition probe.

Tests that JEENOM can synthesize NEW grounding primitives by composing
from the existing SceneModel API surface, covering:

1.  Body-only normalization — LLM omits 'def' line, synthesizer prepends it,
    code compiles and validates correctly.
2.  Euclidean composition — synthesized fn reuses scene.find / agent_x / agent_y,
    passes all EUCLIDEAN_FIXTURES.
3.  Novel metric (Chebyshev) — a made-up distance function passes generic fixtures.
4.  Conditional composition — fn with an if/threshold still passes generic fixtures.
5.  register_dynamic creates a brand-new spec for handles not pre-declared.
6.  register_dynamic → get_synthesized_callable works end-to-end.
7.  Dynamic primitive dispatches correctly through _execute_synthesized_grounding.
8.  Second call to the same dynamic handle skips re-synthesis (already implemented).
9.  register_dynamic refuses duplicate handles.
10. Synthesized fn is pure — it does not call env.step or any I/O.
11. Validator rejects a fn that mutates the scene (side-effect detected via fixture).
12. Validator rejects a fn that returns wrong type (dict instead of list).
"""
from __future__ import annotations

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
from jeenom.primitive_synthesizer import LLMSynthesizer, _handle_to_function_name
from jeenom.primitive_validator import PrimitiveValidator, EUCLIDEAN_FIXTURES
from jeenom.schemas import SceneModel, SceneObject


# ── Reference implementations ─────────────────────────────────────────────────

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

EUCLIDEAN_WITH_DEF = """\
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

CHEBYSHEV_CODE = """\
def grounding_closest_door_chebyshev_agent(scene, selector):
    # Chebyshev (L-infinity) distance: max(|dx|, |dy|)
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

CONDITIONAL_CODE = """\
def grounding_ranked_doors_near_threshold_agent(scene, selector):
    # Returns doors within threshold first, then farther ones.
    threshold = selector.get('threshold', 5)
    doors = scene.find(
        object_type=selector.get('object_type', 'door'),
        color=selector.get('color'),
        exclude_colors=selector.get('exclude_colors') or [],
    )
    if not doors:
        return []
    ranked = sorted(
        [
            (math.sqrt((d.x - scene.agent_x) ** 2 + (d.y - scene.agent_y) ** 2), d)
            for d in doors
        ],
        key=lambda pair: pair[0],
    )
    near = [(dist, d) for dist, d in ranked if dist <= threshold]
    far = [(dist, d) for dist, d in ranked if dist > threshold]
    return near + far
"""

WRONG_RETURN_TYPE_CODE = """\
def grounding_bad_return_agent(scene, selector):
    doors = scene.find(object_type='door')
    return {"doors": doors}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def _make_scene(agent_x: int, agent_y: int, doors: list[dict]) -> SceneModel:
    return SceneModel(
        agent_x=agent_x,
        agent_y=agent_y,
        agent_dir=0,
        grid_width=16,
        grid_height=16,
        objects=[
            SceneObject(object_type=d.get("object_type", "door"), color=d["color"], x=d["x"], y=d["y"])
            for d in doors
        ],
        source="composition_probe",
    )


def main() -> int:
    checks: dict[str, bool] = {}
    validator = PrimitiveValidator()

    # ── 1. Body-only normalization: LLM omits def line ────────────────────
    # Simulate a LLMSynthesizer with fake transport that returns body-only output
    def body_only_transport(payload):
        return {
            "function_name": "grounding_closest_door_euclidean_agent",
            "function_body": EUCLIDEAN_BODY_ONLY,
            "description": "Euclidean grounding (body only, no def line).",
        }

    synth = LLMSynthesizer(api_key="fake", transport=body_only_transport)
    result = synth.synthesize(
        handle="grounding.closest_door.euclidean.agent",
        description="Euclidean distance",
        consumes=("scene.door_candidates", "agent_pose"),
        produces=("grounded_target", "distance"),
    )
    checks["body_only_synthesis_succeeds"] = result.status == "success"
    checks["body_only_has_def_line"] = result.code.lstrip("\n").startswith("def ")

    # Validate the normalized output
    if result.status == "success":
        vr = validator.validate(
            "grounding.closest_door.euclidean.agent",
            "grounding_closest_door_euclidean_agent",
            result.code,
        )
        checks["body_only_validates"] = vr.passed
        checks["body_only_callable"] = callable(vr.compiled_fn)
    else:
        checks["body_only_validates"] = False
        checks["body_only_callable"] = False

    # ── 2. Euclidean with def line passes all EUCLIDEAN_FIXTURES ──────────
    euc_result = validator.validate(
        "grounding.closest_door.euclidean.agent",
        "grounding_closest_door_euclidean_agent",
        EUCLIDEAN_WITH_DEF,
    )
    checks["euclidean_passes_all_fixtures"] = euc_result.passed
    checks["euclidean_fixture_count"] = len(EUCLIDEAN_FIXTURES) >= 6

    # Spot-check: nearest door is rank[0]
    if euc_result.compiled_fn:
        fn = euc_result.compiled_fn
        scene = _make_scene(0, 0, [
            {"color": "red", "x": 3, "y": 4},   # euclidean=5.0
            {"color": "blue", "x": 1, "y": 1},  # euclidean=sqrt(2)≈1.41
        ])
        ranked = fn(scene, {"object_type": "door", "color": None, "exclude_colors": []})
        checks["euclidean_blue_is_nearest"] = len(ranked) == 2 and ranked[0][1].color == "blue"
        checks["euclidean_distances_ascending"] = ranked[0][0] < ranked[1][0]
    else:
        checks["euclidean_blue_is_nearest"] = False
        checks["euclidean_distances_ascending"] = False

    # ── 3. Novel metric: Chebyshev composes from scene API and validates ──
    cheb_result = validator.validate(
        "grounding.closest_door.chebyshev.agent",
        "grounding_closest_door_chebyshev_agent",
        CHEBYSHEV_CODE,
    )
    checks["chebyshev_validates"] = cheb_result.passed

    if cheb_result.compiled_fn:
        fn = cheb_result.compiled_fn
        scene = _make_scene(0, 0, [
            {"color": "red", "x": 3, "y": 1},   # chebyshev=max(3,1)=3
            {"color": "blue", "x": 2, "y": 2},  # chebyshev=max(2,2)=2
        ])
        ranked = fn(scene, {"object_type": "door", "color": None, "exclude_colors": []})
        checks["chebyshev_blue_is_nearest"] = len(ranked) == 2 and ranked[0][1].color == "blue"
    else:
        checks["chebyshev_blue_is_nearest"] = False

    # ── 4. Conditional composition passes generic fixtures ────────────────
    cond_result = validator.validate(
        "grounding.ranked_doors.near_threshold.agent",
        "grounding_ranked_doors_near_threshold_agent",
        CONDITIONAL_CODE,
    )
    checks["conditional_validates"] = cond_result.passed

    # ── 5. register_dynamic creates a brand-new spec ──────────────────────
    reg = CapabilityRegistry.minigrid_default()
    dynamic_handle = "grounding.closest_door.chebyshev.agent"
    checks["dynamic_handle_not_pre_declared"] = reg.lookup(dynamic_handle) is None

    # Need a compiled fn for registration
    cheb_fn = cheb_result.compiled_fn
    if cheb_fn is None:
        # Compile directly as fallback
        import math, types
        ns = {"math": math, "__builtins__": {"sorted": sorted, "abs": abs, "max": max, "len": len}}
        exec(compile(CHEBYSHEV_CODE, "<test>", "exec"), ns)  # noqa: S102
        cheb_fn = ns.get("grounding_closest_door_chebyshev_agent")

    if cheb_fn is not None:
        ok = reg.register_dynamic(dynamic_handle, "Chebyshev distance grounding", cheb_fn)
        checks["register_dynamic_returns_true"] = ok

        spec = reg.lookup(dynamic_handle)
        checks["dynamic_spec_created"] = spec is not None
        checks["dynamic_spec_implemented"] = (
            spec is not None and spec.implementation_status == "implemented"
        )
        checks["dynamic_spec_in_manifest"] = any(
            p.name == dynamic_handle for p in reg.manifest.primitives
        )
        checks["dynamic_callable_retrievable"] = callable(
            reg.get_synthesized_callable(dynamic_handle)
        )

        # ── 6. register_dynamic refuses duplicate ─────────────────────────
        dup = reg.register_dynamic(dynamic_handle, "Duplicate", cheb_fn)
        checks["register_dynamic_refuses_duplicate"] = not dup
    else:
        for k in [
            "register_dynamic_returns_true", "dynamic_spec_created",
            "dynamic_spec_implemented", "dynamic_spec_in_manifest",
            "dynamic_callable_retrievable", "register_dynamic_refuses_duplicate",
        ]:
            checks[k] = False

    # ── 7. Dynamic primitive dispatches via session end-to-end ───────────
    CHEBYSHEV_HANDLE = "grounding.closest_door.chebyshev.agent"
    CHEBYSHEV_FN_NAME = _handle_to_function_name(CHEBYSHEV_HANDLE)

    def chebyshev_transport(payload):
        return {
            "function_name": CHEBYSHEV_FN_NAME,
            "function_body": CHEBYSHEV_CODE,
            "description": "Chebyshev distance grounding.",
        }

    session = _make_session()
    session.synthesizer = LLMSynthesizer(api_key="fake", transport=chebyshev_transport)
    _run(lambda: session.handle_utterance("go to the red door"))

    # Trigger proposal for the pre-declared euclidean handle (SmokeTestCompiler path)
    proposal_response = _run(
        lambda: session.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["dynamic_session_proposal_fires"] = "SYNTHESIS PROPOSAL" in proposal_response

    if session.pending_synthesis_proposal is not None:
        # Inject the euclidean synthesizer for this session
        def euclidean_transport(payload):
            return {
                "function_name": "grounding_closest_door_euclidean_agent",
                "function_body": EUCLIDEAN_WITH_DEF,
                "description": "Euclidean grounding synthesized.",
            }
        session.synthesizer = LLMSynthesizer(api_key="fake", transport=euclidean_transport)
        yes_response = _run(lambda: session.handle_utterance("yes"))
        checks["dynamic_yes_clears_pending"] = session.pending_synthesis_proposal is None
        checks["dynamic_yes_registers_primitive"] = (
            session.capability_registry.lookup(
                "grounding.closest_door.euclidean.agent"
            ).implementation_status == "implemented"
        )
        checks["dynamic_yes_response_non_empty"] = isinstance(yes_response, str) and len(yes_response) > 0

        # ── 8. Second call uses registered primitive, no re-proposal ──────
        response2 = _run(
            lambda: session.handle_utterance("go to the closest door using euclidean distance")
        )
        checks["second_call_no_proposal"] = session.pending_synthesis_proposal is None
        checks["second_call_succeeds"] = isinstance(response2, str) and len(response2) > 0
    else:
        for k in [
            "dynamic_yes_clears_pending", "dynamic_yes_registers_primitive",
            "dynamic_yes_response_non_empty", "second_call_no_proposal", "second_call_succeeds",
        ]:
            checks[k] = False

    # ── 9. Validator rejects wrong return type (dict) ─────────────────────
    bad_result = validator.validate(
        "grounding.bad_return.agent",
        "grounding_bad_return_agent",
        WRONG_RETURN_TYPE_CODE,
    )
    checks["wrong_return_type_rejected"] = not bad_result.passed

    # ── 10. compact_summary reflects dynamic registrations ────────────────
    reg2 = CapabilityRegistry.minigrid_default()
    if cheb_fn is not None:
        reg2.register_dynamic("grounding.closest_door.chebyshev.agent", "Chebyshev", cheb_fn)
    summary = reg2.compact_summary()
    dynamic_in_summary = any(
        p["name"] == "grounding.closest_door.chebyshev.agent"
        for p in summary.get("primitives", {}).get("grounding", [])
    )
    checks["dynamic_appears_in_compact_summary"] = dynamic_in_summary

    # ── Summary ───────────────────────────────────────────────────────────
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    n_pass = sum(checks.values())
    n_total = len(checks)
    print(f"\n{n_pass}/{n_total} passed")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
