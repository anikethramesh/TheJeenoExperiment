"""Phase 9B probe: substrate primitive contracts gate readiness.

This is the foundation for JEENOM as retrofit cognition. Primitive handles must
carry enough contract metadata for ReadinessGraph to block unsafe or stale
execution before a robot/simulator adapter ever actuates.
"""
from __future__ import annotations

from harness import emit_result

from jeenom.capability_registry import CapabilityRegistry
from jeenom.plan_reuse import PlanReuseCache
from jeenom.readiness_graph import evaluate_request_plan
from jeenom.schemas import (
    EnvironmentIdentity,
    GroundedObjectEntry,
    PrimitiveManifest,
    RequestPlan,
    RequestPlanStep,
    SchemaValidationError,
    StationActiveClaims,
)


def _contract_spec(
    name: str,
    *,
    safety_class: str = "query",
    authority_level: str = "none",
    validation_hooks: list[str] | None = None,
    required_frames: list[str] | None = None,
) -> dict[str, object]:
    return {
        "name": name,
        "primitive_type": "action",
        "layer": "action",
        "description": f"Test primitive {name}.",
        "inputs": ["robot.pose"],
        "outputs": ["robot.pose"],
        "side_effects": ["moves_robot"],
        "implementation_status": "implemented",
        "safe_to_synthesize": False,
        "runtime_binding": {"kind": "test", "value": name},
        "preconditions": ["robot.pose is fresh"],
        "postconditions": ["robot pose may change"],
        "required_claims": ["robot.pose"],
        "produced_claims": ["robot.pose"],
        "units": {"distance": "m", "angle": "rad"},
        "frame_id": "map",
        "required_frames": required_frames or ["map"],
        "safety_class": safety_class,
        "authority_level": authority_level,
        "failure_modes": ["blocked", "controller_fault"],
        "validation_hooks": validation_hooks if validation_hooks is not None else ["shadow_validate"],
        "substrate_fingerprint": "robot-stack-a",
    }


def _registry(*specs: dict[str, object]) -> CapabilityRegistry:
    return CapabilityRegistry(
        PrimitiveManifest.from_dict(
            {
                "name": "phase9b_contract_manifest",
                "primitives": list(specs),
            }
        )
    )


def _single_step_plan(handle: str, **constraints: object) -> RequestPlan:
    return RequestPlan(
        request_id=f"phase9b:{handle}",
        original_utterance="phase9b contract probe",
        objective_type="task",
        objective_summary="Probe substrate contract readiness.",
        steps=[
            RequestPlanStep(
                step_id="execute",
                layer="action",
                operation="execute",
                required_handle=handle,
                constraints=dict(constraints),
                scene_fingerprint_required=bool(constraints.get("requires_claims")),
            )
        ],
        expected_response="execute_task",
    )


def _claims(*, confidence: float = 1.0, frame_id: str = "map") -> StationActiveClaims:
    entry = GroundedObjectEntry(color="red", object_type="door", x=1, y=1, distance=1)
    return StationActiveClaims(
        scene_fingerprint=(0, 0, 0),
        ranked_scene_doors=[entry],
        last_grounded_target=entry,
        last_grounded_rank=0,
        last_grounding_query={"probe": "phase9b"},
        confidence=confidence,
        frame_id=frame_id,
    )


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}

    valid_registry = _registry(_contract_spec("action.safe_move"))
    safe_spec = valid_registry.lookup("action.safe_move")
    metrics["manifest_accepts_contract_metadata"] = (
        safe_spec is not None
        and safe_spec.safety_class == "query"
        and safe_spec.authority_level == "none"
        and safe_spec.required_claims == ["robot.pose"]
        and safe_spec.validation_hooks == ["shadow_validate"]
    )

    try:
        _registry(_contract_spec("action.bad_contract", safety_class="impossible"))
    except SchemaValidationError:
        metrics["invalid_safety_class_rejected"] = True
    else:
        metrics["invalid_safety_class_rejected"] = False

    no_validator_registry = _registry(
        _contract_spec(
            "action.no_validator",
            safety_class="actuation",
            validation_hooks=[],
        )
    )
    no_validator_graph = evaluate_request_plan(
        _single_step_plan("action.no_validator"),
        registry=no_validator_registry,
    )
    metrics["actuation_without_validation_blocks"] = (
        no_validator_graph.graph_status == "validation_required"
    )
    details["no_validator_status"] = no_validator_graph.graph_status

    restricted_registry = _registry(
        _contract_spec(
            "action.restricted_move",
            safety_class="actuation",
            authority_level="restricted",
        )
    )
    restricted_graph = evaluate_request_plan(
        _single_step_plan("action.restricted_move"),
        registry=restricted_registry,
    )
    metrics["restricted_without_authority_blocks"] = (
        restricted_graph.graph_status == "needs_authorization"
    )
    details["restricted_status"] = restricted_graph.graph_status

    low_conf_graph = evaluate_request_plan(
        _single_step_plan(
            "action.safe_move",
            requires_claims=True,
            min_claim_confidence=0.8,
        ),
        registry=valid_registry,
        active_claims=_claims(confidence=0.2, frame_id="map"),
        claims_valid=True,
    )
    metrics["low_confidence_claim_blocks"] = (
        low_conf_graph.graph_status == "claim_contract_failed"
    )
    details["low_confidence_status"] = low_conf_graph.graph_status

    frame_mismatch_graph = evaluate_request_plan(
        _single_step_plan(
            "action.safe_move",
            requires_claims=True,
            required_frame_id="map",
        ),
        registry=valid_registry,
        active_claims=_claims(confidence=1.0, frame_id="camera"),
        claims_valid=True,
    )
    metrics["frame_mismatch_claim_blocks"] = (
        frame_mismatch_graph.graph_status == "claim_contract_failed"
    )
    details["frame_mismatch_status"] = frame_mismatch_graph.graph_status

    cache = PlanReuseCache()
    plan = _single_step_plan("action.safe_move")
    plan.environment_assumptions = [
        assumption
        for assumption in plan.environment_assumptions
    ]
    from jeenom.request_planner import build_environment_assumptions

    identity_a = EnvironmentIdentity(
        env_id="robot-sim",
        substrate_fingerprint="robot-stack-a",
    )
    identity_b = EnvironmentIdentity(
        env_id="robot-sim",
        substrate_fingerprint="robot-stack-b",
    )
    plan.environment_assumptions = build_environment_assumptions(identity_a)
    plan.steps[0].environment_assumption_ids = [
        assumption.assumption_id for assumption in plan.environment_assumptions
    ]
    entry = cache.store(plan)
    verdict = cache.can_reuse(entry, valid_registry, identity_b)
    metrics["substrate_fingerprint_change_invalidates_reuse"] = (
        verdict.verdict == "recompile"
        and "env.substrate_fingerprint" in verdict.blocking_assumption_ids
    )
    details["reuse_verdict"] = verdict.__dict__

    metrics["substrate_contract_foundation_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="substrate_contract_foundation_holds")


if __name__ == "__main__":
    raise SystemExit(main())
