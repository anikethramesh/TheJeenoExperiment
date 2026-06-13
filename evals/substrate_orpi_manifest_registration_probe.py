"""ORPI conformance: MiniGrid registers a manifest at adapter/runtime init.

The permissive-window-is-never-live guarantee requires that the manifest exists on
the adapter *before* any other method is called on it.  We verify this by inspecting
the adapter's internal state immediately after construction, without calling
orpi_manifest() first.
"""
from __future__ import annotations

from harness import emit_result


def main() -> int:
    from jeenom.minigrid_operational_context import MiniGridOperationalContext
    from jeenom.minigrid_substrate_adapter import MiniGridSubstrateAdapter
    from jeenom.minigrid_runtime_package import build_minigrid_runtime_package
    from jeenom.orpi import OrpiManifest, OrpiProcedure
    from jeenom.readiness_graph import evaluate_request_plan
    from jeenom.schemas import (
        GroundedObjectEntry,
        RequestPlan,
        RequestPlanStep,
        SchemaValidationError,
        StationActiveClaims,
    )

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}

    # Check the adapter directly: manifest must be present before any method call.
    fresh_adapter = MiniGridSubstrateAdapter(
        env_id="MiniGrid-GoToDoor-8x8-v0",
        render_mode="none",
    )
    manifest_at_init = fresh_adapter._orpi_manifest  # noqa: SLF001
    metrics["manifest_exists_immediately_after_adapter_init"] = isinstance(
        manifest_at_init, OrpiManifest
    )

    # Check the runtime package path.
    package = build_minigrid_runtime_package(
        env_id="MiniGrid-GoToDoor-8x8-v0",
        render_mode="none",
    )
    adapter_manifest = package.substrate.orpi_manifest()
    runtime_manifest = package.resolve_orpi_manifest()
    payload = runtime_manifest.as_dict()
    details["manifest_keys"] = sorted(payload)
    details["symbol_mapping_keys"] = sorted(payload["symbol_mappings"])
    metrics["adapter_exposes_orpi_manifest_method"] = isinstance(adapter_manifest, OrpiManifest)
    metrics["runtime_package_caches_registered_manifest"] = runtime_manifest is adapter_manifest
    metrics["manifest_has_substrate_identity"] = (
        payload["substrate_id"] == "minigrid"
        and bool(payload["substrate_fingerprint"])
        and payload["orpi_version"] == "0.1"
    )
    metrics["manifest_has_symbol_frames_units_risk_policy"] = (
        bool(payload["symbol_mappings"].get("object_index"))
        and bool(payload["symbol_mappings"].get("color_index"))
        and "grid" in payload["frames"]
        and bool(payload["units"])
        and "actuation" in payload["risk_policy"]
    )
    low_conf_target = GroundedObjectEntry(
        color="red",
        x=1,
        y=1,
        distance=1,
        object_type="door",
    )
    low_conf_claims = StationActiveClaims(
        scene_fingerprint=(0, 0, 0),
        ranked_scene_doors=[low_conf_target],
        last_grounded_target=low_conf_target,
        last_grounded_rank=0,
        last_grounding_query={},
        confidence=0.1,
    )
    risk_plan = RequestPlan(
        request_id="request:risk-policy",
        original_utterance="turn left",
        objective_type="control",
        objective_summary="probe manifest risk policy",
        steps=[
            RequestPlanStep(
                step_id="turn_left",
                layer="action",
                operation="execute",
                required_handle="action.turn_left",
                scene_fingerprint_required=True,
            )
        ],
        expected_response="execute_motor",
    )
    risk_graph = evaluate_request_plan(
        risk_plan,
        registry=package.resolve_capability_registry(),
        active_claims=low_conf_claims,
        claims_valid=True,
        risk_policy=runtime_manifest.risk_policy,
    )
    metrics["readiness_reads_claim_thresholds_from_manifest_risk_policy"] = (
        risk_graph.graph_status == "claim_contract_failed"
    )
    metrics["manifest_has_bundled_procedures_field"] = (
        "bundled_procedures" in payload
        and payload["bundled_procedures"] == []
    )
    oem_turn = OrpiProcedure.from_recipe(
        name="procedure.oem.turn_left_once",
        recipe={"steps": ["action.turn_left"]},
        registry=package.resolve_capability_registry(),
        declared_postconditions=["agent direction changes"],
        declared_preconditions=["operator_authorized"],
        provenance="oem",
        substrate_fingerprint=runtime_manifest.substrate_fingerprint,
    )
    synthetic_manifest = OrpiManifest(
        substrate_id=runtime_manifest.substrate_id,
        substrate_fingerprint=runtime_manifest.substrate_fingerprint,
        object_vocabulary=list(runtime_manifest.object_vocabulary),
        primitives=list(runtime_manifest.primitives),
        symbol_mappings=dict(runtime_manifest.symbol_mappings),
        frames=dict(runtime_manifest.frames),
        units=dict(runtime_manifest.units),
        risk_policy=dict(runtime_manifest.risk_policy),
        bundled_procedures=[oem_turn],
    )
    procedure_candidates = package.resolve_capability_registry().candidates_for_postcondition(
        "agent direction changes",
        orpi_manifest=synthetic_manifest,
    )
    metrics["orpi_procedure_selectable_by_postcondition"] = any(
        candidate["kind"] == "procedure"
        and candidate["name"] == "procedure.oem.turn_left_once"
        and candidate["provenance"] == "oem"
        for candidate in procedure_candidates
    )
    metrics["orpi_procedure_provenance_is_typed"] = oem_turn.provenance == "oem"
    try:
        OrpiManifest(
            substrate_id=runtime_manifest.substrate_id,
            substrate_fingerprint=runtime_manifest.substrate_fingerprint,
            object_vocabulary=list(runtime_manifest.object_vocabulary),
            primitives=list(runtime_manifest.primitives),
            bundled_procedures=[
                OrpiProcedure.from_recipe(
                    name="procedure.synthesized.bad",
                    recipe={"steps": ["action.turn_right"]},
                    registry=package.resolve_capability_registry(),
                    declared_postconditions=["agent direction changes"],
                    provenance="synthesized",
                )
            ],
        )
    except SchemaValidationError as exc:
        details["synthesized_bundle_rejection"] = str(exc)
        metrics["synthesized_procedure_cannot_be_bundled"] = True
    else:
        metrics["synthesized_procedure_cannot_be_bundled"] = False
    metrics["orpi_manifest_registration_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="orpi_manifest_registration_holds")


if __name__ == "__main__":
    raise SystemExit(main())
