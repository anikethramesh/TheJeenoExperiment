"""ORPI 12B: bundled procedures are selectable composite capabilities."""
from __future__ import annotations

from harness import emit_result


def main() -> int:
    from jeenom.command_authority import CommandAuthority
    from jeenom.minigrid_runtime_package import build_minigrid_runtime_package
    from jeenom.orpi import OrpiManifest, OrpiProcedure
    from jeenom.readiness_graph import evaluate_request_plan
    from jeenom.schemas import RequestPlan, RequestPlanStep

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}
    package = build_minigrid_runtime_package(
        env_id="MiniGrid-GoToDoor-8x8-v0",
        render_mode="none",
    )
    registry = package.resolve_capability_registry()
    base_manifest = package.resolve_orpi_manifest()
    procedure = OrpiProcedure.from_recipe(
        name="procedure.oem.turn_left_once",
        recipe={"steps": ["action.turn_left"]},
        registry=registry,
        declared_postconditions=["agent direction changes"],
        declared_preconditions=["active_claims"],
        provenance="oem",
        substrate_fingerprint=base_manifest.substrate_fingerprint,
    )
    manifest = OrpiManifest(
        substrate_id=base_manifest.substrate_id,
        substrate_fingerprint=base_manifest.substrate_fingerprint,
        object_vocabulary=list(base_manifest.object_vocabulary),
        primitives=list(base_manifest.primitives),
        symbol_mappings=dict(base_manifest.symbol_mappings),
        frames=dict(base_manifest.frames),
        units=dict(base_manifest.units),
        risk_policy=dict(base_manifest.risk_policy),
        bundled_procedures=[procedure],
    )

    candidates = registry.candidates_for_postcondition(
        "agent direction changes",
        orpi_manifest=manifest,
    )
    details["candidate_names"] = [candidate["name"] for candidate in candidates]
    metrics["postcondition_candidates_include_primitive_and_procedure"] = (
        any(candidate["kind"] == "primitive" for candidate in candidates)
        and any(candidate["kind"] == "procedure" for candidate in candidates)
    )

    expanded = registry.expand_procedure_candidate(
        "procedure.oem.turn_left_once",
        orpi_manifest=manifest,
    )
    metrics["selected_procedure_expands_to_primitive_handles"] = expanded == [
        "action.turn_left"
    ]

    selected_plan = RequestPlan(
        request_id="request:orpi-procedure-readiness",
        original_utterance="turn left through OEM bundle",
        objective_type="control",
        objective_summary="probe procedure readiness",
        steps=[
            RequestPlanStep(
                step_id="oem_turn_left",
                layer="action",
                operation="execute",
                required_handle=expanded[0],
                scene_fingerprint_required=True,
                constraints={
                    "candidate_kind": "procedure",
                    "candidate_name": procedure.name,
                    "candidate_provenance": procedure.provenance,
                },
            )
        ],
        expected_response="execute_motor",
    )
    blocked_graph = evaluate_request_plan(selected_plan, registry=registry)
    metrics["procedure_selection_does_not_bypass_readiness"] = (
        blocked_graph.graph_status == "stale_claims"
    )

    primitive_plan = RequestPlan(
        request_id="request:orpi-primitive-candidate",
        original_utterance="turn left primitive",
        objective_type="control",
        objective_summary="probe primitive candidate",
        steps=[
            RequestPlanStep(
                step_id="primitive_turn_left",
                layer="action",
                operation="execute",
                required_handle="action.turn_left",
                constraints={
                    "candidate_kind": "primitive",
                    "candidate_name": "action.turn_left",
                    "candidate_provenance": "registry",
                },
            ),
            RequestPlanStep(
                step_id="procedure_turn_left",
                layer="action",
                operation="execute",
                required_handle="action.turn_left",
                constraints={
                    "candidate_kind": "procedure",
                    "candidate_name": procedure.name,
                    "candidate_provenance": procedure.provenance,
                },
            ),
        ],
        expected_response="execute_motor",
    )
    graph = evaluate_request_plan(primitive_plan, registry=registry)
    command_result = CommandAuthority().record_result(
        "turn left",
        "PROBE",
        intent=None,
        plan=primitive_plan,
        graph=graph,
        tickets=(),
        compiler_name="probe",
    )
    episode_candidates = command_result.labelled_episode.as_dict()["plan"]["candidates"]
    details["episode_candidates"] = episode_candidates
    metrics["candidate_trace_preserves_only_provenance_difference"] = (
        {
            (candidate["kind"], candidate["name"], candidate["provenance"])
            for candidate in episode_candidates
        }
        == {
            ("primitive", "action.turn_left", "registry"),
            ("procedure", "procedure.oem.turn_left_once", "oem"),
        }
    )
    metrics["orpi_procedure_selection_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="orpi_procedure_selection_holds")


if __name__ == "__main__":
    raise SystemExit(main())
