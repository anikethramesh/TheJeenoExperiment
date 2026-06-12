"""ORPI conformance: compiled plans cannot reference deliberative meta primitives."""
from __future__ import annotations

from harness import emit_result


def main() -> int:
    from jeenom.capability_registry import CapabilityRegistry
    from jeenom.orpi import assert_no_deliberative_meta_plan_references
    from jeenom.schemas import (
        PrimitiveManifest,
        PrimitiveSpec,
        ReadinessGraph,
        ReadinessNode,
        RequestPlan,
        RequestPlanStep,
        SchemaValidationError,
    )

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}
    primitive = PrimitiveSpec(
        name="meta.repair_with_llm",
        primitive_type="meta",
        layer="cortex",
        description="Deliberative repair primitive.",
        mode="deliberative",
        cadence="deliberation",
        invariant_level="intent",
    )
    registry = CapabilityRegistry(PrimitiveManifest(name="probe", primitives=[primitive]))
    plan = RequestPlan(
        request_id="request:orpi-deliberative",
        original_utterance="repair it",
        objective_type="query",
        objective_summary="probe",
        steps=[
            RequestPlanStep(
                step_id="deliberative_step",
                layer="answer",
                operation="answer",
                required_handle="meta.repair_with_llm",
            )
        ],
        expected_response="answer_query",
    )
    try:
        assert_no_deliberative_meta_plan_references(plan, registry)
    except SchemaValidationError as exc:
        details["blocked_reason"] = str(exc)
        metrics["deliberative_meta_reference_is_rejected"] = True
    else:
        metrics["deliberative_meta_reference_is_rejected"] = False
    executable_graph = ReadinessGraph(
        request_id=plan.request_id,
        nodes=[
            ReadinessNode(
                step_id="deliberative_step",
                status="executable",
                layer="answer",
                operation="answer",
            )
        ],
        graph_status="executable",
        next_action="answer_query",
    )
    metrics["probe_plan_shape_valid"] = executable_graph.request_id == plan.request_id
    metrics["orpi_no_llm_in_loop_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="orpi_no_llm_in_loop_holds")


if __name__ == "__main__":
    raise SystemExit(main())
