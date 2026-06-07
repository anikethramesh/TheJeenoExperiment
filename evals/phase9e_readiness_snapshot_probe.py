"""Phase 9E probe: readiness consumes typed knowledge snapshots."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from harness import emit_result


def _plan_requiring_claims():
    from jeenom.schemas import RequestPlan, RequestPlanStep

    return RequestPlan(
        request_id="phase9e-readiness",
        original_utterance="pick another grounded target",
        objective_type="query",
        objective_summary="Use active grounding claims.",
        expected_response="answer_query",
        steps=[
            RequestPlanStep(
                step_id="use_active_claims",
                layer="claims",
                operation="select",
                required_handle=None,
                scene_fingerprint_required=True,
                constraints={"min_claim_confidence": 0.5},
                memory_reads=["active_claims.ranked_scene_doors"],
            )
        ],
    )


def _active_claims():
    from jeenom.schemas import GroundedDoorEntry, StationActiveClaims

    return StationActiveClaims(
        scene_fingerprint=(1, 1, 0),
        ranked_scene_doors=[
            GroundedDoorEntry(color="red", x=2, y=2, distance=2, metric="manhattan")
        ],
        last_grounded_target=GroundedDoorEntry(
            color="red", x=2, y=2, distance=2, metric="manhattan"
        ),
        last_grounded_rank=0,
        last_grounding_query={"relation": "closest"},
        confidence=0.9,
    )


def main() -> int:
    from jeenom.capability_registry import CapabilityRegistry
    from jeenom.knowledge_base import KnowledgeBase
    from jeenom.memory import OperationalMemory
    from jeenom.readiness_graph import evaluate_request_plan
    from jeenom.representation import RepresentationStore

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    plan = _plan_requiring_claims()
    registry = CapabilityRegistry.minigrid_default()

    without_snapshot = evaluate_request_plan(plan, registry=registry)
    metrics["readiness_without_snapshot_blocks_stale_claims"] = (
        without_snapshot.graph_status == "stale_claims"
    )
    details["without_snapshot"] = without_snapshot.as_dict()

    store = RepresentationStore(
        memory=OperationalMemory(root=Path(tempfile.mkdtemp())),
        knowledge_base=KnowledgeBase(storage_path=None),
    )
    store.set_active_claims(_active_claims())
    snapshot = store.snapshot(claims_valid=True)
    with_snapshot = evaluate_request_plan(
        plan,
        registry=registry,
        knowledge_snapshot=snapshot,
    )
    metrics["readiness_with_snapshot_is_executable"] = (
        with_snapshot.graph_status == "executable"
    )
    details["with_snapshot"] = with_snapshot.as_dict()

    metrics["phase9e_readiness_snapshot_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase9e_readiness_snapshot_holds")


if __name__ == "__main__":
    raise SystemExit(main())
