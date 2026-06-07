"""Phase 9E probe: block boundaries use typed schemas."""
from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from typing import Any

from harness import emit_result


def main() -> int:
    import jeenom.schemas as schemas
    from jeenom.knowledge_base import KnowledgeBase
    from jeenom.memory import OperationalMemory
    from jeenom.readiness_graph import evaluate_request_plan

    try:
        from jeenom.representation import RepresentationStore
    except Exception:
        RepresentationStore = None  # type: ignore[assignment]

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    claim_cls = getattr(schemas, "ClaimRecord", None)
    snapshot_cls = getattr(schemas, "KnowledgeSnapshot", None)
    metrics["claim_record_schema_exists"] = claim_cls is not None
    metrics["knowledge_snapshot_schema_exists"] = snapshot_cls is not None

    required_claim_fields = {
        "claim_id",
        "key",
        "value",
        "kind",
        "status",
        "scope",
        "authority",
        "source",
        "confidence",
        "provenance",
        "freshness",
        "invalidation",
    }
    if claim_cls is not None:
        claim_fields = {field.name for field in fields(claim_cls)}
        metrics["claim_record_has_required_fields"] = required_claim_fields <= claim_fields
        details["claim_record_fields"] = sorted(claim_fields)
        try:
            claim_cls(
                claim_id="bad",
                key="x",
                value=True,
                kind="not-a-kind",
                status="confirmed",
                scope="grounding",
                authority="runtime",
                source="probe",
            )
        except Exception as exc:
            metrics["claim_record_rejects_unknown_kind"] = True
            details["unknown_kind_rejection"] = type(exc).__name__
        else:
            metrics["claim_record_rejects_unknown_kind"] = False
    else:
        metrics["claim_record_has_required_fields"] = False
        metrics["claim_record_rejects_unknown_kind"] = False

    metrics["representation_store_exists"] = RepresentationStore is not None
    if RepresentationStore is not None:
        store = RepresentationStore(
            memory=OperationalMemory(root=Path("/tmp/jeenom-phase9e-schema-probe")),
            knowledge_base=KnowledgeBase(storage_path=None),
        )
        try:
            store.put_claim({"key": "loose", "value": True})  # type: ignore[arg-type]
        except Exception as exc:
            metrics["put_claim_rejects_loose_dict"] = True
            details["put_claim_loose_dict_rejection"] = type(exc).__name__
        else:
            metrics["put_claim_rejects_loose_dict"] = False

        try:
            store.apply_memory_write_ticket({"writes": []})  # type: ignore[arg-type]
        except Exception as exc:
            metrics["memory_write_rejects_loose_dict"] = True
            details["memory_write_loose_dict_rejection"] = type(exc).__name__
        else:
            metrics["memory_write_rejects_loose_dict"] = False
    else:
        metrics["put_claim_rejects_loose_dict"] = False
        metrics["memory_write_rejects_loose_dict"] = False

    readiness_sig = inspect.signature(evaluate_request_plan)
    metrics["readiness_accepts_knowledge_snapshot"] = (
        "knowledge_snapshot" in readiness_sig.parameters
    )
    details["evaluate_request_plan_signature"] = str(readiness_sig)

    metrics["phase9e_schema_boundary_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase9e_schema_boundary_holds")


if __name__ == "__main__":
    raise SystemExit(main())
