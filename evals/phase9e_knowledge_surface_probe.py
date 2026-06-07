"""Phase 9E probe: one knowledge surface owns claims/procedures/provenance."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from harness import emit_result


def main() -> int:
    from jeenom.knowledge_base import KnowledgeBase
    from jeenom.memory import OperationalMemory
    from jeenom.schemas import ClaimRecord, KnowledgeSnapshot
    from jeenom.representation import RepresentationStore

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    root = Path(tempfile.mkdtemp())
    store = RepresentationStore(
        memory=OperationalMemory(root=root),
        knowledge_base=KnowledgeBase(storage_path=None),
    )

    records = [
        ClaimRecord(
            claim_id="fact:door-count",
            key="door_count",
            value=4,
            kind="fact",
            status="confirmed",
            scope="grounding",
            authority="runtime",
            source="phase9e_probe",
            provenance={"primitive": "grounding.all_doors"},
        ),
        ClaimRecord(
            claim_id="belief:path-risk",
            key="path_risk",
            value="low",
            kind="belief",
            status="inferred",
            scope="grounding",
            authority="runtime",
            source="phase9e_probe",
            confidence=0.7,
            provenance={"reason": "mock inference"},
        ),
        ClaimRecord(
            claim_id="hypothesis:target-hidden",
            key="target_hidden",
            value=True,
            kind="hypothesis",
            status="hypothesis",
            scope="grounding",
            authority="runtime",
            source="phase9e_probe",
            confidence=0.4,
            provenance={"reason": "not visible yet"},
        ),
    ]
    for record in records:
        store.put_claim(record)

    metrics["claim_roundtrip_fact"] = store.get_claim("door_count").value == 4
    metrics["belief_query_roundtrip"] = [
        claim.key for claim in store.query_claims(kind="belief")
    ] == ["path_risk"]
    metrics["hypothesis_query_roundtrip"] = [
        claim.key for claim in store.query_claims(kind="hypothesis")
    ] == ["target_hidden"]

    store.invalidate_claims(scope="grounding", reason="scene changed")
    invalidated = store.get_claim("door_count")
    metrics["invalidate_claims_marks_freshness"] = (
        invalidated is not None
        and invalidated.status == "invalidated"
        and invalidated.freshness == "stale"
    )

    concept = store.put_procedure(
        name="bingo",
        utterance="go to the red door",
        provenance={"source": "operator"},
    )
    metrics["procedure_write_roundtrip"] = (
        concept.name == "bingo"
        and store.get_procedure("bingo") is not None
    )

    store.record_provenance({"event": "probe", "source": "phase9e"})
    snapshot = store.snapshot()
    metrics["snapshot_is_typed"] = isinstance(snapshot, KnowledgeSnapshot)
    metrics["snapshot_contains_claims"] = "door_count" in snapshot.claims
    metrics["snapshot_contains_provenance"] = bool(snapshot.provenance)
    details["snapshot_claim_keys"] = sorted(snapshot.claims.keys())

    metrics["phase9e_knowledge_surface_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase9e_knowledge_surface_holds")


if __name__ == "__main__":
    raise SystemExit(main())
