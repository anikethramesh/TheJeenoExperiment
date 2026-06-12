"""Phase 9E: block boundaries, schema enforcement, knowledge surface, readiness snapshots."""
from __future__ import annotations

import ast
import inspect
import tempfile
from dataclasses import fields
from pathlib import Path
from typing import Any

from harness import ROOT, emit_result, make_session


# ── Block-boundary AST helpers ────────────────────────────────────────────────

ALLOWED_MEMORY_MUTATION_FILES = {
    "jeenom/memory.py",
    "jeenom/representation.py",
}


def _py_files() -> list[Path]:
    return sorted((ROOT / "jeenom").glob("*.py"))


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _is_memory_attr(node: ast.AST, attr: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attr
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "memory"
    )


def _external_memory_stores() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in _py_files():
        rel = _rel(path)
        if rel in ALLOWED_MEMORY_MUTATION_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets: list[ast.AST]
                if isinstance(node, ast.Assign):
                    targets = list(node.targets)
                else:
                    targets = [node.target]
                for target in targets:
                    if isinstance(target, ast.Subscript) and (
                        _is_memory_attr(target.value, "knowledge")
                        or _is_memory_attr(target.value, "episodic_memory")
                    ):
                        hits.append((rel, target.lineno, ast.unparse(target)))
                    if _is_memory_attr(target, "scene_model"):
                        hits.append((rel, target.lineno, ast.unparse(target)))
    return hits


def _external_update_knowledge_calls() -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for path in _py_files():
        rel = _rel(path)
        if rel in ALLOWED_MEMORY_MUTATION_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update_knowledge"
            ):
                hits.append((rel, node.lineno, ast.unparse(node.func)))
    return hits


# ── Readiness snapshot helpers ────────────────────────────────────────────────

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
    from jeenom.schemas import GroundedObjectEntry, StationActiveClaims

    return StationActiveClaims(
        scene_fingerprint=(1, 1, 0),
        ranked_scene_doors=[
            GroundedObjectEntry(color="red", x=2, y=2, distance=2, metric="manhattan")
        ],
        last_grounded_target=GroundedObjectEntry(
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
    import jeenom.schemas as schemas

    try:
        from jeenom.representation import RepresentationStore
    except Exception as exc:
        RepresentationStore = None  # type: ignore[assignment]

    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # ── Block boundary (9E gate 1) ────────────────────────────────────────────
    metrics["representation_store_exists"] = RepresentationStore is not None
    if RepresentationStore is not None:
        session = make_session()
        metrics["operator_station_has_representation_store"] = isinstance(
            getattr(session, "representation", None), RepresentationStore
        )
        metrics["active_claims_is_property"] = isinstance(
            getattr(type(session), "active_claims", None), property
        )
    else:
        metrics["operator_station_has_representation_store"] = False
        metrics["active_claims_is_property"] = False

    memory_stores = _external_memory_stores()
    update_knowledge_calls = _external_update_knowledge_calls()
    metrics["no_external_memory_dict_or_scene_stores"] = not memory_stores
    metrics["no_external_durable_update_knowledge_calls"] = not update_knowledge_calls
    details["external_memory_stores"] = memory_stores
    details["external_update_knowledge_calls"] = update_knowledge_calls

    # ── Schema boundary (9E gate 2) ───────────────────────────────────────────
    claim_cls = getattr(schemas, "ClaimRecord", None)
    snapshot_cls = getattr(schemas, "KnowledgeSnapshot", None)
    metrics["claim_record_schema_exists"] = claim_cls is not None
    metrics["knowledge_snapshot_schema_exists"] = snapshot_cls is not None

    required_claim_fields = {
        "claim_id", "key", "value", "kind", "status", "scope",
        "authority", "source", "confidence", "provenance", "freshness", "invalidation",
    }
    if claim_cls is not None:
        claim_fields = {field.name for field in fields(claim_cls)}
        metrics["claim_record_has_required_fields"] = required_claim_fields <= claim_fields
        details["claim_record_fields"] = sorted(claim_fields)
        try:
            claim_cls(
                claim_id="bad", key="x", value=True, kind="not-a-kind",
                status="confirmed", scope="grounding", authority="runtime", source="probe",
            )
        except Exception as exc:
            metrics["claim_record_rejects_unknown_kind"] = True
            details["unknown_kind_rejection"] = type(exc).__name__
        else:
            metrics["claim_record_rejects_unknown_kind"] = False
    else:
        metrics["claim_record_has_required_fields"] = False
        metrics["claim_record_rejects_unknown_kind"] = False

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

    # ── Knowledge surface (9E gate 3) ─────────────────────────────────────────
    if RepresentationStore is not None:
        from jeenom.schemas import ClaimRecord, KnowledgeSnapshot
        root = Path(tempfile.mkdtemp())
        kstore = RepresentationStore(
            memory=OperationalMemory(root=root),
            knowledge_base=KnowledgeBase(storage_path=None),
        )
        records = [
            ClaimRecord(
                claim_id="fact:door-count", key="door_count", value=4, kind="fact",
                status="confirmed", scope="grounding", authority="runtime",
                source="phase9e_probe", provenance={"primitive": "grounding.all_doors"},
            ),
            ClaimRecord(
                claim_id="belief:path-risk", key="path_risk", value="low", kind="belief",
                status="inferred", scope="grounding", authority="runtime",
                source="phase9e_probe", confidence=0.7, provenance={"reason": "mock inference"},
            ),
            ClaimRecord(
                claim_id="hypothesis:target-hidden", key="target_hidden", value=True,
                kind="hypothesis", status="hypothesis", scope="grounding", authority="runtime",
                source="phase9e_probe", confidence=0.4, provenance={"reason": "not visible yet"},
            ),
        ]
        for record in records:
            kstore.put_claim(record)

        metrics["claim_roundtrip_fact"] = kstore.get_claim("door_count").value == 4
        metrics["belief_query_roundtrip"] = [
            c.key for c in kstore.query_claims(kind="belief")
        ] == ["path_risk"]
        metrics["hypothesis_query_roundtrip"] = [
            c.key for c in kstore.query_claims(kind="hypothesis")
        ] == ["target_hidden"]

        kstore.invalidate_claims(scope="grounding", reason="scene changed")
        invalidated = kstore.get_claim("door_count")
        metrics["invalidate_claims_marks_freshness"] = (
            invalidated is not None
            and invalidated.status == "invalidated"
            and invalidated.freshness == "stale"
        )

        concept = kstore.put_procedure(
            name="bingo", utterance="go to the red door", provenance={"source": "operator"}
        )
        metrics["procedure_write_roundtrip"] = (
            concept.name == "bingo" and kstore.get_procedure("bingo") is not None
        )

        kstore.record_provenance({"event": "probe", "source": "phase9e"})
        snapshot = kstore.snapshot()
        metrics["snapshot_is_typed"] = isinstance(snapshot, KnowledgeSnapshot)
        metrics["snapshot_contains_claims"] = "door_count" in snapshot.claims
        metrics["snapshot_contains_provenance"] = bool(snapshot.provenance)
        details["snapshot_claim_keys"] = sorted(snapshot.claims.keys())
    else:
        for k in ("claim_roundtrip_fact", "belief_query_roundtrip", "hypothesis_query_roundtrip",
                  "invalidate_claims_marks_freshness", "procedure_write_roundtrip",
                  "snapshot_is_typed", "snapshot_contains_claims", "snapshot_contains_provenance"):
            metrics[k] = False

    # ── Readiness snapshot (9E gate 4) ────────────────────────────────────────
    plan = _plan_requiring_claims()
    registry = CapabilityRegistry.minigrid_default()

    without_snapshot = evaluate_request_plan(plan, registry=registry)
    metrics["readiness_without_snapshot_blocks_stale_claims"] = (
        without_snapshot.graph_status == "stale_claims"
    )
    details["without_snapshot"] = without_snapshot.as_dict()

    if RepresentationStore is not None:
        rstore = RepresentationStore(
            memory=OperationalMemory(root=Path(tempfile.mkdtemp())),
            knowledge_base=KnowledgeBase(storage_path=None),
        )
        rstore.set_active_claims(_active_claims())
        snapshot = rstore.snapshot(claims_valid=True)
        with_snapshot = evaluate_request_plan(plan, registry=registry, knowledge_snapshot=snapshot)
        metrics["readiness_with_snapshot_is_executable"] = (
            with_snapshot.graph_status == "executable"
        )
        details["with_snapshot"] = with_snapshot.as_dict()
    else:
        metrics["readiness_with_snapshot_is_executable"] = False

    metrics["phase9e_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase9e_holds")


if __name__ == "__main__":
    raise SystemExit(main())
