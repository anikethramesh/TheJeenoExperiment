from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jeenom.capability_registry import CapabilityRegistry
from jeenom.knowledge_base import KnowledgeBase
from jeenom.memory import OperationalMemory
from jeenom.readiness_graph import evaluate_request_plan
from jeenom.turn_orchestrator import KnowledgeChannel
from jeenom.schemas import (
    ClaimRecord,
    GroundedObjectEntry,
    KnowledgeSnapshot,
    RequestPlan,
    RequestPlanStep,
    SchemaValidationError,
    StationActiveClaims,
)


class TestPhase9ERepresentation(unittest.TestCase):
    def make_store(self):
        from jeenom.representation import RepresentationStore

        return RepresentationStore(
            memory=OperationalMemory(root=Path(tempfile.mkdtemp())),
            knowledge_channel=KnowledgeChannel(KnowledgeBase(storage_path=None)),
        )

    def test_claim_record_validates_kind(self):
        with self.assertRaises(SchemaValidationError):
            ClaimRecord(
                claim_id="bad",
                key="bad",
                value=True,
                kind="loose",
                status="confirmed",
                scope="grounding",
                authority="runtime",
                source="test",
            )

    def test_claims_round_trip_through_representation_store(self):
        store = self.make_store()
        store.put_claim(
            ClaimRecord(
                claim_id="fact:x",
                key="x",
                value=7,
                kind="fact",
                status="confirmed",
                scope="grounding",
                authority="runtime",
                source="test",
                provenance={"primitive": "unit"},
            )
        )
        self.assertEqual(store.get_claim("x").value, 7)
        self.assertEqual([claim.key for claim in store.query_claims(kind="fact")], ["x"])

    def test_put_claim_rejects_loose_dict(self):
        store = self.make_store()
        with self.assertRaises(TypeError):
            store.put_claim({"key": "x", "value": 1})  # type: ignore[arg-type]

    def test_procedure_and_snapshot_surface(self):
        store = self.make_store()
        store.put_procedure("bingo", "go to the red door")
        store.record_provenance({"event": "unit"})
        snapshot = store.snapshot()
        self.assertIsInstance(snapshot, KnowledgeSnapshot)
        self.assertIsNotNone(store.get_procedure("bingo"))
        self.assertTrue(snapshot.provenance)

    def test_readiness_can_use_knowledge_snapshot(self):
        store = self.make_store()
        claim = GroundedObjectEntry(
            color="red",
            x=2,
            y=2,
            distance=2,
            metric="manhattan",
        )
        store.set_active_claims(
            StationActiveClaims(
                scene_fingerprint=(1, 1, 0),
                ranked_scene_doors=[claim],
                last_grounded_target=claim,
                last_grounded_rank=0,
                last_grounding_query={"relation": "closest"},
                confidence=0.9,
            )
        )
        plan = RequestPlan(
            request_id="unit-readiness-snapshot",
            original_utterance="use claims",
            objective_type="query",
            objective_summary="Use active claims.",
            expected_response="answer_query",
            steps=[
                RequestPlanStep(
                    step_id="use_active_claims",
                    layer="claims",
                    operation="select",
                    scene_fingerprint_required=True,
                    constraints={"min_claim_confidence": 0.5},
                )
            ],
        )
        graph = evaluate_request_plan(
            plan,
            registry=CapabilityRegistry.minigrid_default(),
            knowledge_snapshot=store.snapshot(claims_valid=True),
        )
        self.assertEqual(graph.graph_status, "executable")


if __name__ == "__main__":
    unittest.main()
