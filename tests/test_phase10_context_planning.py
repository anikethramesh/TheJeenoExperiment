from __future__ import annotations

import unittest

from jeenom.schemas import OperationalContext, OperatorIntent


class TestPhase10ContextPlanning(unittest.TestCase):
    def _token_context(self) -> OperationalContext:
        return OperationalContext(
            context_id="probe.tokens",
            substrate_id="probe",
            object_vocabulary=["token"],
            attribute_vocabulary=["name", "score"],
            grounding_semantics={
                "distance_metrics": ["score"],
                "distance_references": ["agent"],
                "ranked_claims_output": "active_claims.ranked_tokens",
                "capability_handles": {
                    "ranked": "grounding.all_{object_type_plural}.ranked.{metric}.agent",
                    "filter_threshold": "claims.filter.threshold.{metric}",
                    "unique": "grounding.unique_{object_type}.name_filter",
                    "task_go_to_object": "task.go_to_object.{object_type}",
                },
            },
            reference_semantics={"closest": {"default_metric": "score"}},
        )

    def test_request_plan_uses_injected_context_handles(self):
        from jeenom.planning_semantics import PlanningSemantics
        from jeenom.request_planner import build_request_plan

        semantics = PlanningSemantics(self._token_context())
        intent = OperatorIntent(
            intent_type="status_query",
            status_query="ground_target",
            grounding_query_plan={
                "operation": "rank",
                "object_type": "token",
                "metric": "score",
                "primitive_handle": None,
                "required_capabilities": [],
            },
            confidence=1.0,
            reason="Probe token ranking.",
        )

        plan = build_request_plan(
            "rank all tokens by score",
            intent,
            planning_semantics=semantics,
        )

        handles = [step.required_handle for step in plan.steps]
        self.assertIn("grounding.all_tokens.ranked.score.agent", handles)
        self.assertNotIn("grounding.all_doors.ranked.score.agent", repr(plan.as_dict()))

    def test_intent_verifier_uses_injected_context_handles(self):
        from jeenom.intent_verifier import IntentVerifier
        from jeenom.planning_semantics import PlanningSemantics

        semantics = PlanningSemantics(self._token_context())
        verifier = IntentVerifier(planning_semantics=semantics)
        intent = OperatorIntent(
            intent_type="status_query",
            status_query="ground_target",
            required_capabilities=[],
            confidence=1.0,
            reason="Probe token ranking.",
        )

        enriched, _ = verifier.enrich("rank all tokens by score", intent)

        self.assertIn(
            "grounding.all_tokens.ranked.score.agent",
            enriched.required_capabilities,
        )
        self.assertFalse(any("all_doors" in h for h in enriched.required_capabilities))

    def test_operator_station_owns_planning_semantics(self):
        from jeenom.operator_station import OperatorStationSession
        from jeenom.planning_semantics import PlanningSemantics

        session = OperatorStationSession(compiler_name="smoke_test", render_mode="none")

        self.assertIsInstance(session.planning_semantics, PlanningSemantics)
        self.assertIs(session.intent_verifier.planning_semantics, session.planning_semantics)


if __name__ == "__main__":
    unittest.main()
