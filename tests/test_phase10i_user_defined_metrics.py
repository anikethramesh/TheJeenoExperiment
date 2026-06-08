from __future__ import annotations

import unittest
from typing import Any

from evals.harness import make_session
from jeenom.capability_registry import CapabilityRegistry
from jeenom.readiness_graph import evaluate_request_plan
from jeenom import schemas
from jeenom.schemas import RequestPlan, RequestPlanStep


def _handles(session: Any) -> set[str]:
    return set(session.capability_registry.primitive_names())


def _plan_handles(session: Any) -> set[str]:
    if session.last_request_plan is None:
        return set()
    return {
        step.required_handle
        for step in session.last_request_plan.steps
        if step.required_handle is not None
    }


def _metric_supported(session: Any, metric: str) -> bool:
    return session.planning_semantics.metric_supported(metric)


class TestPhase10IUserDefinedMetrics(unittest.TestCase):
    def test_typed_primitive_definition_schema_exists(self):
        self.assertTrue(
            hasattr(schemas, "PrimitiveDefinitionRequest"),
            "10I needs a typed PrimitiveDefinitionRequest, not ad-hoc strings.",
        )

    def test_ramesian_metric_can_be_defined_approved_registered_and_used(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        proposal = session.handle_utterance(
            "define a new distance metric called ramesian as euclidean distance mod 5"
        )

        self.assertNotIn("I didn't understand", proposal)
        self.assertIn("ramesian", proposal.lower())
        self.assertIn("euclidean", proposal.lower())
        self.assertTrue(
            hasattr(session, "pending_primitive_definition")
            or session.pending_synthesis_proposal is not None,
            "The station must keep a pending typed definition/proposal before approval.",
        )

        session.handle_utterance("yes")

        expected_handle = "grounding.all_doors.ranked.ramesian.agent"
        self.assertIn(expected_handle, _handles(session))
        self.assertTrue(_metric_supported(session, "ramesian"))

        ranked = session.handle_utterance("rank all doors by ramesian")
        self.assertIn("DOORS RANKED", ranked.upper())
        self.assertIn("RAMESIAN", ranked.upper())
        self.assertIn(expected_handle, _plan_handles(session))

    def test_convenient_distance_composes_existing_metrics(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        proposal = session.handle_utterance(
            "synthesize a new distance metric which is the minimum between "
            "euclidean and manhattan distance. call it convenientDistance"
        )

        self.assertNotIn("I didn't understand", proposal)
        proposal_lower = proposal.lower()
        self.assertIn("convenient", proposal_lower)
        self.assertIn("euclidean", proposal_lower)
        self.assertIn("manhattan", proposal_lower)
        self.assertTrue("minimum" in proposal_lower or "min(" in proposal_lower)

        session.handle_utterance("yes")

        expected_handle = "grounding.all_doors.ranked.convenient_distance.agent"
        self.assertIn(expected_handle, _handles(session))
        self.assertTrue(_metric_supported(session, "convenient_distance"))

        ranked = session.handle_utterance("what is the convenientDistance to all the doors")
        self.assertIn("DOORS RANKED", ranked.upper())
        self.assertIn("CONVENIENT", ranked.upper())
        self.assertIn(expected_handle, _plan_handles(session))

    def test_manclid_equals_shorthand_can_be_defined_approved_and_used(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        proposal = session.handle_utterance(
            "create a new distance metric called manclid = max(manhattan distance, euclidean distance)"
        )

        self.assertNotIn("I didn't understand", proposal)
        proposal_lower = proposal.lower()
        self.assertIn("manclid", proposal_lower)
        self.assertIn("max(", proposal_lower)
        self.assertIn("manhattan", proposal_lower)
        self.assertIn("euclidean", proposal_lower)
        self.assertTrue(
            hasattr(session, "pending_primitive_definition")
            or session.pending_synthesis_proposal is not None,
            "Equals-sign metric definitions must become pending typed definitions.",
        )

        session.handle_utterance("yes")

        expected_handle = "grounding.all_doors.ranked.manclid.agent"
        self.assertIn(expected_handle, _handles(session))
        self.assertTrue(_metric_supported(session, "manclid"))

        ranked = session.handle_utterance("whats the manclid distance to all the doors")
        self.assertIn("DOORS RANKED", ranked.upper())
        self.assertIn("MANCLID", ranked.upper())
        self.assertIn(expected_handle, _plan_handles(session))

    def test_inline_sum_metric_inside_task_is_defined_then_resumed(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        proposal = session.handle_utterance(
            "go to the third farthest door based on the sum of both distance metrics"
        )

        self.assertNotIn("Which distance metric should I use", proposal)
        self.assertIn("PRIMITIVE DEFINITION PROPOSAL", proposal)
        self.assertIn("sum", proposal.lower())
        self.assertIn("euclidean", proposal.lower())
        self.assertIn("manhattan", proposal.lower())
        self.assertIsNotNone(session.pending_primitive_definition)
        self.assertIsNotNone(session.last_request_plan)
        self.assertEqual(session.last_request_plan.objective_type, "primitive_definition")

        resumed = session.handle_utterance("yes")

        expected_handle = "grounding.all_doors.ranked.sum_euclidean_manhattan.agent"
        self.assertIn(expected_handle, _handles(session))
        self.assertTrue(_metric_supported(session, "sum_euclidean_manhattan"))
        self.assertIn("PRIMITIVE DEFINITION REGISTERED", resumed)
        self.assertIn("RESUMING ORIGINAL REQUEST", resumed)
        self.assertIn("RUN COMPLETE", resumed)
        self.assertIn("go to the yellow door", resumed)
        self.assertIn(expected_handle, _plan_handles(session))
        self.assertIsNotNone(session.last_request_plan)
        self.assertEqual(session.last_request_plan.objective_type, "task")
        self.assertIsNotNone(session.last_readiness_graph)
        self.assertEqual(session.last_readiness_graph.next_action, "execute_task")

    def test_readiness_allows_claims_produced_by_prior_plan_step(self):
        plan = RequestPlan(
            request_id="inline-metric-plan",
            original_utterance="go to the third farthest door by a derived metric",
            objective_type="task",
            objective_summary="rank, select, execute",
            expected_response="execute_task",
            steps=[
                RequestPlanStep(
                    step_id="rank_scene_doors",
                    layer="grounding",
                    operation="rank",
                    required_handle="grounding.all_doors.ranked.manhattan.agent",
                    outputs=["active_claims.ranked_scene_doors"],
                ),
                RequestPlanStep(
                    step_id="select_grounded_target",
                    layer="claims",
                    operation="select",
                    inputs={"entries": "active_claims.ranked_scene_doors"},
                    outputs=["grounded_target"],
                    depends_on=["rank_scene_doors"],
                    memory_reads=["active_claims.ranked_scene_doors"],
                    scene_fingerprint_required=True,
                ),
                RequestPlanStep(
                    step_id="execute_task",
                    layer="task",
                    operation="execute",
                    required_handle="task.go_to_object.door",
                    depends_on=["select_grounded_target"],
                ),
            ],
        )

        graph = evaluate_request_plan(
            plan,
            registry=CapabilityRegistry.minigrid_default(),
            active_claims=None,
            claims_valid=False,
        )

        self.assertEqual(graph.graph_status, "executable")
        self.assertEqual(graph.next_action, "execute_task")

    def test_undefined_custom_metric_does_not_fallback_to_builtin_metric(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        response = session.handle_utterance("rank all doors by ramesian")

        self.assertNotIn("DOORS RANKED BY MANHATTAN", response.upper())
        self.assertNotIn("DOORS RANKED BY EUCLIDEAN", response.upper())
        self.assertTrue(
            any(
                token in response.lower()
                for token in (
                    "not defined",
                    "unknown metric",
                    "unsupported metric",
                    "define",
                    "missing",
                    "clarify",
                )
            ),
            "Undefined custom metrics must be reported as missing, not silently remapped.",
        )

    def test_rejected_metric_definition_registers_nothing(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        proposal = session.handle_utterance(
            "make a distance metric called nopeDistance as manhattan plus 99"
        )
        self.assertNotIn("I didn't understand", proposal)
        self.assertIn("nope", proposal.lower())

        session.handle_utterance("no")

        self.assertNotIn("grounding.all_doors.ranked.nope_distance.agent", _handles(session))
        self.assertFalse(_metric_supported(session, "nope_distance"))

    def test_actuation_metric_definition_is_refused(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        response = session.handle_utterance(
            "make a distance metric called rammer that moves forward then returns euclidean distance"
        )

        self.assertTrue(
            any(
                token in response.lower()
                for token in (
                    "refuse",
                    "unsafe",
                    "not allowed",
                    "query-only",
                    "cannot authorize",
                    "actuation",
                )
            ),
            "Metric definitions must stay query-only and refuse actuation side effects.",
        )
        self.assertNotIn("grounding.all_doors.ranked.rammer.agent", _handles(session))
        self.assertFalse(_metric_supported(session, "rammer"))


if __name__ == "__main__":
    unittest.main()
