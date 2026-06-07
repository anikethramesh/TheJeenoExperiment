from __future__ import annotations

import unittest

from evals.harness import make_session


class TestPhase10LiveOperatorRegressions(unittest.TestCase):
    def test_verified_ranked_query_records_query_plan_not_refuse(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        response = session.handle_utterance("what is the distance of all the doors from you")

        self.assertIn("DOORS RANKED BY MANHATTAN DISTANCE", response)
        self.assertIsNotNone(session.last_operator_intent)
        self.assertEqual(session.last_operator_intent.intent_type, "status_query")
        self.assertIsNotNone(session.last_request_plan)
        self.assertEqual(session.last_request_plan.objective_type, "query")
        self.assertIsNotNone(session.last_readiness_graph)
        self.assertEqual(session.last_readiness_graph.next_action, "answer_query")

    def test_metric_followup_uses_previous_ranked_context(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)
        session.handle_utterance("what is the distance of all the doors from you")

        response = session.handle_utterance("what is the euclidean distance")

        self.assertNotIn("I didn't understand", response)
        self.assertIsNotNone(session.last_operator_intent)
        self.assertEqual(session.last_operator_intent.intent_type, "status_query")
        self.assertIsNotNone(session.last_request_plan)
        handles = {
            step.required_handle
            for step in session.last_request_plan.steps
            if step.required_handle is not None
        }
        self.assertIn("grounding.all_doors.ranked.euclidean.agent", handles)
        self.assertIsNotNone(session.last_readiness_graph)
        self.assertIn(
            session.last_readiness_graph.next_action,
            {"answer_query", "propose_synthesis"},
        )

    def test_unsupported_refuse_plan_is_not_cached_for_reuse(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        first = session.handle_utterance("synthesize a new distance metric named convenientDistance")
        first_cache_size = len(session.request_plan_reuse_cache.entries)
        second = session.handle_utterance("another unknown operator request")

        self.assertIn("I didn't understand", first)
        self.assertIn("I didn't understand", second)
        self.assertEqual(first_cache_size, 0)
        self.assertEqual(len(session.request_plan_reuse_cache.entries), 0)
        self.assertIsNone(session.last_plan_reuse_verdict)


if __name__ == "__main__":
    unittest.main()
