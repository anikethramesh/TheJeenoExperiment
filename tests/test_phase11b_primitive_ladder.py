from __future__ import annotations

import unittest
from typing import Any

from evals.harness import make_session


ENV_ID = "MiniGrid-GoToDoor-16x16-v0"
SEED = 8
RANKED_MANHATTAN = "grounding.all_doors.ranked.manhattan.agent"


def _handles(session: Any) -> set[str]:
    plan = getattr(session, "last_request_plan", None)
    if plan is None:
        return set()
    return {
        step.required_handle
        for step in plan.steps
        if getattr(step, "required_handle", None) is not None
    }


def _actions(session: Any) -> list[tuple[str, int]]:
    plan = getattr(session, "last_request_plan", None)
    if plan is None:
        return []
    result: list[tuple[str, int]] = []
    for step in plan.steps:
        action = step.inputs.get("action_name")
        if action is not None:
            result.append((action, int(step.inputs.get("repeat_count") or 1)))
    return result


def _assert_ranked_distance_query(testcase: unittest.TestCase, session: Any, response: str) -> None:
    testcase.assertNotIn("I didn't understand", response)
    testcase.assertIsNotNone(session.last_operator_intent)
    testcase.assertEqual(session.last_operator_intent.intent_type, "status_query")
    testcase.assertIsNotNone(session.last_request_plan)
    testcase.assertEqual(session.last_request_plan.objective_type, "query")
    testcase.assertIn(RANKED_MANHATTAN, _handles(session))
    testcase.assertIsNotNone(session.last_readiness_graph)
    testcase.assertEqual(session.last_readiness_graph.next_action, "answer_query")
    testcase.assertIsNone(session.last_raw_motor_ticket)
    testcase.assertIsNone(session.last_execution_ticket)
    testcase.assertIsNotNone(session.active_claims)
    testcase.assertTrue(session.active_claims.ranked_scene_doors)


class TestPhase11BPrimitiveLadder(unittest.TestCase):
    def test_distance_query_paraphrases_are_semantically_equivalent(self):
        for utterance in [
            "how far are all the doors from you",
            "how far are the doors from you",
            "distance to the doors",
            "show me the door distances",
            "what are the distances to the doors",
        ]:
            with self.subTest(utterance=utterance):
                session = make_session(env_id=ENV_ID, seed=SEED)
                response = session.handle_utterance(utterance)
                _assert_ranked_distance_query(self, session, response)

    def test_stateful_visible_set_reference_after_scene_query(self):
        session = make_session(env_id=ENV_ID, seed=SEED)
        scene_response = session.handle_utterance("what are the doors you see around you")
        self.assertIn("SCENE", scene_response)

        response = session.handle_utterance("how far are the doors from you")

        _assert_ranked_distance_query(self, session, response)

    def test_low_level_sense_paraphrase_routes_to_query_without_motion(self):
        session = make_session(env_id=ENV_ID, seed=SEED)

        response = session.handle_utterance("what is in front of me")

        self.assertNotIn("I didn't understand", response)
        self.assertIsNotNone(session.last_request_plan)
        self.assertEqual(session.last_request_plan.objective_type, "query")
        self.assertIsNotNone(session.last_readiness_graph)
        self.assertIn(
            session.last_readiness_graph.next_action,
            {"answer_query", "ask_clarification"},
        )
        self.assertIsNone(session.last_raw_motor_ticket)
        self.assertIsNone(session.last_execution_ticket)

    def test_low_level_spine_paraphrase_gets_raw_motor_ticket(self):
        session = make_session(env_id=ENV_ID, seed=SEED)

        response = session.handle_utterance("advance one cell")

        self.assertNotIn("I didn't understand", response)
        self.assertIsNotNone(session.last_operator_intent)
        self.assertEqual(session.last_operator_intent.intent_type, "motor_command")
        self.assertIsNotNone(session.last_request_plan)
        self.assertEqual(session.last_request_plan.objective_type, "motor_control")
        self.assertIsNotNone(session.last_readiness_graph)
        self.assertEqual(session.last_readiness_graph.next_action, "execute_motor")
        self.assertIsNotNone(session.last_raw_motor_ticket)
        self.assertEqual(session.last_raw_motor_ticket.action_name, "move_forward")
        self.assertEqual(session.last_raw_motor_ticket.repeat_count, 1)

    def test_named_procedure_teaching_uses_typed_memory_update_plan(self):
        session = make_session(env_id=ENV_ID, seed=SEED)

        teach_response = session.handle_utterance(
            "when I say fing fam foom, give me the distance to all doors"
        )

        self.assertIn("STORED", teach_response.upper())
        self.assertIsNotNone(session.last_request_plan)
        self.assertEqual(session.last_request_plan.objective_type, "knowledge_update")
        self.assertIsNotNone(session.last_readiness_graph)
        self.assertEqual(session.last_readiness_graph.next_action, "update_memory")
        procedure = session.representation.snapshot().procedures.get("fing fam foom")
        self.assertIsNotNone(procedure)
        self.assertTrue(procedure.get("plan"))

        invoke_response = session.handle_utterance("run fing fam foom")
        _assert_ranked_distance_query(self, session, invoke_response)

    def test_procedure_sequence_uses_cached_atomic_plans_for_relative_and_query_concepts(self):
        session = make_session(env_id=ENV_ID, seed=SEED)
        session.handle_utterance("if I say bingo go to the closest door. Store that")
        session.handle_utterance(
            "if I say bongo you need to print the distance to all doors. Store that"
        )

        bare_response = session.handle_utterance("bingo bongo")
        then_response = session.handle_utterance("do bingo then bongo")

        for response in (bare_response, then_response):
            self.assertIn("PROCEDURE COMPLETE", response)
            self.assertIn("RUN COMPLETE", response)
            self.assertIn("DOORS RANKED BY MANHATTAN DISTANCE FROM AGENT", response)
        self.assertIsNotNone(session.last_request_plan)
        self.assertIn(
            "grounding.all_doors.ranked.manhattan.agent",
            _handles(session),
        )

    def test_action_sequence_preserves_all_child_actions_not_only_final_child(self):
        session = make_session(env_id=ENV_ID, seed=SEED)

        response = session.handle_utterance("go straight two steps and turn left")

        self.assertNotIn("SEQUENCE ERROR", response)
        self.assertEqual(_actions(session), [("move_forward", 2), ("turn_left", 1)])
        self.assertIsNotNone(session.last_raw_motor_ticket)

    def test_conditional_actuation_requires_evidence_before_motor_ticket(self):
        session = make_session(env_id=ENV_ID, seed=SEED)

        response = session.handle_utterance(
            "if there is a red door in front of me, go forward, otherwise stay"
        )

        self.assertNotIn("MOTOR COMPLETE", response)
        self.assertIsNone(session.last_raw_motor_ticket)
        self.assertIsNotNone(session.last_request_plan)
        step_ids = {step.step_id for step in session.last_request_plan.steps}
        self.assertTrue(any("sense" in step or "evidence" in step for step in step_ids))
        self.assertTrue(any("execute" in step for step in step_ids))

    def test_compound_mission_variants_create_mission_plan(self):
        for utterance in [
            (
                "find euclidean distance to all doors, find manhattan distance to all "
                "doors, then go to the third farthest by their sum"
            ),
            "go to the second highest door by max(euclidean, manhattan)",
        ]:
            with self.subTest(utterance=utterance):
                session = make_session(env_id=ENV_ID, seed=SEED)
                response = session.handle_utterance(utterance)
                pending = getattr(session, "pending_primitive_definition", None)
                pending_mission = getattr(pending, "mission_plan", None)
                last_mission = getattr(session, "last_mission_execution_plan", None)

                self.assertNotIn("I didn't understand", response)
                self.assertTrue(pending_mission is not None or last_mission is not None)

    def test_negative_control_does_not_degrade_actuation_to_answer_only(self):
        session = make_session(env_id=ENV_ID, seed=SEED)

        response = session.handle_utterance("move to the farthest door by walking randomly")

        self.assertNotIn("GROUNDING ANSWER", response)
        self.assertIsNotNone(session.last_operator_intent)
        self.assertNotEqual(session.last_operator_intent.intent_type, "status_query")
        self.assertIsNone(session.last_raw_motor_ticket)
        self.assertIsNone(session.last_execution_ticket)


if __name__ == "__main__":
    unittest.main()
