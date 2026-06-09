from __future__ import annotations

import unittest
from dataclasses import fields

from evals.harness import make_session
from jeenom.schemas import ExecutionTicket, MissionExecutionPlan


INLINE_TASK = "go to the third farthest door based on the sum of euclidean and manhattan distance"
EXPECTED_HANDLE = "grounding.all_doors.ranked.sum_euclidean_manhattan.agent"


def _step_ids(plan) -> list[str]:
    if plan is None:
        return []
    return [step.step_id for step in plan.steps]


class TestPhase11MissionFlow(unittest.TestCase):
    def test_mission_cortex_builds_typed_inline_metric_mission(self):
        from jeenom.mission_cortex import MissionCortex, parse_inline_metric_request

        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)
        parsed = parse_inline_metric_request(INLINE_TASK, session.capability_registry)
        self.assertIsNotNone(parsed)
        self.assertFalse(isinstance(parsed, str), parsed)

        cortex = MissionCortex(
            planning_semantics=session.planning_semantics,
            registry=session.capability_registry,
        )
        mission_plan = cortex.plan_inline_metric_request(
            parsed,
            active_claims=session.active_claims,
            claims_valid=session._claims_valid_for_current_environment(),
            environment_identity=session.current_environment_identity,
        )

        self.assertIsInstance(mission_plan, MissionExecutionPlan)
        self.assertIsNotNone(mission_plan.mission_contract)
        self.assertIsNotNone(mission_plan.primitive_definition)
        self.assertIsNotNone(mission_plan.continuation_intent)
        self.assertEqual(mission_plan.primitive_definition.proposed_handle, EXPECTED_HANDLE)
        self.assertEqual(mission_plan.continuation_intent.intent_type, "task_instruction")
        self.assertEqual(mission_plan.continuation_intent.selection_objective.ordinal, 3)
        self.assertEqual(mission_plan.continuation_intent.selection_objective.direction, "maximum")
        self.assertIn("mission_id", mission_plan.continuation_intent.reason)

    def test_station_pending_definition_carries_typed_mission_plan(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        proposal = session.handle_utterance(INLINE_TASK)

        self.assertIn("PRIMITIVE DEFINITION PROPOSAL", proposal)
        self.assertIsNotNone(session.pending_primitive_definition)
        self.assertFalse(
            hasattr(session.pending_primitive_definition, "resume_payload"),
            "Pending primitive definitions must not carry station-local resume dicts.",
        )
        self.assertIsInstance(
            session.pending_primitive_definition.mission_plan,
            MissionExecutionPlan,
        )
        self.assertEqual(
            session.pending_primitive_definition.mission_plan.primitive_definition.proposed_handle,
            EXPECTED_HANDLE,
        )

    def test_approval_preserves_mission_lineage_on_execution_ticket(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        session.handle_utterance(INLINE_TASK)
        response = session.handle_utterance("yes")

        self.assertIn("PRIMITIVE DEFINITION REGISTERED", response)
        self.assertIn("RESUMING ORIGINAL REQUEST", response)
        self.assertIn("RUN COMPLETE", response)
        self.assertIn("go to the yellow door", response)
        self.assertIsNotNone(session.last_mission_execution_plan)
        mission_plan = session.last_mission_execution_plan
        ticket = session.last_execution_ticket
        self.assertIsInstance(ticket, ExecutionTicket)
        self.assertEqual(ticket.mission_id, mission_plan.mission_id)
        self.assertEqual(ticket.parent_request_id, mission_plan.request_plan.request_id)
        self.assertEqual(ticket.provenance.get("original_utterance"), INLINE_TASK)
        self.assertEqual(ticket.provenance.get("primitive_handle"), EXPECTED_HANDLE)
        self.assertIn(ticket, mission_plan.child_tickets)

        continuation_steps = _step_ids(mission_plan.continuation_request_plan)
        self.assertIn("rank_scene_doors", continuation_steps)
        self.assertIn("select_grounded_target", continuation_steps)
        self.assertIn("execute_task", continuation_steps)
        self.assertEqual(session.last_readiness_graph.next_action, "execute_task")
        self.assertEqual(session.last_result["runtime_llm_calls_during_render"], 0)

    def test_rejection_registers_nothing_and_resumes_nothing(self):
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)

        session.handle_utterance(INLINE_TASK)
        response = session.handle_utterance("no")

        self.assertIn("PRIMITIVE DEFINITION REJECTED", response)
        self.assertIsNone(session.pending_primitive_definition)
        self.assertIsNone(getattr(session, "last_mission_execution_plan", None))
        self.assertIsNone(session.last_execution_ticket)
        self.assertNotIn(EXPECTED_HANDLE, session.capability_registry.primitive_names())

    def test_execution_ticket_schema_has_mission_lineage_fields(self):
        field_names = {field.name for field in fields(ExecutionTicket)}
        self.assertIn("mission_id", field_names)
        self.assertIn("parent_request_id", field_names)
        self.assertIn("provenance", field_names)


if __name__ == "__main__":
    unittest.main()
