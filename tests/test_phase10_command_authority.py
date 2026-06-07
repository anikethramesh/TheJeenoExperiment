from __future__ import annotations

import unittest

from jeenom.command_authority import CommandAuthority
from jeenom.schemas import (
    CommandResult,
    CorticalEnvelope,
    RawMotorTicket,
    ReadinessGraph,
    RequestPlan,
    RequestPlanStep,
)


def make_motor_plan_and_graph() -> tuple[RequestPlan, ReadinessGraph]:
    plan = RequestPlan(
        request_id="unit-phase10-command-authority",
        original_utterance="turn right twice",
        objective_type="motor",
        objective_summary="Execute a raw motor command.",
        expected_response="execute_motor",
        steps=[
            RequestPlanStep(
                step_id="execute_motor",
                layer="spine",
                operation="execute",
            )
        ],
    )
    graph = ReadinessGraph(
        request_id=plan.request_id,
        graph_status="executable",
        next_action="execute_motor",
        explanation="Unit executable motor command.",
    )
    return plan, graph


class TestPhase10CommandAuthority(unittest.TestCase):
    def test_record_result_builds_typed_trace_and_preserves_ticket(self):
        plan, graph = make_motor_plan_and_graph()
        ticket = RawMotorTicket(
            request_id=plan.request_id,
            action_name="turn_right",
            repeat_count=2,
            request_plan=plan,
            readiness_graph=graph,
        )
        authority = CommandAuthority(station_name="UnitStation")

        result = authority.record_result(
            "turn right twice",
            "MOTOR COMPLETE\nsteps=2",
            intent=None,
            plan=plan,
            graph=graph,
            tickets=[ticket],
            compiler_name="smoke",
            pending_context={"clarification": None, "synthesis": None},
            last_result={"steps": 2},
        )

        self.assertIsInstance(result, CommandResult)
        self.assertIsInstance(result.envelope, CorticalEnvelope)
        self.assertEqual(result.envelope.provenance["station"], "UnitStation")
        self.assertEqual(result.envelope.provenance["compiler"], "smoke")
        self.assertIs(result.envelope.request_plan, plan)
        self.assertIs(result.envelope.readiness_graph, graph)
        self.assertEqual(result.command.command_type, "execute_motor")
        self.assertEqual(result.command.request_id, plan.request_id)
        self.assertIs(result.ticket, ticket)
        self.assertEqual(result.result["last_result"], {"steps": 2})

    def test_pending_clarification_trace_returns_schema_payload(self):
        plan, graph = make_motor_plan_and_graph()
        authority = CommandAuthority(station_name="UnitStation")

        trace = authority.pending_clarification_trace(
            "closest door",
            "distance_metric",
            intent=None,
            plan=plan,
            graph=graph,
            compiler_name="smoke",
        )

        self.assertIs(trace["request_plan"], plan)
        self.assertIs(trace["readiness_graph"], graph)
        self.assertIsInstance(trace["pending_envelope"], CorticalEnvelope)
        self.assertEqual(
            trace["pending_envelope"].pending_context,
            {"clarification": "distance_metric"},
        )


if __name__ == "__main__":
    unittest.main()
