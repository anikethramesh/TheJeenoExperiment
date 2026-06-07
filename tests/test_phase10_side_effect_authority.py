from __future__ import annotations

import unittest

from jeenom.schemas import (
    ExecutionTicket,
    MemoryUpdate,
    MemoryWriteTicket,
    RawMotorTicket,
    ReadinessGraph,
    RequestPlan,
    RequestPlanStep,
    SchemaValidationError,
)
from jeenom.side_effect_authority import SideEffectAuthority


def make_plan_and_graph(
    *,
    request_id: str,
    objective_type: str,
    expected_response: str,
    next_action: str,
) -> tuple[RequestPlan, ReadinessGraph]:
    plan = RequestPlan(
        request_id=request_id,
        original_utterance="unit request",
        objective_type=objective_type,
        objective_summary="Unit side-effect authority request.",
        expected_response=expected_response,
        steps=[
            RequestPlanStep(
                step_id=next_action,
                layer="spine",
                operation="execute",
            )
        ],
    )
    graph = ReadinessGraph(
        request_id=request_id,
        graph_status="executable",
        next_action=next_action,
        explanation="Unit executable side effect.",
    )
    return plan, graph


class TestPhase10SideEffectAuthority(unittest.TestCase):
    def test_issue_execution_ticket(self):
        plan, graph = make_plan_and_graph(
            request_id="unit-exec-ticket",
            objective_type="task",
            expected_response="execute_task",
            next_action="execute_task",
        )
        ticket = SideEffectAuthority().issue_execution_ticket(
            instruction="go to the red door",
            task_type="go_to_object",
            params={"object_type": "door", "color": "red"},
            request_plan=plan,
            readiness_graph=graph,
            source="unit",
        )

        self.assertIsInstance(ticket, ExecutionTicket)
        self.assertEqual(ticket.request_id, plan.request_id)
        self.assertIs(ticket.request_plan, plan)
        self.assertIs(ticket.readiness_graph, graph)
        self.assertEqual(ticket.task_type, "go_to_object")

    def test_execution_ticket_rejects_wrong_next_action(self):
        plan, graph = make_plan_and_graph(
            request_id="unit-exec-wrong-action",
            objective_type="task",
            expected_response="execute_task",
            next_action="execute_motor",
        )

        with self.assertRaises(SchemaValidationError):
            SideEffectAuthority().issue_execution_ticket(
                instruction="go to the red door",
                task_type="go_to_object",
                params={"object_type": "door", "color": "red"},
                request_plan=plan,
                readiness_graph=graph,
                source="unit",
            )

    def test_issue_raw_motor_ticket(self):
        plan, graph = make_plan_and_graph(
            request_id="unit-motor-ticket",
            objective_type="motor",
            expected_response="execute_motor",
            next_action="execute_motor",
        )
        ticket = SideEffectAuthority().issue_raw_motor_ticket(
            action_name="turn_right",
            repeat_count=2,
            request_plan=plan,
            readiness_graph=graph,
            source="unit",
        )

        self.assertIsInstance(ticket, RawMotorTicket)
        self.assertEqual(ticket.action_name, "turn_right")
        self.assertEqual(ticket.repeat_count, 2)

    def test_issue_memory_write_ticket(self):
        plan, graph = make_plan_and_graph(
            request_id="unit-memory-ticket",
            objective_type="memory_update",
            expected_response="update_memory",
            next_action="update_memory",
        )
        write = MemoryUpdate(
            scope="knowledge",
            key="delivery_target",
            value={"object_type": "door", "color": "red"},
            reason="unit",
        )
        ticket = SideEffectAuthority().issue_memory_write_ticket(
            writes=[write],
            request_plan=plan,
            readiness_graph=graph,
            source="unit",
        )

        self.assertIsInstance(ticket, MemoryWriteTicket)
        self.assertEqual(ticket.writes, [write])

    def test_memory_write_ticket_rejects_empty_writes(self):
        plan, graph = make_plan_and_graph(
            request_id="unit-empty-memory-ticket",
            objective_type="memory_update",
            expected_response="update_memory",
            next_action="update_memory",
        )

        with self.assertRaises(SchemaValidationError):
            SideEffectAuthority().issue_memory_write_ticket(
                writes=[],
                request_plan=plan,
                readiness_graph=graph,
                source="unit",
            )


if __name__ == "__main__":
    unittest.main()
