from __future__ import annotations

import unittest

from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import ApprovedCommand


class TestPhase10TurnOrchestrator(unittest.TestCase):
    def test_station_has_turn_orchestrator(self):
        from jeenom.turn_orchestrator import TurnOrchestrator

        session = OperatorStationSession(
            compiler_name="smoke_test",
            render_mode="none",
        )

        self.assertIsInstance(session.turn_orchestrator, TurnOrchestrator)

    def test_execute_command_compatibility_is_preserved(self):
        session = OperatorStationSession(
            compiler_name="smoke_test",
            render_mode="none",
        )

        message = session.execute_command(
            ApprovedCommand(
                kind="clarification",
                utterance="unit",
                payload={"message": "orchestrated"},
            )
        )

        self.assertEqual(message, "orchestrated")

    def test_handle_utterance_still_records_command_result(self):
        session = OperatorStationSession(
            compiler_name="smoke_test",
            render_mode="none",
        )

        result = session.handle_utterance("help")

        self.assertTrue(result.message)
        self.assertIs(session.last_command_result, result)


if __name__ == "__main__":
    unittest.main()
