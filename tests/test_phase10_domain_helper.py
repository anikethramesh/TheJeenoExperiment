from __future__ import annotations

import unittest

from jeenom.minigrid_operational_context import MiniGridOperationalContext
from jeenom.operator_station import OperatorStationSession, classify_utterance


class TestPhase10DomainHelper(unittest.TestCase):
    def test_station_has_context_bound_domain_helper(self):
        from jeenom.minigrid_domain_helper import MiniGridDomainHelper

        session = OperatorStationSession(
            compiler_name="smoke_test",
            render_mode="none",
        )

        self.assertIsInstance(session.domain_helper, MiniGridDomainHelper)
        self.assertIs(session.domain_helper.operational_context, session.operational_context)

    def test_domain_helper_uses_operational_context_vocabulary(self):
        from jeenom.minigrid_domain_helper import MiniGridDomainHelper

        context = MiniGridOperationalContext.default()
        helper = MiniGridDomainHelper(context)

        self.assertIn("door", helper.object_types)
        self.assertIn("red", helper.supported_colors)
        self.assertEqual(helper.normalize_color("gray"), "grey")
        parsed = helper.parse_exact_go_to_object_utterance("go to the gray door")
        self.assertEqual(parsed["color"], "grey")
        self.assertEqual(parsed["object_type"], "door")

    def test_classify_utterance_preserves_domain_behavior(self):
        command = classify_utterance("go to the gray door")

        self.assertEqual(command.command_type, "task_instruction")


if __name__ == "__main__":
    unittest.main()
