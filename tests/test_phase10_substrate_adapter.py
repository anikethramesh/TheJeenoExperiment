from __future__ import annotations

import unittest

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_substrate_adapter import MiniGridSubstrateAdapter
from jeenom.plan_cache import PlanCache
from jeenom.sense import MiniGridSense
from jeenom.spine import MiniGridSpine


class TestPhase10SubstrateAdapter(unittest.TestCase):
    def test_minigrid_substrate_creates_role_bindings(self):
        substrate = MiniGridSubstrateAdapter(
            env_id="MiniGrid-GoToDoor-8x8-v0",
            render_mode="none",
        )
        memory = OperationalMemory()
        compiler = SmokeTestCompiler()
        plan_cache = PlanCache(enabled=True)

        self.assertIsInstance(
            substrate.create_sense(memory, compiler, plan_cache),
            MiniGridSense,
        )
        self.assertIsInstance(
            substrate.create_spine(memory, compiler, plan_cache),
            MiniGridSpine,
        )
        self.assertIn("turn_right", substrate.known_action_names())

    def test_minigrid_substrate_raw_motor_execution(self):
        substrate = MiniGridSubstrateAdapter(
            env_id="MiniGrid-GoToDoor-8x8-v0",
            render_mode="none",
        )

        result = substrate.run_motor_actions(seed=42, actions=["turn_right"])

        self.assertTrue(result["success"])
        self.assertEqual(result["steps_taken"], 1)
        self.assertEqual(result["actions_executed"], ["turn_right"])


if __name__ == "__main__":
    unittest.main()
