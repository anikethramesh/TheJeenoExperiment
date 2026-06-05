"""Expose Problem 1: repair loop marks success=True without re-dispatching.

Blueprint invariant (Phase 9A): "repair must re-evaluate and re-dispatch
when it claims success — otherwise it must honestly report that it only
cleared state."

RepairLoop._repair_refresh_claims() currently clears active_claims and
returns RepairEvent(success=True). It does NOT re-dispatch. So when stale
claims trigger a repair, the station reports success but the original
request never executes.

These tests should FAIL until the repair loop actually re-dispatches.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import StationActiveClaims


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session(**kwargs) -> OperatorStationSession:
    defaults = dict(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    defaults.update(kwargs)
    return OperatorStationSession(**defaults)


class TestRepairLoopRedispatch(unittest.TestCase):
    """The repair loop must re-dispatch after repairing, not just clear state."""

    def setUp(self):
        self.patcher = patch("jeenom.run_demo.build_env", side_effect=_build_env)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_refresh_claims_does_not_mark_success_without_redispatch(self):
        """REFRESH_CLAIMS must not return success=True if execution did not resume.

        Currently FAILS: RepairLoop._repair_refresh_claims returns success=True
        immediately after clearing claims, with no re-dispatch.
        """
        session = _make_session()
        session.handle_utterance("rank all the doors by manhattan distance")

        # Corrupt the fingerprint to force STALE_CLAIMS mismatch.
        self.assertIsNotNone(session.active_claims, "warm-up should produce claims")
        session.active_claims = StationActiveClaims(
            scene_fingerprint=(999, 999, 999),
            ranked_scene_doors=session.active_claims.ranked_scene_doors,
            last_grounded_target=session.active_claims.last_grounded_target,
            last_grounded_rank=session.active_claims.last_grounded_rank,
            last_grounding_query=session.active_claims.last_grounding_query,
        )

        response = session.handle_utterance("go to the closest door")
        repairs = session.last_repair_events

        stale_repairs = [
            r for r in repairs
            if r.mismatch_type == "STALE_CLAIMS" and r.repair_action == "REFRESH_CLAIMS"
        ]
        self.assertTrue(len(stale_repairs) > 0, "Expected STALE_CLAIMS repair event")

        success_claimed = any(r.success for r in stale_repairs)
        execution_resumed = "RUN COMPLETE" in response

        # The invariant: success=True is only valid if execution actually resumed.
        # This assertion FAILS today because success=True is set but the task
        # never re-dispatches (RUN COMPLETE is absent).
        self.assertFalse(
            success_claimed and not execution_resumed,
            f"Repair claimed success=True but execution did not resume.\n"
            f"Response: {response[:200]}\n"
            f"Repair events: {[r.as_dict() for r in repairs]}",
        )

    def test_repair_actually_resumes_execution(self):
        """After repairing stale claims, the original request must execute.

        Currently FAILS: after REFRESH_CLAIMS clears state, the operator
        station does not retry the task — it either returns a repair message
        or leaves the task incomplete.
        """
        session = _make_session()
        session.handle_utterance("rank all the doors by manhattan distance")

        self.assertIsNotNone(session.active_claims)
        session.active_claims = StationActiveClaims(
            scene_fingerprint=(999, 999, 999),
            ranked_scene_doors=session.active_claims.ranked_scene_doors,
            last_grounded_target=session.active_claims.last_grounded_target,
            last_grounded_rank=session.active_claims.last_grounded_rank,
            last_grounding_query=session.active_claims.last_grounding_query,
        )

        response = session.handle_utterance("go to the closest door")

        # After repair + re-dispatch, the task should complete.
        # This FAILS today: repair clears claims but does not retry execution.
        self.assertIn(
            "RUN COMPLETE",
            response,
            f"Task should have executed after repair re-dispatch.\n"
            f"Response: {response[:400]}",
        )

    def test_repair_success_false_if_execution_not_resumed(self):
        """If execution is not resumed, all repair events must have success=False.

        Currently FAILS: REFRESH_CLAIMS sets success=True unconditionally in
        RepairLoop._repair_refresh_claims, regardless of whether execution resumed.
        """
        session = _make_session()
        session.handle_utterance("rank all the doors by manhattan distance")

        self.assertIsNotNone(session.active_claims)
        session.active_claims = StationActiveClaims(
            scene_fingerprint=(999, 999, 999),
            ranked_scene_doors=session.active_claims.ranked_scene_doors,
            last_grounded_target=session.active_claims.last_grounded_target,
            last_grounded_rank=session.active_claims.last_grounded_rank,
            last_grounding_query=session.active_claims.last_grounding_query,
        )

        response = session.handle_utterance("go to the closest door")
        repairs = session.last_repair_events
        execution_resumed = "RUN COMPLETE" in response

        if not execution_resumed:
            # No execution → no repair should claim success.
            # This FAILS today: REFRESH_CLAIMS returns success=True.
            false_successes = [r for r in repairs if r.success]
            self.assertEqual(
                len(false_successes),
                0,
                f"Repair(s) claimed success without execution resuming:\n"
                f"{[r.as_dict() for r in false_successes]}\n"
                f"Response: {response[:200]}",
            )


if __name__ == "__main__":
    unittest.main()
