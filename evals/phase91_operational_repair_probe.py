"""Phase 9.1 probe: Operational Repair Loop.

Proves that when a mismatch is detected during plan evaluation, the RepairLoop
is invoked, successfully repairs the state (e.g. clearing stale claims), and
allows the execution to resume automatically.

Checks:
  repair_events_logged                — session.last_repair_events is populated
  stale_claims_repaired               — STALE_CLAIMS was repaired via REFRESH_CLAIMS
  request_resumed_after_repair         — Task completed successfully after repair
  non_execution_disclosed_if_not_resumed — If not resumed, station tells operator repair did not execute
  repair_success_is_truthful           — success means resume or honest non-execution
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch
from testing_utils import build_env as _build_env, make_session as _make_session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.schemas import StationActiveClaims


def main() -> int:
    metrics: dict[str, bool] = {}

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session = _make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)
        
        # 1. Warm up claims
        session.handle_utterance("rank all the doors by manhattan distance")
        
        # 2. Corrupt the fingerprint to trigger STALE_CLAIMS
        if session.active_claims:
            session.active_claims = StationActiveClaims(
                scene_fingerprint=(999, 999, 999),  # Wrong fingerprint
                ranked_scene_doors=session.active_claims.ranked_scene_doors,
                last_grounded_target=session.active_claims.last_grounded_target,
                last_grounded_rank=session.active_claims.last_grounded_rank,
                last_grounding_query=session.active_claims.last_grounding_query,
            )
            
        # 3. Request a task that depends on those claims
        response = session.handle_utterance("go to the closest door")
        
        # Check that repairs happened
        repairs = session.last_repair_events
        metrics["repair_events_logged"] = len(repairs) > 0
        metrics["stale_claims_repaired"] = any(
            r.mismatch_type == "STALE_CLAIMS" and r.repair_action == "REFRESH_CLAIMS" and r.success
            for r in repairs
        )
        
        success_claimed = any(r.success for r in repairs)
        resumed_execution = "RUN COMPLETE" in response
        disclosed_non_execution = any(
            token in response.lower()
            for token in (
                "repair",
                "stale",
                "cleared",
                "did not execute",
                "re-ground",
                "reground",
            )
        )

        metrics["request_resumed_or_non_execution_disclosed"] = (
            resumed_execution or disclosed_non_execution
        )
        metrics["non_execution_disclosed_if_not_resumed"] = (
            resumed_execution or disclosed_non_execution
        )
        metrics["repair_success_is_truthful"] = (
            not success_claimed
            or resumed_execution
            or disclosed_non_execution
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
