"""Phase 9.1 probe: Operational Repair Loop.

Proves that when a mismatch is detected during plan evaluation, the RepairLoop
is invoked, successfully repairs the state (e.g. clearing stale claims), and
allows the execution to resume automatically.

Checks:
  repair_events_logged                — session.last_repair_events is populated
  stale_claims_repaired               — STALE_CLAIMS was repaired via REFRESH_CLAIMS
  repaired_plan_executes              — Task completed successfully after repair
  golden_path_unaffected              — Normal execution still works
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

from jeenom.schemas import StationActiveClaims, GroundedDoorEntry


def main() -> int:
    metrics: dict[str, bool] = {}

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session = _make_session(seed=42)
        
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
        
        # Check that the task actually ran after repair
        metrics["repaired_plan_executes"] = "RUN COMPLETE" in response
        
        # 4. Golden path
        golden = _make_session(seed=43)
        golden_response = golden.handle_utterance("go to the red door")
        metrics["golden_path_unaffected"] = "RUN COMPLETE" in golden_response

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
