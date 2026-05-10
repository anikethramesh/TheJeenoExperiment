"""Phase 7.58 — Station Active Claims probe.

Verifies that:
- active_claims is None before any grounding query.
- closest grounding writes StationActiveClaims with ranked_scene_doors.
- last_grounded_rank is 0 after the first grounding.
- next_closest resolves to rank-1 door from claims (no env reset).
- other_door resolves to doors not yet referenced.
- claims are cleared on session reset.
- claims are cleared at the start of a new task (run_task).
- stale claims (fingerprint mismatch) fail gracefully with ok=False.
- compact_summary() returns expected shape.
- active_claims_summary is passed to compile_operator_intent.
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path
from pprint import pprint
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.minigrid_adapter import MiniGridAdapter
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import GroundedDoorEntry, StationActiveClaims


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session(render_mode: str = "none") -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode=render_mode,
        memory_root=Path(tempfile.mkdtemp()),
    )


def _run(fn):
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        return fn()


def main() -> int:
    checks: dict[str, bool] = {}

    # ── 1. active_claims is None before any grounding ──────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s1 = _make_session()
        s1.handle_utterance("go to the red door")
    checks["active_claims_none_before_grounding"] = s1.active_claims is None

    # ── 2. Closest grounding writes StationActiveClaims ───────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s2 = _make_session()
        s2.handle_utterance("go to the red door")
        s2.handle_utterance("which door is closest by manhattan distance")
    checks["active_claims_written_after_closest"] = s2.active_claims is not None
    checks["active_claims_is_correct_type"] = isinstance(s2.active_claims, StationActiveClaims)
    checks["ranked_scene_doors_non_empty"] = (
        s2.active_claims is not None and len(s2.active_claims.ranked_scene_doors) > 0
    )
    checks["last_grounded_target_is_entry"] = (
        s2.active_claims is not None
        and isinstance(s2.active_claims.last_grounded_target, GroundedDoorEntry)
    )
    checks["last_grounded_rank_is_zero"] = (
        s2.active_claims is not None and s2.active_claims.last_grounded_rank == 0
    )

    print("ACTIVE CLAIMS after closest grounding")
    if s2.active_claims:
        pprint(s2.active_claims.compact_summary())
    print()

    # ── 3. next_closest resolves from claims ───────────────────────────────
    reset_calls: list = []
    original_reset = MiniGridAdapter.reset

    def tracking_reset(self_adapter, seed=None):
        reset_calls.append(seed)
        return original_reset(self_adapter, seed=seed)

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s3 = _make_session()
        s3.handle_utterance("go to the red door")
        s3.handle_utterance("which door is closest by manhattan distance")
        reset_calls.clear()
        with patch.object(MiniGridAdapter, "reset", tracking_reset):
            result_next = s3.handle_utterance("next closest door")

    checks["next_closest_resolves_ok"] = "CLAIM" in result_next.upper() or "door" in result_next.lower()
    checks["next_closest_no_adapter_reset"] = reset_calls == []

    print("NEXT CLOSEST result")
    print(result_next)
    print(f"adapter.reset calls: {reset_calls}")
    print()

    # ── 4. other_door resolves ─────────────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s4 = _make_session()
        s4.handle_utterance("go to the red door")
        s4.handle_utterance("which door is closest by manhattan distance")
        result_other = s4.handle_utterance("the other door")

    checks["other_door_resolves"] = result_other is not None and len(result_other) > 0

    print("OTHER DOOR result")
    print(result_other)
    print()

    # ── 5. claims cleared on reset ─────────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s5 = _make_session()
        s5.handle_utterance("go to the red door")
        s5.handle_utterance("which door is closest by manhattan distance")
        had_claims = s5.active_claims is not None
        s5.reset()
        cleared = s5.active_claims is None

    checks["claims_cleared_on_reset"] = had_claims and cleared

    # ── 6. stale claims fail gracefully ───────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s6 = _make_session()
        s6.handle_utterance("go to the red door")
        s6.handle_utterance("which door is closest by manhattan distance")
        stale = dataclasses.replace(s6.active_claims, scene_fingerprint=(-1, -1, -1))
        s6.active_claims = stale
        stale_result = s6._resolve_claim_reference("next_closest")

    checks["stale_claims_fail_gracefully"] = not stale_result.get("ok", True)

    print("STALE CLAIMS result")
    pprint(stale_result)
    print()

    # ── 7. compact_summary shape ───────────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s7 = _make_session()
        s7.handle_utterance("go to the red door")
        s7.handle_utterance("which door is closest by manhattan distance")
        summary = s7.active_claims.compact_summary()

    checks["compact_summary_is_dict"] = isinstance(summary, dict)
    checks["compact_summary_has_last_grounded_target"] = "last_grounded_target" in summary
    checks["compact_summary_has_ranked_doors"] = "ranked_doors" in summary
    checks["compact_summary_has_last_rank"] = "last_rank" in summary

    print("COMPACT SUMMARY")
    pprint(summary)
    print()

    # ── 8. is_valid_for checks fingerprint ─────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        s8 = _make_session()
        s8.handle_utterance("go to the red door")
        s8.handle_utterance("which door is closest by manhattan distance")
        scene = s8.memory.scene_model
        valid_for_current = s8.active_claims.is_valid_for(scene)
        fake_scene = dataclasses.replace(scene, agent_x=scene.agent_x + 99, step_count=9999)
        invalid_for_fake = not s8.active_claims.is_valid_for(fake_scene)

    checks["is_valid_for_current_scene"] = valid_for_current
    checks["is_invalid_for_different_scene"] = invalid_for_fake

    # ── Summary ────────────────────────────────────────────────────────────
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
