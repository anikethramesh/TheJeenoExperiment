"""Visual demonstration of Phase 8.3 plan reuse.

Run with:
    python demo_plan_reuse.py

Shows three scenarios back-to-back:
  1. First call  → plan compiled fresh, stored in cache
  2. Second call (same env type, different seed) → plan reused
  3. Third call  (different env size 16x16) → plan rejected, recompiled fresh
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Minigrid"))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.plan_reuse import PlanReuseCache, plan_semantic_key


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session(env_id: str, seed: int, cache: PlanReuseCache) -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id=env_id,
        seed=seed,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
        request_plan_reuse_cache=cache,
    )


UTTERANCE = "which door is closest by manhattan distance"
PRIME    = "what doors do you see?"


def _show_cache(cache: PlanReuseCache) -> None:
    print(f"  cache entries : {len(cache.entries)}")
    for entry in cache.entries.values():
        print(f"    key={entry.key}  reuse_count={entry.reuse_count}")
    if cache.history:
        last = cache.history[-1]
        print(f"  last verdict  : {last.verdict}  ({last.reason[:70]})")


def main() -> None:
    shared_cache = PlanReuseCache()

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):

        # ── Scenario 1: First call ────────────────────────────────────────────
        print("=" * 60)
        print("SCENARIO 1 — first call (8x8, seed=42)")
        print("=" * 60)
        s1 = _make_session("MiniGrid-GoToDoor-8x8-v0", 42, shared_cache)
        s1.handle_utterance(PRIME)      # populate environment identity
        s1.handle_utterance(UTTERANCE)
        verdict = s1.last_plan_reuse_verdict
        key = plan_semantic_key(s1.last_request_plan) if s1.last_request_plan else "?"
        print(f"  plan key      : {key}")
        print(f"  reuse verdict : {verdict}")   # None = no prior plan to compare against
        print(f"  → Plan compiled fresh and stored.")
        _show_cache(shared_cache)

        # ── Scenario 2: Same env type, different seed ─────────────────────────
        print()
        print("=" * 60)
        print("SCENARIO 2 — same env type (8x8), different seed=43")
        print("=" * 60)
        s2 = _make_session("MiniGrid-GoToDoor-8x8-v0", 43, shared_cache)
        s2.handle_utterance(PRIME)      # different door layout, same grid
        s2.handle_utterance(UTTERANCE)
        verdict2 = s2.last_plan_reuse_verdict
        print(f"  reuse verdict : {verdict2.verdict}")
        print(f"  reason        : {verdict2.reason}")
        print(f"  → Plan structure transferred. Doors re-ranked against new layout.")
        _show_cache(shared_cache)

        # ── Scenario 3: Different env size ────────────────────────────────────
        print()
        print("=" * 60)
        print("SCENARIO 3 — different env size (16x16), seed=42")
        print("=" * 60)
        s3 = _make_session("MiniGrid-GoToDoor-16x16-v0", 42, shared_cache)
        s3.handle_utterance(PRIME)
        s3.handle_utterance(UTTERANCE)
        verdict3 = s3.last_plan_reuse_verdict
        print(f"  reuse verdict : {verdict3.verdict}")
        print(f"  reason        : {verdict3.reason}")
        print(f"  blocking      : {verdict3.blocking_assumption_ids}")
        print(f"  → Required assumption failed. Plan recompiled fresh for 16x16.")
        _show_cache(shared_cache)

        # ── Summary ───────────────────────────────────────────────────────────
        print()
        print("=" * 60)
        print("REUSE HISTORY")
        print("=" * 60)
        for record in shared_cache.history:
            print(f"  {record.verdict:<12} env={record.env_id}  seed={record.seed}")


if __name__ == "__main__":
    main()
