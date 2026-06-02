"""Phase 8.3 probe: conservative RequestPlan reuse.

Proves that a compiled RequestPlan is stored after first execution and
correctly reused or rejected depending on whether the target environment's
required assumptions still hold.

Checks:
  plan_stored_after_first_request        — cache has an entry after first run
  same_key_on_repeated_utterance         — same utterance → same structural key
  reuse_verdict_same_env_different_seed  — seed change (diagnostic) → reuse
  reuse_count_incremented                — reuse_count on entry goes up
  reuse_history_recorded                 — history contains a "reuse" record
  recompile_verdict_different_env_size   — env size change → recompile
  recompile_history_recorded             — history contains a "recompile" record
  golden_path_unaffected                 — "go to the red door" still works
  runtime_llm_calls_during_render_zero
  cache_miss_during_render_zero
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from testing_utils import build_env as _build_env, make_session as _make_session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.plan_reuse import PlanReuseCache, plan_semantic_key






# Utterance that produces an immediately executable plan (full metric specified).
RANKED_UTTERANCE = "which door is closest by manhattan distance"
GOLDEN_UTTERANCE = "go to the red door"


def main() -> int:
    metrics: dict[str, bool] = {}

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):

        # ── Stage 1: First request — plan should be stored ───────────────────
        # Prime identity first so the stored plan carries environment assumptions
        # that can later be evaluated against other environments.
        shared_cache = PlanReuseCache()
        session1 = _make_session(seed=42, request_plan_reuse_cache=shared_cache)
        session1.handle_utterance("what doors do you see?")
        session1.handle_utterance(RANKED_UTTERANCE)

        metrics["plan_stored_after_first_request"] = len(shared_cache.entries) > 0

        # ── Stage 2: Determinism — same utterance → same key ─────────────────
        plan1 = session1.last_request_plan
        session1b = _make_session(seed=42, request_plan_reuse_cache=PlanReuseCache())
        session1b.handle_utterance(RANKED_UTTERANCE)
        plan1b = session1b.last_request_plan

        metrics["same_key_on_repeated_utterance"] = (
            plan1 is not None
            and plan1b is not None
            and plan_semantic_key(plan1) == plan_semantic_key(plan1b)
        )

        # ── Stage 3: Reuse — same env type, different seed (diagnostic) ──────
        # Pass the shared_cache so session2 can find session1's stored plan.
        # Prime the scene model first so current_environment_identity is set
        # before the ranked query triggers the cache lookup.
        session2 = _make_session(seed=43, request_plan_reuse_cache=shared_cache)
        session2.handle_utterance("what doors do you see?")
        session2.handle_utterance(RANKED_UTTERANCE)

        reuse_verdict = session2.last_plan_reuse_verdict
        metrics["reuse_verdict_same_env_different_seed"] = (
            reuse_verdict is not None and reuse_verdict.verdict == "reuse"
        )

        # reuse_count on the ranked plan entry should have gone up
        metrics["reuse_count_incremented"] = any(
            entry.reuse_count >= 1 for entry in shared_cache.entries.values()
        )

        metrics["reuse_history_recorded"] = any(
            r.verdict == "reuse" for r in shared_cache.history
        )

        # ── Stage 4: Recompile — different env size (required assumption fails) ─
        # 16x16 has different grid_size → env.grid_size assumption violated.
        session3 = _make_session(
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=42,
            request_plan_reuse_cache=shared_cache,
        )
        session3.handle_utterance("what doors do you see?")
        session3.handle_utterance(RANKED_UTTERANCE)

        recompile_verdict = session3.last_plan_reuse_verdict
        metrics["recompile_verdict_different_env_size"] = (
            recompile_verdict is not None and recompile_verdict.verdict == "recompile"
        )

        metrics["recompile_history_recorded"] = any(
            r.verdict == "recompile" for r in shared_cache.history
        )

        # ── Stage 5: Golden path — no interference ───────────────────────────
        golden = _make_session(seed=42)
        golden_response = golden.handle_utterance(GOLDEN_UTTERANCE)
        metrics["golden_path_unaffected"] = (
            "RUN COMPLETE" in golden_response
            and "task_complete=True" in golden_response
        )
        metrics["runtime_llm_calls_during_render_zero"] = (
            golden.last_result is not None
            and golden.last_result.get("runtime_llm_calls_during_render") == 0
        )
        metrics["cache_miss_during_render_zero"] = (
            golden.last_result is not None
            and golden.last_result.get("cache_miss_during_render") == 0
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
