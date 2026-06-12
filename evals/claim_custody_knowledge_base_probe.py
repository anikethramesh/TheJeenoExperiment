"""Phase 8.35 probe: Named Concept Knowledge Base.

Proves that the operator can define named shorthand concepts that expand to
full utterances, carry pre-compiled plans, persist across sessions, and
integrate transparently with the Phase 8.3 plan-reuse cache.

Checks:
  concept_stored_after_teach          — KB has entry after 'remember bingo means ...'
  plan_precompiled_at_teach_time      — plan was compiled and cached during teach
  plan_in_reuse_cache                 — plan also seeded into PlanReuseCache
  concept_recall_runs_task            — saying 'bingo' executes the task
  recall_count_incremented            — recall_count goes up on each use
  concept_list_query                  — 'list concepts' returns CONCEPTS report
  concept_forget_works                — after forget, concept is gone
  concept_persisted_across_sessions   — new session at same memory_root reloads concept
  plan_restored_after_reload          — reloaded concept still has plan
  define_syntax_works                 — 'define X as Y' also teaches a concept
  search_finds_partial_match          — search('bing') finds 'bingo'
  clear_memory_clears_concepts        — reset(clear_memory=True) removes all concepts
  forget_everything_nlu_clears_memory — 'forget everything' utterance routes to hard reset
  golden_path_unaffected              — 'go to the red door' unaffected
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from harness import build_env as _build_env, make_session as _make_session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession






def main() -> int:
    metrics: dict[str, bool] = {}

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        memory_root = Path(tempfile.mkdtemp())

        # ── Stage 1: Teach a concept ─────────────────────────────────────────
        session1 = _make_session(memory_root=memory_root)
        teach_resp = session1.handle_utterance("remember bingo means go to the red door")

        metrics["concept_stored_after_teach"] = (
            "CONCEPT STORED" in teach_resp
            and len(session1.knowledge_base.all_concepts()) == 1
        )

        bingo = session1.knowledge_base.all_concepts()[0]
        metrics["plan_precompiled_at_teach_time"] = bingo.plan is not None

        metrics["plan_in_reuse_cache"] = (
            bingo.plan is not None
            and len(session1.request_plan_reuse_cache.entries) > 0
        )

        # ── Stage 2: Recall executes the task ────────────────────────────────
        recall_resp = session1.handle_utterance("bingo")
        metrics["concept_recall_runs_task"] = (
            "RUN COMPLETE" in recall_resp and "task_complete=True" in recall_resp
        )

        bingo_after = session1.knowledge_base.recall("bingo")
        metrics["recall_count_incremented"] = (
            bingo_after is not None and bingo_after.recall_count >= 2
        )

        # ── Stage 3: List concepts ───────────────────────────────────────────
        list_resp = session1.handle_utterance("list concepts")
        metrics["concept_list_query"] = (
            "CONCEPTS" in list_resp and "bingo" in list_resp
        )

        # ── Stage 4: Forget ──────────────────────────────────────────────────
        forget_resp = session1.handle_utterance("forget concept bingo")
        metrics["concept_forget_works"] = (
            "CONCEPT FORGOTTEN" in forget_resp
            and len(session1.knowledge_base.all_concepts()) == 0
        )

        # ── Stage 5: Persistence across sessions ─────────────────────────────
        # Teach in session2, then reload in session3.
        session2 = _make_session(memory_root=memory_root)
        session2.handle_utterance("remember bingo means go to the red door")

        session3 = _make_session(memory_root=memory_root)
        reloaded = session3.knowledge_base.recall("bingo")
        metrics["concept_persisted_across_sessions"] = (
            reloaded is not None and reloaded.utterance == "go to the red door"
        )
        metrics["plan_restored_after_reload"] = (
            reloaded is not None and reloaded.plan is not None
        )

        # ── Stage 6: Alternate teach syntax ─────────────────────────────────
        session4 = _make_session(memory_root=Path(tempfile.mkdtemp()))
        define_resp = session4.handle_utterance("define alpha as go to the blue door")
        metrics["define_syntax_works"] = (
            "CONCEPT STORED" in define_resp
            and session4.knowledge_base.recall("alpha") is not None
        )

        # ── Stage 7: Search ──────────────────────────────────────────────────
        session5 = _make_session(memory_root=Path(tempfile.mkdtemp()))
        session5.handle_utterance("remember bingo means go to the red door")
        results = session5.knowledge_base.search("bing")
        metrics["search_finds_partial_match"] = (
            len(results) >= 1 and any(c.name == "bingo" for c in results)
        )

        # ── Stage 8: clear memory clears concepts (direct API) ──────────────
        session6 = _make_session(memory_root=Path(tempfile.mkdtemp()))
        session6.handle_utterance("remember bingo means go to the red door")
        session6.reset(clear_memory=True)
        metrics["clear_memory_clears_concepts"] = (
            len(session6.knowledge_base.all_concepts()) == 0
        )

        # ── Stage 8b: "forget everything" NLU path clears memory ─────────────
        # Regression: IntentCache concept_forget pattern intercepted "forget everything"
        # before classify_utterance could route it as a hard reset.
        session7 = _make_session(memory_root=Path(tempfile.mkdtemp()))
        session7.handle_utterance("remember bingo means go to the red door")
        forget_resp = session7.handle_utterance("forget everything")
        metrics["forget_everything_nlu_clears_memory"] = (
            len(session7.knowledge_base.all_concepts()) == 0
        )

        # ── Stage 9: Golden path unaffected ─────────────────────────────────
        golden = _make_session(memory_root=Path(tempfile.mkdtemp()))
        golden_resp = golden.handle_utterance("go to the red door")
        metrics["golden_path_unaffected"] = (
            "RUN COMPLETE" in golden_resp and "task_complete=True" in golden_resp
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
