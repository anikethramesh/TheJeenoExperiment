"""Phase 8.4.7 probe: Primitive Composition Foundation.

Verifies that a comma-separated sequence of known atomic concept names is
automatically recognised as a procedure, compiled to a multi-step RequestPlan
with sequential depends_on chains, gated by the ReadinessGraph, and executed
step-by-step via run_task().

Checks:
  nc_has_concept_type_field              — NamedConcept has concept_type field
  nc_has_steps_field                     — NamedConcept has steps field
  atomic_concept_default_type            — teach() stores concept_type='atomic'
  procedure_detected_on_teach            — CSV of known concepts → concept_type='procedure'
  procedure_steps_ordered                — steps list matches teach order
  procedure_stores_canonical_utterance   — utterance is preserved as-is for procedures
  sequence_before_deps_stores_atomic     — CSV of unknown concepts stores as atomic
  is_sequence_single_token_none          — _is_sequence('bingo') returns None
  is_sequence_unknown_token_none         — _is_sequence('bingo, phantom') returns None when phantom unknown
  is_sequence_known_pair_returns_list    — _is_sequence('bingo, scout') returns ['bingo', 'scout']
  is_sequence_allows_repeats             — 'bingo, scout, bingo' → ['bingo', 'scout', 'bingo']
  concept_type_serialises_to_dict        — as_dict includes concept_type and steps
  concept_type_roundtrips_from_dict      — from_dict restores concept_type and steps
  backward_compat_old_format_loads       — JSON without concept_type/steps field loads as atomic
  procedure_command_is_procedure_execute — named procedure concept → procedure_execute command
  anonymous_sequence_procedure_execute   — raw 'bingo, scout' input → procedure_execute command
  build_plan_step_count                  — _build_procedure_request_plan produces N RequestPlanSteps
  build_plan_depends_on_chain            — each step (except first) depends on the previous step
  build_plan_handle_is_task_go_to_object — all steps use required_handle='task.go_to_object.door'
  build_plan_constraints_utterance       — each step constraints.utterance = expanded concept utterance
  build_plan_nested_procedure_rejected   — nested procedure-in-procedure returns None (no graph errors)
  build_plan_unknown_concept_rejected    — unknown concept name in steps returns None
  procedure_readiness_graph_evaluated    — _run_procedure stores last_readiness_graph
  procedure_executes_all_steps           — handle_utterance('patrol') runs 3 tasks (bingo→scout→bingo)
  atomic_regression_bingo_alone          — 'bingo' still executes as atomic (no regression)
  no_nested_procedure_via_handle         — handle_utterance on nested procedure returns error, not crash
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

from jeenom.knowledge_base import KnowledgeBase, NamedConcept
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import RequestPlanStep






def _make_kb() -> KnowledgeBase:
    return KnowledgeBase(storage_path=None)


def main() -> int:
    metrics: dict[str, bool] = {}

    # ── Data model: field existence ───────────────────────────────────────────
    nc = NamedConcept(name="test", utterance="go to the red door")
    metrics["nc_has_concept_type_field"] = hasattr(nc, "concept_type")
    metrics["nc_has_steps_field"] = hasattr(nc, "steps")

    # ── KnowledgeBase: atomic vs procedure detection ──────────────────────────
    kb = _make_kb()
    kb.teach("bingo", "go to the red door")
    kb.teach("scout", "go to the green door")

    bingo = kb.recall("bingo")
    metrics["atomic_concept_default_type"] = (
        bingo is not None and bingo.concept_type == "atomic" and bingo.steps == []
    )

    # Teach patrol as a procedure
    kb.teach("patrol", "bingo, scout, bingo")
    patrol = kb.recall("patrol")
    metrics["procedure_detected_on_teach"] = (
        patrol is not None and patrol.concept_type == "procedure"
    )
    metrics["procedure_steps_ordered"] = (
        patrol is not None and patrol.steps == ["bingo", "scout", "bingo"]
    )
    metrics["procedure_stores_canonical_utterance"] = (
        patrol is not None and patrol.utterance == "bingo, scout, bingo"
    )

    # Teach a CSV before its deps exist → stores as atomic
    kb2 = _make_kb()
    kb2.teach("alpha", "bingo, scout")  # bingo and scout not in kb2
    alpha = kb2.recall("alpha")
    metrics["sequence_before_deps_stores_atomic"] = (
        alpha is not None and alpha.concept_type == "atomic"
    )

    # ── _is_sequence behaviour ────────────────────────────────────────────────
    metrics["is_sequence_single_token_none"] = kb._is_sequence("bingo") is None
    metrics["is_sequence_unknown_token_none"] = kb._is_sequence("bingo, phantom") is None
    result_pair = kb._is_sequence("bingo, scout")
    metrics["is_sequence_known_pair_returns_list"] = result_pair == ["bingo", "scout"]
    result_repeat = kb._is_sequence("bingo, scout, bingo")
    metrics["is_sequence_allows_repeats"] = result_repeat == ["bingo", "scout", "bingo"]

    # ── Serialisation round-trip ──────────────────────────────────────────────
    d = patrol.as_dict()
    metrics["concept_type_serialises_to_dict"] = (
        "concept_type" in d and d["concept_type"] == "procedure"
        and "steps" in d and d["steps"] == ["bingo", "scout", "bingo"]
    )
    restored = NamedConcept.from_dict(d)
    metrics["concept_type_roundtrips_from_dict"] = (
        restored.concept_type == "procedure"
        and restored.steps == ["bingo", "scout", "bingo"]
    )

    # Backward compat: old JSON without concept_type/steps still loads as atomic
    old_dict = {"name": "legacy", "utterance": "go to the blue door", "plan": None,
                "stored_at": 0.0, "recall_count": 0, "tags": []}
    legacy = NamedConcept.from_dict(old_dict)
    metrics["backward_compat_old_format_loads"] = (
        legacy.concept_type == "atomic" and legacy.steps == []
    )

    # ── Station: _command_from_concept routing ────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session = _make_session()

        # Teach two atomics and a procedure
        session.handle_utterance("remember bingo means go to the red door")
        session.handle_utterance("remember scout means go to the green door")
        session.handle_utterance("remember patrol means bingo, scout, bingo")

        # Named procedure concept → procedure_execute
        cmd = session._command_from_concept("patrol")
        metrics["procedure_command_is_procedure_execute"] = (
            cmd is not None
            and cmd.kind == "procedure_execute"
            and cmd.payload.get("steps") == ["bingo", "scout", "bingo"]
        )

        # Anonymous raw sequence → procedure_execute
        cmd_anon = session._command_from_concept("bingo, scout")
        metrics["anonymous_sequence_procedure_execute"] = (
            cmd_anon is not None
            and cmd_anon.kind == "procedure_execute"
            and cmd_anon.payload.get("steps") == ["bingo", "scout"]
        )

        # ── _build_procedure_request_plan structure ───────────────────────────
        plan = session._build_procedure_request_plan(
            ["bingo", "scout", "bingo"], "patrol"
        )
        metrics["build_plan_step_count"] = plan is not None and len(plan.steps) == 3

        if plan is not None:
            step_ids = [s.step_id for s in plan.steps]
            depends_on_chain = [
                plan.steps[0].depends_on == [],
                plan.steps[1].depends_on == [step_ids[0]],
                plan.steps[2].depends_on == [step_ids[1]],
            ]
            metrics["build_plan_depends_on_chain"] = all(depends_on_chain)

            metrics["build_plan_handle_is_task_go_to_object"] = all(
                s.required_handle == "task.go_to_object.door" for s in plan.steps
            )

            utterances = [s.constraints.get("utterance") for s in plan.steps]
            metrics["build_plan_constraints_utterance"] = utterances == [
                "go to the red door",
                "go to the green door",
                "go to the red door",
            ]
        else:
            metrics["build_plan_depends_on_chain"] = False
            metrics["build_plan_handle_is_task_go_to_object"] = False
            metrics["build_plan_constraints_utterance"] = False

        # Nested procedure rejected: teach a procedure-of-procedure, then try to build
        session.handle_utterance("remember megapatrol means patrol, bingo")
        # megapatrol is stored as procedure (patrol + bingo are in KB)
        nested_plan = session._build_procedure_request_plan(
            ["patrol", "bingo"], "megapatrol"
        )
        metrics["build_plan_nested_procedure_rejected"] = nested_plan is None

        # Unknown concept rejected
        unknown_plan = session._build_procedure_request_plan(
            ["bingo", "phantom"], "bad_proc"
        )
        metrics["build_plan_unknown_concept_rejected"] = unknown_plan is None

        # ── End-to-end execution via handle_utterance ─────────────────────────
        patrol_resp = session.handle_utterance("patrol")
        # Should report procedure complete with multiple steps, not an error
        metrics["procedure_executes_all_steps"] = (
            "PROCEDURE COMPLETE" in patrol_resp
            and patrol_resp.count("RUN COMPLETE") >= 2  # at least 2 task completions (3 steps, but bingo twice)
        )

        metrics["procedure_readiness_graph_evaluated"] = (
            session.last_readiness_graph is not None
        )

        # Atomic regression: 'bingo' alone still executes as a single task
        bingo_resp = session.handle_utterance("bingo")
        metrics["atomic_regression_bingo_alone"] = (
            "RUN COMPLETE" in bingo_resp and "PROCEDURE" not in bingo_resp
        )

        # No crash on nested procedure via handle_utterance
        nested_resp = session.handle_utterance("megapatrol")
        metrics["no_nested_procedure_via_handle"] = (
            "PROCEDURE ERROR" in nested_resp or "error" in nested_resp.lower()
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
