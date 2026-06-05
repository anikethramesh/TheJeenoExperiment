"""Phase 8.4.8 probe: Sequential Concept Intent (LLM Schema Extension).

Verifies that the OperatorIntent schema and station dispatch correctly handle
multi-concept sequential requests via the procedure_recall intent type.

Root cause addressed: OperatorIntent.concept_name is singular — the LLM had no
field to express "run X then Y", so it fell back to unsupported → arbitrator
incorrectly returned synthesizable.

Fix: Add procedure_recall intent type with concept_steps: list[str] field, teach
both the SmokeTestCompiler and LLMCompiler about it, and route it in the station
dispatch chain through the existing _run_procedure pathway.

Checks:
  procedure_recall_in_intent_types        — 'procedure_recall' in OPERATOR_INTENT_TYPES
  operator_intent_has_concept_steps       — OperatorIntent has concept_steps field
  operator_intent_steps_default_none      — default value is None
  smoke_compile_do_x_then_y               — SmokeTestCompiler emits procedure_recall for
                                            'do bingo then scout'
  smoke_compile_x_first_and_then_y        — same for 'execute scout first and then bingo'
  smoke_compile_x_followed_by_y           — same for 'bingo followed by scout'
  smoke_steps_ordered_correctly           — concept_steps matches left-to-right order
  schema_roundtrip_procedure_recall       — OperatorIntent.from_dict encodes/decodes
                                            procedure_recall with concept_steps
  schema_validates_steps_required         — from_dict raises if concept_steps < 2
  procedure_recall_routes_to_execute      — command_from_operator_intent(procedure_recall)
                                            returns procedure_execute command
  handle_utterance_do_x_then_y_runs       — handle_utterance('do bingo then scout')
                                            returns PROCEDURE COMPLETE
  handle_utterance_natural_seq_steps      — executed steps match concept expansions
  unsupported_returns_error_not_synth     — unsupported intent with empty capabilities
                                            returns plain 'I didn't understand' error
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import fields
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
from jeenom.memory import OperationalMemory
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import OPERATOR_INTENT_TYPES, OperatorIntent






def main() -> int:
    metrics: dict[str, bool] = {}

    # ── Schema checks ─────────────────────────────────────────────────────────
    metrics["procedure_recall_in_intent_types"] = "procedure_recall" in OPERATOR_INTENT_TYPES

    field_names = {f.name for f in fields(OperatorIntent)}
    metrics["operator_intent_has_concept_steps"] = "concept_steps" in field_names

    dummy = OperatorIntent(intent_type="unsupported", confidence=0.0, reason="")
    metrics["operator_intent_steps_default_none"] = dummy.concept_steps is None

    # ── SmokeTestCompiler emits procedure_recall ──────────────────────────────
    compiler = SmokeTestCompiler()
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))

    intent_dxy = compiler.compile_operator_intent("do bingo then scout", memory=memory)
    metrics["smoke_compile_do_x_then_y"] = intent_dxy.intent_type == "procedure_recall"

    intent_xfty = compiler.compile_operator_intent(
        "execute scout first and then bingo", memory=memory
    )
    metrics["smoke_compile_x_first_and_then_y"] = intent_xfty.intent_type == "procedure_recall"

    intent_xfy = compiler.compile_operator_intent("bingo followed by scout", memory=memory)
    metrics["smoke_compile_x_followed_by_y"] = intent_xfy.intent_type == "procedure_recall"

    metrics["smoke_steps_ordered_correctly"] = (
        intent_dxy.concept_steps == ["bingo", "scout"]
        and intent_xfty.concept_steps == ["scout", "bingo"]
        and intent_xfy.concept_steps == ["bingo", "scout"]
    )

    # ── Schema round-trip ─────────────────────────────────────────────────────
    intent_obj = OperatorIntent(
        intent_type="procedure_recall",
        concept_steps=["bingo", "scout"],
        confidence=0.8,
        reason="test",
    )
    # Manually build a dict representing what the LLM would return
    raw = {
        "intent_type": "procedure_recall",
        "concept_steps": ["bingo", "scout"],
        "canonical_instruction": None,
        "task_type": None,
        "target": None,
        "knowledge_update": None,
        "reference": None,
        "status_query": None,
        "claim_reference": None,
        "control": None,
        "target_selector": None,
        "grounding_query_plan": None,
        "capability_status": "executable",
        "required_capabilities": [],
        "clear_memory": False,
        "confidence": 0.8,
        "reason": "test",
        "concept_name": None,
        "concept_utterance": None,
        "concept_steps": ["bingo", "scout"],
    }
    restored = OperatorIntent.from_dict(raw)
    metrics["schema_roundtrip_procedure_recall"] = (
        restored.intent_type == "procedure_recall"
        and restored.concept_steps == ["bingo", "scout"]
    )

    # Validation rejects fewer than 2 steps
    from jeenom.schemas import SchemaValidationError
    bad = dict(raw, concept_steps=["bingo"])
    try:
        OperatorIntent.from_dict(bad)
        metrics["schema_validates_steps_required"] = False
    except SchemaValidationError:
        metrics["schema_validates_steps_required"] = True

    # ── Station dispatch: procedure_recall → procedure_execute ────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        sess.handle_utterance("remember bingo means go to the red door")
        sess.handle_utterance("remember scout means go to the green door")

        from jeenom.operator_station import ApprovedCommand
        cmd = sess.command_from_operator_intent(
            OperatorIntent.from_dict(dict(raw, reason="test dispatch")),
            "do bingo then scout",
        )
        metrics["procedure_recall_routes_to_execute"] = (
            isinstance(cmd, ApprovedCommand) and cmd.kind == "procedure_execute"
            and cmd.payload.get("steps") == ["bingo", "scout"]
        )

    # ── End-to-end: handle_utterance with natural-language sequence ───────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess2 = _make_session()
        sess2.handle_utterance("remember bingo means go to the red door")
        sess2.handle_utterance("remember scout means go to the green door")

        resp = sess2.handle_utterance("do bingo then scout")
        metrics["handle_utterance_do_x_then_y_runs"] = "PROCEDURE COMPLETE" in resp

        metrics["handle_utterance_natural_seq_steps"] = (
            sess2.last_result is not None
            and sess2.last_result.get("task", {}).get("instruction") in {
                "go to the red door", "go to the green door"
            }
        )

    # ── unsupported intent returns plain error, not synthesizable ─────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess3 = _make_session()
        resp3 = sess3.handle_utterance("flibbertigibbet wobbleforth the grimbaz")
        metrics["unsupported_returns_error_not_synth"] = (
            "synthesizable" not in resp3.lower()
            and "didn't understand" in resp3.lower()
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
