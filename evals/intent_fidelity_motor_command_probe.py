"""Phase 8.5.0 probe: Motor Command Composition.

Verifies that the station can execute direct motor-primitive commands like
'go straight for 3 steps' (move_forward × 3) without going through task
planning (Cortex/Spine A* path). This closes the abstraction ladder gap:

  motor primitives  → move_forward, turn_right (ACTION_PRIMITIVES / Spine)
  motor commands    → "go straight for 3 steps" → move_forward × 3 (NEW)
  task primitives   → "go to the red door" → A* plan → motor sequence (existing)

Checks:
  motor_command_in_intent_types     — 'motor_command' in OPERATOR_INTENT_TYPES
  operator_intent_has_action_name   — OperatorIntent has action_name field
  operator_intent_has_repeat_count  — OperatorIntent has repeat_count field
  action_name_default_none          — default value of action_name is None
  repeat_count_default_none         — default value of repeat_count is None
  smoke_compile_go_straight_3       — SmokeTestCompiler emits motor_command for
                                      'go straight for 3 steps'
  smoke_compile_turn_right_twice    — same for 'turn right twice'
  smoke_compile_turn_left_4         — same for 'turn left 4 times'
  smoke_action_name_correct         — action_name == 'move_forward' for go-straight
  smoke_repeat_count_correct        — repeat_count == 3 for 'go straight for 3 steps'
  schema_roundtrip_motor_command    — OperatorIntent.from_dict round-trips correctly
  schema_validates_action_required  — missing action_name raises SchemaValidationError
  schema_validates_count_ge_1       — repeat_count=0 raises SchemaValidationError
  motor_command_routes_to_execute   — command_from_operator_intent returns motor_execute
  handle_utterance_go_straight_runs — handle_utterance('go straight for 3 steps')
                                      returns 'MOTOR COMPLETE'
  handle_utterance_result_set       — last_result is populated after motor execution
  handle_utterance_turn_right       — handle_utterance('turn right twice') works
  task_like_pickup_not_motor        — 'pick up the red key' does not bypass task readiness
  task_like_toggle_not_motor        — 'toggle the blue door' does not bypass task readiness
  task_instruction_no_regression    — 'go to the red door' still returns RUN COMPLETE
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import fields
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
from jeenom.memory import OperationalMemory
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import OPERATOR_INTENT_TYPES, OperatorIntent, SchemaValidationError






def main() -> int:
    metrics: dict[str, bool] = {}

    # ── Schema checks ─────────────────────────────────────────────────────────
    metrics["motor_command_in_intent_types"] = "motor_command" in OPERATOR_INTENT_TYPES

    field_names = {f.name for f in fields(OperatorIntent)}
    metrics["operator_intent_has_action_name"] = "action_name" in field_names
    metrics["operator_intent_has_repeat_count"] = "repeat_count" in field_names

    dummy = OperatorIntent(intent_type="unsupported", confidence=0.0, reason="")
    metrics["action_name_default_none"] = dummy.action_name is None
    metrics["repeat_count_default_none"] = dummy.repeat_count is None

    # ── SmokeTestCompiler emits motor_command ─────────────────────────────────
    compiler = SmokeTestCompiler()
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))

    intent_fwd = compiler.compile_operator_intent("go straight for 3 steps", memory=memory)
    metrics["smoke_compile_go_straight_3"] = intent_fwd.intent_type == "motor_command"
    metrics["smoke_action_name_correct"] = intent_fwd.action_name == "move_forward"
    metrics["smoke_repeat_count_correct"] = intent_fwd.repeat_count == 3

    intent_tr = compiler.compile_operator_intent("turn right twice", memory=memory)
    metrics["smoke_compile_turn_right_twice"] = intent_tr.intent_type == "motor_command"

    intent_tl = compiler.compile_operator_intent("turn left 4 times", memory=memory)
    metrics["smoke_compile_turn_left_4"] = intent_tl.intent_type == "motor_command"

    # ── Schema round-trip ─────────────────────────────────────────────────────
    raw = {
        "intent_type": "motor_command",
        "action_name": "move_forward",
        "repeat_count": 3,
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
        "confidence": 0.9,
        "reason": "test",
        "concept_name": None,
        "concept_utterance": None,
        "concept_steps": None,
        "utterance_steps": None,
    }
    restored = OperatorIntent.from_dict(raw)
    metrics["schema_roundtrip_motor_command"] = (
        restored.intent_type == "motor_command"
        and restored.action_name == "move_forward"
        and restored.repeat_count == 3
    )

    bad_no_action = dict(raw, action_name=None)
    try:
        OperatorIntent.from_dict(bad_no_action)
        metrics["schema_validates_action_required"] = False
    except SchemaValidationError:
        metrics["schema_validates_action_required"] = True

    bad_count = dict(raw, repeat_count=0)
    try:
        OperatorIntent.from_dict(bad_count)
        metrics["schema_validates_count_ge_1"] = False
    except SchemaValidationError:
        metrics["schema_validates_count_ge_1"] = True

    # ── Station dispatch: motor_command → motor_execute ───────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        from jeenom.operator_station import ApprovedCommand
        cmd = sess.turn_orchestrator.dispatch(sess, 
            OperatorIntent.from_dict(raw),
            "go straight for 3 steps",
        )
        metrics["motor_command_routes_to_execute"] = (
            isinstance(cmd, ApprovedCommand)
            and cmd.kind == "motor_execute"
            and cmd.payload.get("action") == "move_forward"
            and cmd.payload.get("count") == 3
        )

    # ── End-to-end: handle_utterance with motor command ───────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess2 = _make_session()
        resp = sess2.handle_utterance("go straight for 3 steps")
        metrics["handle_utterance_go_straight_runs"] = "MOTOR COMPLETE" in resp
        metrics["handle_utterance_result_set"] = sess2.last_result is not None

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess3 = _make_session()
        resp3 = sess3.handle_utterance("turn right twice")
        metrics["handle_utterance_turn_right"] = "MOTOR COMPLETE" in resp3

    # ── Safety regression: task-like object requests must not become raw motor commands.
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess_pickup = _make_session()
        resp_pickup = sess_pickup.handle_utterance("pick up the red key")
        metrics["task_like_pickup_not_motor"] = "MOTOR COMPLETE" not in resp_pickup

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess_toggle = _make_session()
        resp_toggle = sess_toggle.handle_utterance("toggle the blue door")
        metrics["task_like_toggle_not_motor"] = "MOTOR COMPLETE" not in resp_toggle

    # ── Regression: task_instruction still works ──────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess4 = _make_session()
        resp4 = sess4.handle_utterance("go to the red door")
        metrics["task_instruction_no_regression"] = "RUN COMPLETE" in resp4 or "TASK COMPLETE" in resp4

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
