"""Phase 8.5.5 probe: Motor Implicit Sequence Splitting.

Verifies that the deterministic fast path compiler correctly expands implicit 
motor sequences (like "go left") and bare turns (like "turn two times") 
before sequence parsing.

Checks:
  smoke_compile_go_left_3         — 'go left three steps' emits motor_sequence
  smoke_compile_go_left_1         — 'go left one step' emits motor_sequence
  smoke_compile_turn_2_go_fwd_1   — 'turn two times and go forward once' emits motor_sequence
  smoke_compile_turn_around       — 'turn around' emits motor_command (turn_right x 2)
  smoke_compile_go_back           — 'go back 2 steps' emits motor_sequence
  handle_utterance_go_left_3_runs — handle_utterance executes sequence properly
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
from jeenom.memory import OperationalMemory
from jeenom.operator_station import OperatorStationSession

def main() -> int:
    metrics: dict[str, bool] = {}

    compiler = SmokeTestCompiler()
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))

    # 1. "go left three steps"
    intent_left3 = compiler.compile_operator_intent("go left three steps", memory=memory)
    metrics["smoke_compile_go_left_3"] = (
        intent_left3.intent_type == "motor_sequence"
        and intent_left3.utterance_steps == ["turn_left:1", "move_forward:3"]
    )

    # 2. "go left one step"
    intent_left1 = compiler.compile_operator_intent("go left one step", memory=memory)
    metrics["smoke_compile_go_left_1"] = (
        intent_left1.intent_type == "motor_sequence"
        and intent_left1.utterance_steps == ["turn_left:1", "move_forward:1"]
    )

    # 3. "turn two times and go forward once"
    intent_turn2fwd1 = compiler.compile_operator_intent("turn two times and go forward once", memory=memory)
    metrics["smoke_compile_turn_2_go_fwd_1"] = (
        intent_turn2fwd1.intent_type == "motor_sequence"
        and intent_turn2fwd1.utterance_steps == ["turn_right:2", "move_forward:1"]
    )

    # 4. "turn around" (standalone motor command)
    intent_turn_around = compiler.compile_operator_intent("turn around", memory=memory)
    metrics["smoke_compile_turn_around"] = (
        intent_turn_around.intent_type == "motor_command"
        and intent_turn_around.action_name == "turn_right"
        and intent_turn_around.repeat_count == 2
    )

    # 5. "go back 2 steps"
    intent_go_back = compiler.compile_operator_intent("go back 2 steps", memory=memory)
    metrics["smoke_compile_go_back"] = (
        intent_go_back.intent_type == "motor_sequence"
        and intent_go_back.utterance_steps == ["turn_right:2", "move_forward:2"]
    )

    # 6. End-to-end station execution
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        resp = sess.handle_utterance("go left three steps")
        # motor sequences execute step-by-step and return a combined MOTOR COMPLETE response
        metrics["handle_utterance_go_left_3_runs"] = "MOTOR COMPLETE" in resp and "move forward × 3" in resp

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
