"""Phase 8.5.2 probe: Motor count words and multi-motor sequences.

Verifies that the SmokeTestCompiler correctly parses word-number counts
("two steps", "three times") and splits compound motor utterances into
ordered motor sequences.

Checks:
  count_two_steps           — 'go straight two steps' → motor_command repeat_count=2
  count_three_steps         — 'go straight three steps' → repeat_count=3
  count_twice               — 'turn right twice' → repeat_count=2  (already worked)
  count_digit               — 'move forward 4 times' → repeat_count=4
  count_default_one         — 'go straight' → repeat_count=1
  smoke_seq_two_actions     — 'turn right once, and go straight two times' → motor_sequence
  smoke_seq_steps_count     — motor_sequence utterance_steps has 2 entries
  smoke_seq_first_action    — first step encodes turn_right
  smoke_seq_second_action   — second step encodes move_forward
  smoke_seq_first_count     — first step count=1
  smoke_seq_second_count    — second step count=2
  handle_count_two_steps    — handle_utterance returns MOTOR COMPLETE (move forward × 2)
  handle_count_three        — handle_utterance returns MOTOR COMPLETE (move forward × 3)
  handle_motor_seq          — handle_utterance for compound motor executes both actions
  regression_single_right   — 'turn right' still returns MOTOR COMPLETE
  regression_seq_no_reg     — 'go to the red door then go to the blue door' still PROCEDURE
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


def _decode_motor_step(step: str) -> tuple[str, int]:
    """Parse 'action_name:count' encoded step from motor_sequence utterance_steps."""
    parts = step.split(":", 1)
    return parts[0], int(parts[1])


def main() -> int:
    metrics: dict[str, bool] = {}

    compiler = SmokeTestCompiler()
    memory = OperationalMemory(root=Path(tempfile.mkdtemp()))

    # ── Count word parsing ────────────────────────────────────────────────────
    i1 = compiler.compile_operator_intent("go straight two steps", memory=memory)
    metrics["count_two_steps"] = (
        i1.intent_type == "motor_command"
        and i1.action_name == "move_forward"
        and i1.repeat_count == 2
    )

    i2 = compiler.compile_operator_intent("go straight three steps", memory=memory)
    metrics["count_three_steps"] = (
        i2.intent_type == "motor_command"
        and i2.action_name == "move_forward"
        and i2.repeat_count == 3
    )

    i3 = compiler.compile_operator_intent("turn right twice", memory=memory)
    metrics["count_twice"] = (
        i3.intent_type == "motor_command"
        and i3.action_name == "turn_right"
        and i3.repeat_count == 2
    )

    i4 = compiler.compile_operator_intent("move forward 4 times", memory=memory)
    metrics["count_digit"] = (
        i4.intent_type == "motor_command"
        and i4.action_name == "move_forward"
        and i4.repeat_count == 4
    )

    i5 = compiler.compile_operator_intent("go straight", memory=memory)
    metrics["count_default_one"] = (
        i5.intent_type == "motor_command"
        and i5.action_name == "move_forward"
        and i5.repeat_count == 1
    )

    # ── Multi-motor sequence detection ────────────────────────────────────────
    i6 = compiler.compile_operator_intent(
        "turn right once, and go straight two times", memory=memory
    )
    metrics["smoke_seq_two_actions"] = i6.intent_type == "motor_sequence"
    steps = list(i6.utterance_steps or [])
    metrics["smoke_seq_steps_count"] = len(steps) == 2
    if len(steps) == 2:
        a0, c0 = _decode_motor_step(steps[0])
        a1, c1 = _decode_motor_step(steps[1])
        metrics["smoke_seq_first_action"] = a0 == "turn_right"
        metrics["smoke_seq_second_action"] = a1 == "move_forward"
        metrics["smoke_seq_first_count"] = c0 == 1
        metrics["smoke_seq_second_count"] = c1 == 2
    else:
        metrics["smoke_seq_first_action"] = False
        metrics["smoke_seq_second_action"] = False
        metrics["smoke_seq_first_count"] = False
        metrics["smoke_seq_second_count"] = False

    # ── End-to-end: handle_utterance ─────────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()

        resp1 = sess.handle_utterance("go straight two steps")
        metrics["handle_count_two_steps"] = (
            "MOTOR COMPLETE" in resp1 and "move forward" in resp1 and "× 2" in resp1
        )

        resp2 = sess.handle_utterance("go straight three steps")
        metrics["handle_count_three"] = (
            "MOTOR COMPLETE" in resp2 and "move forward" in resp2 and "× 3" in resp2
        )

        resp3 = sess.handle_utterance("turn right once, and go straight two times")
        metrics["handle_motor_seq"] = (
            "MOTOR" in resp3
            and "turn" in resp3.lower()
            and ("move forward" in resp3.lower() or "straight" in resp3.lower())
        )

    # ── Regression checks ─────────────────────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess2 = _make_session()
        resp4 = sess2.handle_utterance("turn right")
        metrics["regression_single_right"] = "MOTOR COMPLETE" in resp4

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess3 = _make_session()
        resp5 = sess3.handle_utterance("go to the red door then go to the blue door")
        metrics["regression_seq_no_reg"] = "PROCEDURE COMPLETE" in resp5

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
