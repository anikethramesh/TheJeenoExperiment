"""Phase 8.9 probe: Command Registry.

Verifies that jeenom/command_registry.py correctly encodes all motor and sensory
commands, and that Cortex/Spine use it instead of hardcoded if/elif chains.

Checks:
  motor_commands_complete       — MOTOR_COMMANDS has all expected skills
  direct_action_skills_derived  — DIRECT_ACTION_SKILLS derived from registry
  sensory_commands_complete     — SENSORY_COMMANDS has all expected steps
  motor_navigate_primitives     — navigate_to_object has correct primitive_sequence
  motor_done_primitives         — done has correct primitive_sequence
  motor_direct_primitives       — turn_right has [turn_right] as primitive_sequence
  sensory_locate_claims         — locate_object has correct required_claims
  sensory_navigate_claims       — navigate_to_object has correct required_claims
  sensory_verify_claims         — verify_adjacent has correct required_claims
  evidence_needs_locate         — evidence_needs_for_step('locate_object') correct
  evidence_needs_navigate       — evidence_needs_for_step('navigate_to_object') correct
  evidence_needs_verify         — evidence_needs_for_step('verify_adjacent') correct
  evidence_needs_default        — evidence_needs_for_step('unknown') returns default
  canonical_primitives_nav      — canonical_primitives_for_skill('navigate_to_object') correct
  canonical_primitives_done     — canonical_primitives_for_skill('done') correct
  canonical_primitives_unknown  — canonical_primitives_for_skill('bogus') returns None
  cortex_uses_registry          — Cortex.make_evidence_frame source has no hardcoded if/elif
  spine_uses_registry           — Spine._validate_template source has no hardcoded skill names
"""
from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.command_registry import (
    DIRECT_ACTION_SKILLS,
    MOTOR_COMMANDS,
    SENSORY_COMMANDS,
    canonical_primitives_for_skill,
    evidence_needs_for_step,
)
from jeenom.cortex import Cortex
from jeenom.spine import MiniGridSpine


def main() -> int:
    metrics: dict[str, bool] = {}

    # ── Motor commands completeness ────────────────────────────────────────────
    expected_motor = {
        "navigate_to_object", "done",
        "turn_left", "turn_right", "move_forward",
        "pickup", "drop", "toggle", "abort",
    }
    metrics["motor_commands_complete"] = expected_motor.issubset(MOTOR_COMMANDS)

    metrics["direct_action_skills_derived"] = DIRECT_ACTION_SKILLS == frozenset({
        "turn_left", "turn_right", "move_forward", "pickup", "drop", "toggle", "done",
    })

    # ── Sensory commands completeness ──────────────────────────────────────────
    expected_sensory = {"locate_object", "navigate_to_object", "verify_adjacent"}
    metrics["sensory_commands_complete"] = expected_sensory.issubset(SENSORY_COMMANDS)

    # ── Motor primitive sequences ──────────────────────────────────────────────
    metrics["motor_navigate_primitives"] = (
        MOTOR_COMMANDS["navigate_to_object"].primitive_sequence
        == ("plan_grid_path", "execute_next_path_action")
    )
    metrics["motor_done_primitives"] = (
        MOTOR_COMMANDS["done"].primitive_sequence == ("done",)
    )
    metrics["motor_direct_primitives"] = (
        MOTOR_COMMANDS["turn_right"].primitive_sequence == ("turn_right",)
    )

    # ── Sensory required claims ────────────────────────────────────────────────
    metrics["sensory_locate_claims"] = set(
        SENSORY_COMMANDS["locate_object"].required_claims
    ) == {"object_location", "agent_pose", "occupancy_grid"}

    metrics["sensory_navigate_claims"] = set(
        SENSORY_COMMANDS["navigate_to_object"].required_claims
    ) == {"object_location", "agent_pose", "occupancy_grid", "adjacency_to_target"}

    metrics["sensory_verify_claims"] = set(
        SENSORY_COMMANDS["verify_adjacent"].required_claims
    ) == {"agent_pose", "object_location", "adjacency_to_target"}

    # ── evidence_needs_for_step ────────────────────────────────────────────────
    metrics["evidence_needs_locate"] = set(evidence_needs_for_step("locate_object")) == {
        "object_location", "agent_pose", "occupancy_grid"
    }
    metrics["evidence_needs_navigate"] = set(evidence_needs_for_step("navigate_to_object")) == {
        "object_location", "agent_pose", "occupancy_grid", "adjacency_to_target"
    }
    metrics["evidence_needs_verify"] = set(evidence_needs_for_step("verify_adjacent")) == {
        "agent_pose", "object_location", "adjacency_to_target"
    }
    metrics["evidence_needs_default"] = evidence_needs_for_step("completely_unknown_step") == [
        "adjacency_to_target"
    ]

    # ── canonical_primitives_for_skill ─────────────────────────────────────────
    metrics["canonical_primitives_nav"] = canonical_primitives_for_skill("navigate_to_object") == [
        "plan_grid_path", "execute_next_path_action"
    ]
    metrics["canonical_primitives_done"] = canonical_primitives_for_skill("done") == ["done"]
    metrics["canonical_primitives_unknown"] = canonical_primitives_for_skill("bogus") is None

    # ── Source code: no hardcoded if/elif in Cortex.make_evidence_frame ────────
    cortex_src = inspect.getsource(Cortex.make_evidence_frame)
    metrics["cortex_uses_registry"] = (
        "evidence_needs_for_step" in cortex_src
        and "locate_object" not in cortex_src
    )

    # ── Source code: no hardcoded skill names in Spine._validate_template ──────
    spine_validate_src = inspect.getsource(MiniGridSpine._validate_template)
    metrics["spine_uses_registry"] = (
        "canonical_primitives_for_skill" in spine_validate_src
        and "navigate_to_object" not in spine_validate_src
    )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
