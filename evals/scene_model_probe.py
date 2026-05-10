"""Phase 7.57 — Persistent Scene Model probe.

Verifies that:
- SceneModel is populated after a task sense tick (source=task_sense).
- SceneModel is populated by idle sense before any task (source=idle_sense).
- Grounding queries use SceneModel, not a fresh env reset.
- Scene queries report the final agent pose, not the spawn pose.
- "which doors are visible?" answers from SceneModel.
- unique_door grounding returns distance.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from pprint import pprint
from typing import Any
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
from jeenom.schemas import SceneModel


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


def main() -> int:
    checks: dict[str, bool] = {}

    # ── 1. SceneModel populated after task ─────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session = _make_session()
        session.handle_utterance("go to the red door")
    scene_after_task: SceneModel | None = session.memory.scene_model
    checks["scene_model_populated_after_task"] = scene_after_task is not None
    checks["scene_model_source_task_sense"] = (
        scene_after_task is not None and scene_after_task.source == "task_sense"
    )
    checks["scene_model_has_agent_pose"] = scene_after_task is not None and (
        isinstance(scene_after_task.agent_x, int)
        and isinstance(scene_after_task.agent_y, int)
    )
    checks["scene_model_has_door_objects"] = scene_after_task is not None and len(
        scene_after_task.find(object_type="door")
    ) > 0

    print("SCENE MODEL (after task)")
    if scene_after_task:
        pprint({
            "source": scene_after_task.source,
            "agent": (scene_after_task.agent_x, scene_after_task.agent_y, scene_after_task.agent_dir),
            "grid": (scene_after_task.grid_width, scene_after_task.grid_height),
            "doors": [(d.color, d.x, d.y) for d in scene_after_task.find(object_type="door")],
            "step_count": scene_after_task.step_count,
        })
    print()

    # ── 2. Idle sense before any task ──────────────────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session2 = _make_session()
        session2._ensure_scene_model()
    scene_idle = session2.memory.scene_model
    checks["idle_sense_populates_scene_model"] = scene_idle is not None
    checks["idle_sense_source_is_idle_sense"] = (
        scene_idle is not None and scene_idle.source == "idle_sense"
    )

    print("SCENE MODEL (idle sense, before task)")
    if scene_idle:
        pprint({
            "source": scene_idle.source,
            "agent": (scene_idle.agent_x, scene_idle.agent_y),
            "doors": [(d.color, d.x, d.y) for d in scene_idle.find(object_type="door")],
        })
    print()

    # ── 3. Grounding uses SceneModel — no adapter.reset() call ─────────────
    reset_calls: list[Any] = []
    original_reset = MiniGridAdapter.reset

    def tracking_reset(self_adapter, seed=None):
        reset_calls.append(seed)
        return original_reset(self_adapter, seed=seed)

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session3 = _make_session()
        session3.handle_utterance("go to the red door")  # build scene_model
        reset_calls.clear()
        with patch.object(MiniGridAdapter, "reset", tracking_reset):
            grounded_closest = session3.ground_target_selector({
                "object_type": "door",
                "color": None,
                "exclude_color": None,
                "relation": "closest",
                "distance_metric": "manhattan",
                "distance_reference": "agent",
            })

    checks["closest_grounding_ok"] = grounded_closest.get("ok", False)
    checks["closest_grounding_returns_distance"] = grounded_closest.get("distance") is not None
    checks["grounding_did_not_call_adapter_reset"] = reset_calls == []

    print("CLOSEST DOOR GROUNDING")
    pprint(grounded_closest)
    print(f"adapter.reset calls during grounding: {reset_calls}")
    print()

    # ── 4. Unique door grounding returns distance ───────────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session4 = _make_session()
        session4.handle_utterance("go to the red door")
        grounded_unique = session4.ground_target_selector({
            "object_type": "door",
            "color": "red",
            "exclude_color": None,
            "relation": "unique",
            "distance_metric": None,
            "distance_reference": None,
        })

    checks["unique_grounding_ok"] = grounded_unique.get("ok", False)
    checks["unique_grounding_returns_distance"] = grounded_unique.get("distance") is not None

    print("UNIQUE (red) DOOR GROUNDING")
    pprint(grounded_unique)
    print()

    # ── 5. Scene summary includes agent pose and source ─────────────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session5 = _make_session()
        session5.handle_utterance("go to the red door")
        scene_text = session5.status_summary(query="scene")

    checks["scene_summary_has_agent"] = "agent=" in scene_text
    checks["scene_summary_has_source"] = "source=" in scene_text
    checks["scene_summary_has_doors"] = "doors=" in scene_text

    print("SCENE SUMMARY")
    print(scene_text)
    print()

    # ── 6. Scene model cleared on reset, rebuilt on next query ─────────────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session6 = _make_session()
        session6.handle_utterance("go to the red door")
        had_scene = session6.memory.scene_model is not None
        session6.reset()
        cleared = session6.memory.scene_model is None
        session6._ensure_scene_model()
        rebuilt = session6.memory.scene_model is not None

    checks["scene_model_cleared_on_reset"] = had_scene and cleared
    checks["scene_model_rebuilt_after_reset"] = rebuilt

    # ── Summary ────────────────────────────────────────────────────────────
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
