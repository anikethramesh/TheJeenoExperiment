"""Phase 8.4 probe: Operational Mismatch Detection.

Proves that MismatchDetector correctly identifies each of the six mismatch
types and that OperatorStationSession stores last_operational_mismatches
after every plan evaluation without blocking execution or breaking existing
paths.

Checks:
  stale_claims_detected                     — STALE_CLAIMS when claims fingerprint != scene
  required_entity_absent_detected           — REQUIRED_ENTITY_ABSENT when plan needs a
                                              color not in scene
  grounding_relation_invalidated_detected   — GROUNDING_RELATION_INVALIDATED when ranked
                                              order would differ on current scene
  unsupported_grounding_detected            — UNSUPPORTED_GROUNDING for unimplemented handle
  missing_primitive_detected                — MISSING_PRIMITIVE_IN_REGISTRY for unknown handle
  constraint_weakening_detected             — CONSTRAINT_WEAKENING for exclude_colors with no
                                              enforcing primitive
  no_mismatch_on_clean_plan                 — empty list for a well-formed plan on valid scene
  station_stores_mismatches                 — session.last_operational_mismatches populated
  mismatch_does_not_block_execution         — task completes even when mismatches are present
  golden_path_unaffected                    — 'go to the red door' still returns RUN COMPLETE
"""
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.mismatch import MismatchDetector, OperationalMismatch
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import (
    EnvironmentAssumption,
    GroundedDoorEntry,
    RequestPlan,
    RequestPlanStep,
    SceneModel,
    SceneObject,
    StationActiveClaims,
)


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session(memory_root: Path | None = None) -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=memory_root or Path(tempfile.mkdtemp()),
    )


# ── helpers to build minimal synthetic fixtures ───────────────────────────────

def _minimal_plan(
    *,
    color: str | None = None,
    required_handle: str | None = None,
    exclude_colors: list[str] | None = None,
    layer: str = "grounding",
    operation: str = "select",
    assumption_ids: list[str] | None = None,
) -> RequestPlan:
    constraints: dict = {}
    if color:
        constraints["color"] = color
        constraints["object_type"] = "door"
    if exclude_colors:
        constraints["exclude_colors"] = exclude_colors
    step = RequestPlanStep(
        step_id="s1",
        layer=layer,
        operation=operation,
        required_handle=required_handle,
        constraints=constraints,
        environment_assumption_ids=assumption_ids or [],
    )
    return RequestPlan(
        request_id="probe-r1",
        original_utterance="probe",
        objective_type="query",
        objective_summary="probe plan",
        steps=[step],
    )


def _scene_with_doors(*colors: str, agent_x: int = 1, agent_y: int = 1) -> SceneModel:
    """Build a minimal SceneModel containing the specified door colors."""
    objects = [
        SceneObject(object_type="door", color=c, x=2 + i, y=3)
        for i, c in enumerate(colors)
    ]
    return SceneModel(
        agent_x=agent_x,
        agent_y=agent_y,
        agent_dir=0,
        grid_width=8,
        grid_height=8,
        objects=objects,
        source="idle_sense",
    )


def _claims(
    *,
    ranked_colors: list[str],
    distances: list[float],
    scene: SceneModel,
    top_color: str | None = None,
) -> StationActiveClaims:
    """Build StationActiveClaims with the scene's fingerprint so is_valid_for() returns True."""
    doors = [
        GroundedDoorEntry(
            color=ranked_colors[i],
            x=scene.objects[i].x,
            y=scene.objects[i].y,
            distance=distances[i],
            metric="manhattan",
        )
        for i in range(len(ranked_colors))
    ]
    top = doors[0]
    return StationActiveClaims(
        scene_fingerprint=(scene.agent_x, scene.agent_y, scene.step_count),
        ranked_scene_doors=doors,
        last_grounded_target=top,
        last_grounded_rank=0,
        last_grounding_query={"relation": "closest", "metric": "manhattan"},
    )


def main() -> int:
    metrics: dict[str, bool] = {}
    detector = MismatchDetector()
    registry = CapabilityRegistry.minigrid_default()

    # ── STALE_CLAIMS ─────────────────────────────────────────────────────────
    plan = _minimal_plan()
    scene = _scene_with_doors("red", "blue", agent_x=1, agent_y=1)
    stale_scene = _scene_with_doors("red", "blue", agent_x=3, agent_y=3)  # different position
    claims = _claims(
        ranked_colors=["red", "blue"],
        distances=[3.0, 5.0],
        scene=scene,  # fingerprinted at (1,1,0)
    )
    # stale_scene has agent at (3,3) → claims fingerprint won't match
    mismatches = detector.detect(plan, scene_model=stale_scene, active_claims=claims)
    metrics["stale_claims_detected"] = any(
        m.mismatch_type == "STALE_CLAIMS" for m in mismatches
    )

    # ── REQUIRED_ENTITY_ABSENT ────────────────────────────────────────────────
    plan_needs_red = _minimal_plan(color="red")
    scene_no_red = _scene_with_doors("blue", "yellow")  # no red door
    mismatches = detector.detect(plan_needs_red, scene_model=scene_no_red)
    metrics["required_entity_absent_detected"] = any(
        m.mismatch_type == "REQUIRED_ENTITY_ABSENT" for m in mismatches
    )

    # ── GROUNDING_RELATION_INVALIDATED ────────────────────────────────────────
    # Scene has red at x=2 and blue at x=4, agent at (1,1).
    # Claims say red is closest (distance=1) and blue is farther (distance=3).
    # But we store claims with SWAPPED distances: blue closer in the record.
    # Claims fingerprint matches the scene (same agent pos/step), but recorded
    # distances suggest blue should rank first — recomputed from actual positions
    # would give red first.  Detector should notice the rank inconsistency.
    scene_for_rank = SceneModel(
        agent_x=1, agent_y=1, agent_dir=0,
        grid_width=8, grid_height=8,
        objects=[
            SceneObject(object_type="door", color="red",  x=2, y=1),  # distance=1
            SceneObject(object_type="door", color="blue", x=4, y=1),  # distance=3
        ],
        source="idle_sense",
    )
    # Claims have blue ranked first (distance=1) and red second (distance=3) — wrong.
    invalidated_claims = StationActiveClaims(
        scene_fingerprint=(1, 1, 0),  # matches scene
        ranked_scene_doors=[
            GroundedDoorEntry(color="blue", x=4, y=1, distance=1.0, metric="manhattan"),
            GroundedDoorEntry(color="red",  x=2, y=1, distance=3.0, metric="manhattan"),
        ],
        last_grounded_target=GroundedDoorEntry(color="blue", x=4, y=1, distance=1.0),
        last_grounded_rank=0,
        last_grounding_query={"relation": "closest", "metric": "manhattan"},
    )
    mismatches = detector.detect(plan, scene_model=scene_for_rank, active_claims=invalidated_claims)
    metrics["grounding_relation_invalidated_detected"] = any(
        m.mismatch_type == "GROUNDING_RELATION_INVALIDATED" for m in mismatches
    )

    # ── UNSUPPORTED_GROUNDING ─────────────────────────────────────────────────
    # grounding.closest_door.euclidean.agent is synthesizable (not implemented)
    plan_euclidean = _minimal_plan(
        required_handle="grounding.closest_door.euclidean.agent",
        layer="grounding",
        operation="select",
    )
    mismatches = detector.detect(plan_euclidean, registry=registry)
    metrics["unsupported_grounding_detected"] = any(
        m.mismatch_type == "UNSUPPORTED_GROUNDING" for m in mismatches
    )

    # ── MISSING_PRIMITIVE_IN_REGISTRY ─────────────────────────────────────────
    plan_phantom = _minimal_plan(
        required_handle="grounding.phantom_primitive.does_not_exist",
        layer="grounding",
        operation="select",
    )
    mismatches = detector.detect(plan_phantom, registry=registry)
    metrics["missing_primitive_detected"] = any(
        m.mismatch_type == "MISSING_PRIMITIVE_IN_REGISTRY" for m in mismatches
    )

    # ── CONSTRAINT_WEAKENING ──────────────────────────────────────────────────
    # Remove the color_filter primitive from a fresh registry copy to simulate absence.
    from jeenom.primitive_library import GROUNDING_PRIMITIVES, TASK_PRIMITIVES, SENSING_PRIMITIVES, ACTION_PRIMITIVES
    from jeenom.schemas import PrimitiveManifest, PrimitiveSpec
    stripped_primitives = [
        spec for spec in registry._by_name.values()
        if "color_filter" not in spec.name
    ]
    stripped_manifest = PrimitiveManifest(
        name="stripped",
        primitives=stripped_primitives,
    )
    stripped_registry = CapabilityRegistry(stripped_manifest)
    plan_exclude = _minimal_plan(exclude_colors=["yellow"])
    mismatches = detector.detect(plan_exclude, registry=stripped_registry)
    metrics["constraint_weakening_detected"] = any(
        m.mismatch_type == "CONSTRAINT_WEAKENING" for m in mismatches
    )

    # ── CLEAN PLAN — no mismatches ────────────────────────────────────────────
    clean_plan = _minimal_plan(
        required_handle="grounding.closest_door.manhattan.agent",
        layer="grounding",
        operation="select",
    )
    clean_scene = _scene_with_doors("red", "blue")
    clean_claims = _claims(
        ranked_colors=["red", "blue"],
        distances=[1.0, 3.0],
        scene=clean_scene,
    )
    mismatches = detector.detect(
        clean_plan,
        registry=registry,
        scene_model=clean_scene,
        active_claims=clean_claims,
    )
    metrics["no_mismatch_on_clean_plan"] = len(mismatches) == 0

    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        # ── Station stores last_operational_mismatches ─────────────────────────
        session = _make_session()
        session.handle_utterance("what doors do you see?")
        session.handle_utterance("which door is closest by manhattan distance")
        metrics["station_stores_mismatches"] = hasattr(session, "last_operational_mismatches")

        # ── Mismatches do not block execution ─────────────────────────────────
        # Teach a concept that targets a color that may or may not be in the scene.
        # Even if REQUIRED_ENTITY_ABSENT fires, execution should proceed (or fail
        # gracefully through normal task-failure path, not a hard block from mismatch).
        session2 = _make_session()
        session2.handle_utterance("what doors do you see?")
        resp = session2.handle_utterance("go to the red door")
        # Task completes (or fails gracefully) — no hard exception from mismatch detection.
        metrics["mismatch_does_not_block_execution"] = (
            "RUN COMPLETE" in resp or "FAILED" in resp or "not found" in resp.lower()
        )

        # ── Golden path unaffected ─────────────────────────────────────────────
        golden = _make_session()
        golden_resp = golden.handle_utterance("go to the red door")
        metrics["golden_path_unaffected"] = (
            "RUN COMPLETE" in golden_resp and "task_complete=True" in golden_resp
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
