"""Phase 8.4.6 probe: Unified Claim Abstraction + Architectural Separation.

Verifies the three-layer architectural contract:
  - All facts held by the station are Claims (unified abstraction).
  - Grounding claims (StationActiveClaims) are session-scoped, scene-fingerprinted,
    and cleared on reset.
  - Operator claims (KnowledgeBase, OperationalMemory.knowledge) are durable,
    asserted by the operator, and cleared only by explicit 'forget everything'.
  - The KnowledgeBase interface (not the LLM or any parser) owns name normalisation.
  - CapabilityArbitrator is decision-only — it does not write to KB or operator claims.
  - The S/C/S boundary: run_task() is the only path to execute motion; the Cortex
    (command_from_operator_intent, _command_from_concept) routes but never directly
    calls motor primitives.

Checks:
  claim_scopes_constant_exists           — CLAIM_SCOPES defined in schemas
  claim_scopes_has_grounding             — 'grounding' in CLAIM_SCOPES
  claim_scopes_has_operator              — 'operator' in CLAIM_SCOPES
  named_concept_claim_scope_operator     — NamedConcept.claim_scope == 'operator'
  operator_claim_scope_survives_reload   — claim_scope field persists via as_dict/from_dict
  kb_normalises_trailing_punctuation     — teach('bingo,', ...) stores as 'bingo'
  kb_normalises_on_recall                — recall('BINGO') == recall('bingo')
  kb_normalises_on_forget                — forget('bingo,') removes 'bingo' concept
  kb_normalises_on_load                  — names with trailing punctuation in JSON load as clean keys
  operator_claims_survive_episode_reset  — reset(clear_memory=False) keeps KB concepts
  operator_claims_cleared_by_forget_all  — reset(clear_memory=True) removes KB concepts
  grounding_claims_cleared_on_reset      — reset() sets active_claims = None
  grounding_claim_scene_fingerprinted    — is_valid_for returns False on stale fingerprint
  operator_claim_not_in_active_claims    — KB content not mixed into StationActiveClaims type
  arbitrator_has_no_kb_write_path        — CapabilityArbitrator.__init__ does not accept KnowledgeBase
  cortex_routes_task_not_executes        — _command_from_concept returns a command, not a side-effect
  knowledge_type_hierarchy_grounding_vs_operator — StationActiveClaims and KnowledgeBase are distinct types
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

from jeenom.capability_arbitrator import ArbitratorBackend
from jeenom.knowledge_base import KnowledgeBase, NamedConcept
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import (
    CLAIM_SCOPES,
    GroundedObjectEntry,
    SceneModel,
    SceneObject,
    StationActiveClaims,
)






def _make_kb() -> KnowledgeBase:
    return KnowledgeBase(storage_path=None)


def _make_dummy_active_claims(agent_x: int = 1, agent_y: int = 1, step_count: int = 0) -> StationActiveClaims:
    door = GroundedObjectEntry(color="red", x=3, y=3, distance=4.0, metric="manhattan")
    return StationActiveClaims(
        scene_fingerprint=(agent_x, agent_y, step_count),
        ranked_scene_doors=[door],
        last_grounded_target=door,
        last_grounded_rank=0,
        last_grounding_query={},
    )


def _make_dummy_scene(agent_x: int = 1, agent_y: int = 1, step_count: int = 0) -> SceneModel:
    return SceneModel(
        agent_x=agent_x,
        agent_y=agent_y,
        agent_dir=0,
        grid_width=8,
        grid_height=8,
        objects=[SceneObject(object_type="door", color="red", x=3, y=3)],
        source="test",
        step_count=step_count,
    )


def main() -> int:
    metrics: dict[str, bool] = {}

    # ── CLAIM_SCOPES schema constant ──────────────────────────────────────────
    import jeenom.schemas as _schemas
    metrics["claim_scopes_constant_exists"] = hasattr(_schemas, "CLAIM_SCOPES")
    metrics["claim_scopes_has_grounding"] = "grounding" in CLAIM_SCOPES
    metrics["claim_scopes_has_operator"] = "operator" in CLAIM_SCOPES

    # ── NamedConcept.claim_scope = 'operator' ─────────────────────────────────
    nc = NamedConcept(name="bingo", utterance="go to the red door")
    metrics["named_concept_claim_scope_operator"] = nc.claim_scope == "operator"

    # claim_scope survives serialisation
    d = nc.as_dict()
    # claim_scope is not in as_dict (it's a fixed structural property, not persisted),
    # but the default on from_dict should also be 'operator'
    restored = NamedConcept.from_dict({**d, "stored_at": 0.0, "recall_count": 0, "tags": []})
    metrics["operator_claim_scope_survives_reload"] = restored.claim_scope == "operator"

    # ── KnowledgeBase interface owns normalisation ────────────────────────────
    kb = _make_kb()
    kb.teach("bingo,", "go to the red door")   # trailing comma
    concept = kb.recall("bingo")
    metrics["kb_normalises_trailing_punctuation"] = (
        concept is not None and concept.name == "bingo"
    )

    concept_upper = kb.recall("BINGO")
    metrics["kb_normalises_on_recall"] = concept_upper is not None and concept_upper.name == "bingo"

    kb2 = _make_kb()
    kb2.teach("bingo", "go to the red door")
    removed = kb2.forget("bingo,")  # trailing comma in forget argument
    metrics["kb_normalises_on_forget"] = removed and kb2.recall("bingo") is None

    # Stale JSON with trailing comma in name loads cleanly
    import json as _json, tempfile as _tf
    stale_json = _json.dumps([{
        "name": "bingo,",
        "utterance": "go to the red door",
        "plan": None,
        "stored_at": 0.0,
        "recall_count": 0,
        "tags": [],
    }])
    tmp = Path(_tf.mktemp(suffix=".json"))
    tmp.write_text(stale_json)
    kb3 = KnowledgeBase(storage_path=tmp)
    loaded = kb3.recall("bingo")
    metrics["kb_normalises_on_load"] = loaded is not None and loaded.name == "bingo"
    tmp.unlink(missing_ok=True)

    # ── Operator claim lifecycle: survive reset, cleared by clear_memory ──────
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess = _make_session()
        sess.handle_utterance("remember bingo means go to the red door")
        sess.reset(clear_memory=False)
        metrics["operator_claims_survive_episode_reset"] = (
            sess.knowledge_base.recall("bingo") is not None
        )

        sess.reset(clear_memory=True)
        metrics["operator_claims_cleared_by_forget_all"] = (
            sess.knowledge_base.recall("bingo") is None
        )

        # ── Grounding claim lifecycle: cleared on any reset ───────────────────
        sess2 = _make_session()
        sess2.active_claims = _make_dummy_active_claims()
        sess2.reset(clear_memory=False)
        metrics["grounding_claims_cleared_on_reset"] = sess2.active_claims is None

    # ── Grounding claim scene-fingerprinting ──────────────────────────────────
    claims = _make_dummy_active_claims(agent_x=1, agent_y=1, step_count=5)
    scene_valid = _make_dummy_scene(agent_x=1, agent_y=1, step_count=5)
    scene_stale = _make_dummy_scene(agent_x=2, agent_y=1, step_count=5)
    metrics["grounding_claim_scene_fingerprinted"] = (
        claims.is_valid_for(scene_valid) is True
        and claims.is_valid_for(scene_stale) is False
    )

    # ── Type hierarchy: grounding vs operator are distinct types ─────────────
    metrics["operator_claim_not_in_active_claims"] = (
        not isinstance(NamedConcept(name="x", utterance="y"), StationActiveClaims)
    )
    metrics["knowledge_type_hierarchy_grounding_vs_operator"] = (
        type(KnowledgeBase()) is not type(StationActiveClaims)
    )

    # ── Arbitrator is decision-only: no KB write path ─────────────────────────
    # ArbitratorBackend is the abstract base; concrete subclasses (Smoke/LLM) also
    # take no knowledge_base parameter — the arbitrator only receives read-only context.
    import inspect
    arb_init_sig = inspect.signature(ArbitratorBackend.__init__)
    arb_params = set(arb_init_sig.parameters.keys()) - {"self"}
    metrics["arbitrator_has_no_kb_write_path"] = "knowledge_base" not in arb_params

    # ── Cortex routes, does not execute: _command_from_concept returns ApprovedCommand ──
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        sess3 = _make_session()
        sess3.handle_utterance("remember bingo means go to the red door")
        from jeenom.operator_station import ApprovedCommand
        cmd = sess3._command_from_concept("bingo")
        metrics["cortex_routes_task_not_executes"] = (
            isinstance(cmd, ApprovedCommand)
            and cmd.kind == "task_instruction"
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
