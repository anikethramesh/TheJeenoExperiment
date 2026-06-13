"""ORPI 12C: durable named concepts carry falsifiable transfer scope."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from harness import emit_result


def _effect_manifest():
    from jeenom.orpi import OrpiContract, OrpiManifest
    from jeenom.schemas import PrimitiveSpec

    effect_spec = PrimitiveSpec(
        name="meta.effect_vocab",
        primitive_type="meta",
        layer="claims",
        description="Effect vocabulary entry.",
        postconditions=["object_state_delta"],
    )
    embodied_spec = PrimitiveSpec(
        name="action.robot_dock",
        primitive_type="action",
        layer="action",
        description="Morphology-specific docking action.",
        postconditions=["docked"],
        substrate_fingerprint="robot:v1",
    )
    return OrpiManifest(
        substrate_id="probe",
        substrate_fingerprint="probe:v1",
        object_vocabulary=["dock"],
        primitives=[
            OrpiContract.from_primitive_spec(effect_spec),
            OrpiContract.from_primitive_spec(embodied_spec),
        ],
    )


def main() -> int:
    from jeenom.knowledge_base import KnowledgeBase, NamedConcept, derive_scope
    from jeenom.orpi import OrpiProcedure
    from jeenom.schemas import RequestPlan, RequestPlanStep

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}
    manifest = _effect_manifest()

    kb = KnowledgeBase(storage_path=None)
    taught = kb.teach("bingo", "go to the red door")
    metrics["operator_taught_concepts_default_site_scope"] = taught.scope == "site"

    legacy_path = Path(tempfile.mkdtemp()) / "knowledge_base.json"
    legacy_path.write_text(
        json.dumps(
            [
                {
                    "name": "legacy",
                    "utterance": "go to the red door",
                    "stored_at": 1.0,
                    "recall_count": 0,
                    "tags": [],
                }
            ]
        )
    )
    migrated = KnowledgeBase(storage_path=legacy_path).recall("legacy")
    metrics["persisted_legacy_records_load_as_site"] = (
        migrated is not None and migrated.scope == "site"
    )

    universal_plan = RequestPlan(
        request_id="request:universal-scope",
        original_utterance="express as state delta",
        objective_type="procedure",
        objective_summary="universal effect procedure",
        steps=[
            RequestPlanStep(
                step_id="delta",
                layer="claims",
                operation="derive",
                required_handle="object_state_delta",
            )
        ],
        expected_response="answer_query",
    )
    universal = NamedConcept(
        name="universal",
        utterance="object state delta recipe",
        plan=universal_plan,
    )
    metrics["effect_only_recipe_derives_universal_scope"] = (
        derive_scope(universal, manifest) == "universal"
    )

    embodied_plan = RequestPlan(
        request_id="request:embodiment-scope",
        original_utterance="dock",
        objective_type="procedure",
        objective_summary="robot docking procedure",
        steps=[
            RequestPlanStep(
                step_id="dock",
                layer="action",
                operation="execute",
                required_handle="action.robot_dock",
            )
        ],
        expected_response="execute_task",
    )
    embodied = NamedConcept(
        name="dock",
        utterance="dock using current robot",
        plan=embodied_plan,
    )
    metrics["substrate_fingerprinted_step_derives_embodiment_scope"] = (
        derive_scope(embodied, manifest) == "embodiment"
    )

    oem_procedure = OrpiProcedure(
        name="procedure.oem.dock",
        steps=[{"name": "action.robot_dock", "effect": ["docked"]}],
        declared_postconditions=["docked"],
        provenance="oem",
    )
    metrics["oem_bundled_procedure_derives_embodiment_scope"] = (
        derive_scope(oem_procedure, manifest) == "embodiment"
    )

    try:
        kb.teach("flash", "this happened now", scope="episodic")
    except ValueError as exc:
        details["episodic_rejection"] = str(exc)
        metrics["episodic_scope_rejected_by_kb"] = True
    else:
        metrics["episodic_scope_rejected_by_kb"] = False

    metrics["knowledge_scope_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="knowledge_scope_holds")


if __name__ == "__main__":
    raise SystemExit(main())
