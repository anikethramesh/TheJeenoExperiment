"""ORPI regression: legacy primitive layers bridge to ORPI taxonomy."""
from __future__ import annotations

from harness import emit_result


def main() -> int:
    from jeenom.orpi import OrpiContract
    from jeenom.schemas import PrimitiveSpec, orpi_primitive_type_for

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}
    expected = {
        "sensing": "sense",
        "action": "actuation",
        "task": "meta",
        "grounding": "meta",
        "claims": "meta",
        "sense": "sense",
        "actuation": "actuation",
        "meta": "meta",
    }
    observed = {key: orpi_primitive_type_for(key) for key in expected}
    details["observed_mapping"] = observed
    legacy = PrimitiveSpec(
        name="grounding.example",
        primitive_type="grounding",
        layer="grounding",
        description="Legacy grounding primitive.",
    )
    actuation = PrimitiveSpec(
        name="action.move_forward",
        primitive_type="action",
        layer="action",
        description="Legacy action primitive.",
        safety_class="actuation",
        authority_level="operator",
        postcondition_primitive="sensing.parse_grid_objects",
    )
    metrics["legacy_mapping_matches_orpi_spec"] = observed == expected
    metrics["legacy_schema_defaults_orpi_metadata"] = (
        legacy.mode == "deterministic"
        and legacy.cadence == "deliberation"
        and legacy.invariant_level == "intent"
    )
    metrics["contract_view_remaps_legacy_type"] = (
        OrpiContract.from_primitive_spec(legacy).as_dict()["primitive_type"] == "meta"
        and OrpiContract.from_primitive_spec(actuation).as_dict()["primitive_type"]
        == "actuation"
    )
    metrics["legacy_layer_is_preserved"] = OrpiContract.from_primitive_spec(legacy).as_dict()[
        "layer"
    ] == "grounding"
    metrics["orpi_primitive_type_migration_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="orpi_primitive_type_migration_holds")


if __name__ == "__main__":
    raise SystemExit(main())
