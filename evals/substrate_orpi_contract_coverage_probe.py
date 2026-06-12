"""ORPI conformance: every MiniGrid capability has a contract view."""
from __future__ import annotations

from harness import emit_result


def main() -> int:
    from jeenom.minigrid_runtime_package import build_minigrid_runtime_package
    from jeenom.schemas import ORPI_PRIMITIVE_SPEC_TYPES

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}
    package = build_minigrid_runtime_package(
        env_id="MiniGrid-GoToDoor-8x8-v0",
        render_mode="none",
    )
    registry = package.resolve_capability_registry()
    manifest = package.resolve_orpi_manifest()
    registry_names = {spec.name for spec in registry.manifest.primitives}
    contract_payloads = [contract.as_dict() for contract in manifest.primitives]
    contract_names = {payload["name"] for payload in contract_payloads}
    missing = sorted(registry_names - contract_names)
    unsupported_types = sorted(
        {
            str(payload["primitive_type"])
            for payload in contract_payloads
            if payload["primitive_type"] not in ORPI_PRIMITIVE_SPEC_TYPES
        }
    )
    details["missing_contracts"] = missing
    details["unsupported_orpi_types"] = unsupported_types
    metrics["all_registry_capabilities_have_orpi_contracts"] = not missing
    metrics["contracts_use_orpi_taxonomy"] = not unsupported_types
    metrics["contract_coverage_non_empty"] = bool(contract_payloads)
    metrics["orpi_contract_coverage_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="orpi_contract_coverage_holds")


if __name__ == "__main__":
    raise SystemExit(main())
