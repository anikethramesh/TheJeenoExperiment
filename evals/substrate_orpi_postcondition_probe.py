"""ORPI conformance: actuation contracts name postcondition checkers."""
from __future__ import annotations

from harness import emit_result


def main() -> int:
    from jeenom.minigrid_runtime_package import build_minigrid_runtime_package

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}
    package = build_minigrid_runtime_package(
        env_id="MiniGrid-GoToDoor-8x8-v0",
        render_mode="none",
    )
    manifest = package.resolve_orpi_manifest()
    contracts = [contract.as_dict() for contract in manifest.primitives]
    names = {contract["name"] for contract in contracts}
    actuation = [
        contract
        for contract in contracts
        if contract["primitive_type"] == "actuation"
        and contract["safety_class"] != "query"
    ]
    missing = [
        contract["name"]
        for contract in actuation
        if not contract.get("postcondition_primitive")
    ]
    dangling = [
        contract["name"]
        for contract in actuation
        if contract.get("postcondition_primitive")
        and contract["postcondition_primitive"] not in names
    ]
    non_delta = [
        contract["name"]
        for contract in contracts
        if contract["primitive_type"] == "actuation"
        and not isinstance(contract.get("postconditions"), list)
    ]
    details["actuation_contracts"] = [contract["name"] for contract in actuation]
    details["missing_postcondition_primitive"] = missing
    details["dangling_postcondition_primitive"] = dangling
    metrics["actuation_contracts_exist"] = bool(actuation)
    metrics["actuation_postcondition_checkers_named"] = not missing
    metrics["postcondition_checkers_are_contracts"] = not dangling
    metrics["postconditions_are_contract_lists"] = not non_delta
    metrics["orpi_postcondition_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="orpi_postcondition_holds")


if __name__ == "__main__":
    raise SystemExit(main())
