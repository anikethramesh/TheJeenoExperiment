"""ORPI conformance: cadence declarations keep control paths non-deliberative.

v0 check: data-based — inspects cadence/mode fields on every manifest contract.
Full AST static check (tracing actual Spine call paths) is deferred to v1.
"""
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
    contracts = [contract.as_dict() for contract in package.resolve_orpi_manifest().primitives]
    bad_actuation = [
        contract["name"]
        for contract in contracts
        if contract["primitive_type"] == "actuation"
        and contract["cadence"] != "control"
    ]
    bad_sense = [
        contract["name"]
        for contract in contracts
        if contract["primitive_type"] == "sense"
        and contract["cadence"] != "perception"
    ]
    deliberative_control = [
        contract["name"]
        for contract in contracts
        if contract["cadence"] == "control"
        and contract["mode"] == "deliberative"
    ]
    details["bad_actuation_cadence"] = bad_actuation
    details["bad_sense_cadence"] = bad_sense
    details["deliberative_control_contracts"] = deliberative_control
    metrics["actuation_contracts_are_control_cadence"] = not bad_actuation
    metrics["sense_contracts_are_perception_cadence"] = not bad_sense
    metrics["control_path_has_no_deliberative_contract"] = not deliberative_control
    metrics["orpi_cadence_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="orpi_cadence_holds")


if __name__ == "__main__":
    raise SystemExit(main())
