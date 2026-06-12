"""Op 7 probe: hardware schema fields on correct owners.

Verifies that the hardware requirement fields are placed on the correct schema
owners as specified in the plan.

Metrics:
  op7_claim_record_valid_until         — ClaimRecord has valid_until: float | None field
  op7_primitive_spec_postcondition     — PrimitiveSpec has postcondition_primitive: str | None field
  op7_mission_contract_risk_tier       — MissionContract has risk_tier: str = "low" field
  op7_mission_contract_cadence         — MissionContract has cadence: str | None field
  op7_failure_outcome_exists           — FailureOutcome dataclass exists in schemas with category/detail/blocking_claim_handle
  op7_command_result_failure_outcome   — CommandResult accepts and stores failure_outcome: FailureOutcome | None
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    metrics: dict[str, bool] = {}

    from jeenom import schemas

    # 1. ClaimRecord.valid_until: float | None = None
    cr_fields = {f.name: f for f in dataclasses.fields(schemas.ClaimRecord)}
    metrics["op7_claim_record_valid_until"] = (
        "valid_until" in cr_fields
        and cr_fields["valid_until"].default is None
    )

    # 2. PrimitiveSpec.postcondition_primitive: str | None = None
    ps_fields = {f.name: f for f in dataclasses.fields(schemas.PrimitiveSpec)}
    metrics["op7_primitive_spec_postcondition"] = (
        "postcondition_primitive" in ps_fields
        and ps_fields["postcondition_primitive"].default is None
    )

    # 3. MissionContract.risk_tier: str = "low"
    mc_fields = {f.name: f for f in dataclasses.fields(schemas.MissionContract)}
    metrics["op7_mission_contract_risk_tier"] = (
        "risk_tier" in mc_fields
        and mc_fields["risk_tier"].default == "low"
    )

    # 4. MissionContract.cadence: str | None = None
    metrics["op7_mission_contract_cadence"] = (
        "cadence" in mc_fields
        and mc_fields["cadence"].default is None
    )

    # 5. FailureOutcome dataclass with required fields
    try:
        fo_cls = schemas.FailureOutcome
        fo_fields = {f.name: f for f in dataclasses.fields(fo_cls)}
        metrics["op7_failure_outcome_exists"] = (
            "category" in fo_fields
            and "detail" in fo_fields
            and "blocking_claim_handle" in fo_fields
            and fo_fields["detail"].default is None
            and fo_fields["blocking_claim_handle"].default is None
        )
    except AttributeError:
        metrics["op7_failure_outcome_exists"] = False

    # 6. CommandResult accepts failure_outcome and stores it
    try:
        fo = schemas.FailureOutcome(category="timeout")
        cr = schemas.CommandResult(
            "test",
            failure_outcome=fo,
        )
        metrics["op7_command_result_failure_outcome"] = (
            hasattr(cr, "failure_outcome")
            and cr.failure_outcome is fo
        )
        # Also verify default is None
        cr2 = schemas.CommandResult("test2")
        metrics["op7_command_result_failure_outcome"] = (
            metrics["op7_command_result_failure_outcome"]
            and cr2.failure_outcome is None
        )
    except (AttributeError, TypeError):
        metrics["op7_command_result_failure_outcome"] = False

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
