"""Operational Repair Loop — Phase 9.1.

Consumes OperationalMismatch records from the station and attempts autonomous
repair interventions to restore the RequestPlan to 'executable' status.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .operator_station import OperatorStationSession
    from .mismatch import OperationalMismatch

REPAIR_ACTIONS = ("REFRESH_CLAIMS", "REGROUND", "CLARIFY", "SYNTHESIZE", "ABORT")


@dataclass
class RepairEvent:
    mismatch_type: str
    repair_action: str
    success: bool
    details: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "mismatch_type": self.mismatch_type,
            "repair_action": self.repair_action,
            "success": self.success,
            "details": self.details,
        }


class RepairLoop:
    """Executes deterministic repairs for operational mismatches."""

    def __init__(self, session: OperatorStationSession):
        self.session = session

    def attempt_repair(self, mismatches: list[OperationalMismatch]) -> list[RepairEvent]:
        events: list[RepairEvent] = []
        for mismatch in mismatches:
            if mismatch.recommended_repair == "refresh_claims":
                events.append(self._repair_refresh_claims(mismatch))
            elif mismatch.recommended_repair == "reground_target":
                events.append(self._repair_reground(mismatch))
            elif mismatch.recommended_repair == "clarify_operator":
                events.append(self._repair_clarify(mismatch))
            elif mismatch.recommended_repair == "recompile":
                events.append(self._repair_recompile(mismatch))
            else:
                events.append(RepairEvent(mismatch.mismatch_type, "ABORT", False, "No recommended repair."))
        return events

    def _repair_refresh_claims(self, mismatch: OperationalMismatch) -> RepairEvent:
        self.session.log(f"repair: refreshing claims for {mismatch.mismatch_type}")
        self.session.active_claims = None
        return RepairEvent(mismatch.mismatch_type, "REFRESH_CLAIMS", True, "Cleared active claims.")

    def _repair_reground(self, mismatch: OperationalMismatch) -> RepairEvent:
        self.session.log(f"repair: regrounding target for {mismatch.mismatch_type}")
        self.session.active_claims = None
        return RepairEvent(mismatch.mismatch_type, "REGROUND", True, "Cleared active claims for regrounding.")

    def _repair_clarify(self, mismatch: OperationalMismatch) -> RepairEvent:
        self.session.log(f"repair: clarification required for {mismatch.mismatch_type}")
        return RepairEvent(mismatch.mismatch_type, "CLARIFY", False, "Clarification requires operator interaction.")

    def _repair_recompile(self, mismatch: OperationalMismatch) -> RepairEvent:
        self.session.log(f"repair: synthesis/recompile required for {mismatch.mismatch_type}")
        return RepairEvent(mismatch.mismatch_type, "SYNTHESIZE", False, "Synthesis repair requires operator interaction.")
