"""Phase 13A.2.4 — typed per-turn state.

The `OperatorStationSession` used to carry the trace of the turn-in-flight as ~19 loose
instance attributes (`last_*`, `current_environment_identity`, `active_steering_directive`).
They are one concept — the state produced while handling a single operator turn — so they
live in one typed object here, with guaranteed initialization (no read-before-set
`AttributeError`, the class of bug behind 13A.1's `active_steering_directive` fault) and a
single reset point. The session surfaces each field as a delegating property, preserving the
public `session.last_*` read API the eval suite depends on.

`active_claims` and the `pending_*` continuation state are deliberately NOT here: they have
their own property-backed homes (representation store / pending-state machine).
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mismatch import OperationalMismatch
    from .plan_reuse import ReuseVerdict
    from .repair_loop import RepairEvent
    from .schemas import (
        ApprovedCommand,
        ArbitrationTrace,
        CommandResult,
        CorticalEnvelope,
        EnvironmentIdentity,
        ExecutionTicket,
        MemoryWriteTicket,
        MissionExecutionPlan,
        OperatorIntent,
        RawMotorTicket,
        ReadinessGraph,
        RequestPlan,
        SenseTicket,
        SteeringDirective,
    )


@dataclass
class TurnState:
    """The trace of one operator turn. All fields default to empty so a freshly
    constructed session can be read before its first turn without error."""

    last_result: "dict[str, Any] | None" = None
    active_steering_directive: "SteeringDirective | None" = None
    last_mission_execution_plan: "MissionExecutionPlan | None" = None
    last_execution_ticket: "ExecutionTicket | None" = None
    last_memory_write_ticket: "MemoryWriteTicket | None" = None
    last_raw_motor_ticket: "RawMotorTicket | None" = None
    last_sense_ticket: "SenseTicket | None" = None
    last_operator_intent: "OperatorIntent | None" = None
    last_cortical_envelope: "CorticalEnvelope | None" = None
    last_approved_command: "ApprovedCommand | None" = None
    last_command_result: "CommandResult | None" = None
    current_environment_identity: "EnvironmentIdentity | None" = None
    last_environment_invalidation_reason: "str | None" = None
    last_request_plan: "RequestPlan | None" = None
    last_readiness_graph: "ReadinessGraph | None" = None
    last_plan_reuse_verdict: "ReuseVerdict | None" = None
    last_arbitration_trace: "ArbitrationTrace | None" = None
    last_operational_mismatches: "list[OperationalMismatch]" = field(default_factory=list)
    last_repair_events: "list[RepairEvent]" = field(default_factory=list)


#: The per-turn field names, surfaced by the session as delegating properties.
TURN_STATE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(TurnState))
