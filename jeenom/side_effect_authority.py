from __future__ import annotations

from .schemas import (
    ExecutionTicket,
    MemoryUpdate,
    MemoryWriteTicket,
    RawMotorTicket,
    ReadinessGraph,
    RequestPlan,
)


class SideEffectAuthority:
    """Mint typed authority tokens for world and memory side effects."""

    def __init__(self, *, source_name: str = "OperatorStationSession") -> None:
        self.source_name = source_name

    def issue_execution_ticket(
        self,
        *,
        instruction: str,
        task_type: str,
        params: dict,
        request_plan: RequestPlan,
        readiness_graph: ReadinessGraph,
        source: str | None = None,
        mission_id: str | None = None,
        parent_request_id: str | None = None,
        provenance: dict | None = None,
    ) -> ExecutionTicket:
        return ExecutionTicket(
            request_id=request_plan.request_id,
            instruction=instruction,
            task_type=task_type,
            params=dict(params),
            request_plan=request_plan,
            readiness_graph=readiness_graph,
            source=source or self.source_name,
            mission_id=mission_id,
            parent_request_id=parent_request_id,
            provenance=dict(provenance or {}),
        )

    def issue_raw_motor_ticket(
        self,
        *,
        action_name: str,
        repeat_count: int,
        request_plan: RequestPlan,
        readiness_graph: ReadinessGraph,
        source: str | None = None,
    ) -> RawMotorTicket:
        return RawMotorTicket(
            request_id=request_plan.request_id,
            action_name=action_name,
            repeat_count=repeat_count,
            request_plan=request_plan,
            readiness_graph=readiness_graph,
            source=source or self.source_name,
        )

    def issue_memory_write_ticket(
        self,
        *,
        writes: list[MemoryUpdate],
        request_plan: RequestPlan,
        readiness_graph: ReadinessGraph,
        source: str | None = None,
    ) -> MemoryWriteTicket:
        return MemoryWriteTicket(
            request_id=request_plan.request_id,
            writes=list(writes),
            request_plan=request_plan,
            readiness_graph=readiness_graph,
            source=source or self.source_name,
        )
