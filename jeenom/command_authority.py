from __future__ import annotations

import uuid
from typing import Any, Iterable

from .schemas import (
    ApprovedCommand,
    CommandResult,
    CorticalEnvelope,
    ExecutionTicket,
    MemoryWriteTicket,
    OperatorIntent,
    RawMotorTicket,
    ReadinessGraph,
    RequestPlan,
)


AuthorityTicket = ExecutionTicket | MemoryWriteTicket | RawMotorTicket


class CommandAuthority:
    """Build typed control-plane traces for user-visible command results."""

    def __init__(self, *, station_name: str = "OperatorStationSession") -> None:
        self.station_name = station_name

    def ticket_for_trace(
        self,
        request_id: str | None,
        tickets: Iterable[AuthorityTicket | None],
    ) -> AuthorityTicket | None:
        if request_id is None:
            return None
        for ticket in tickets:
            if ticket is not None and ticket.request_id == request_id:
                return ticket
        return None

    def trace_command_type(
        self,
        message: str,
        *,
        graph: ReadinessGraph | None,
        plan: RequestPlan | None,
    ) -> str:
        if graph is not None:
            return graph.next_action
        if plan is not None:
            return plan.expected_response
        first_line = message.splitlines()[0] if message else ""
        return first_line.split(" ", 1)[0].lower() if first_line else "response"

    def _make_envelope(
        self,
        prefix: str,
        utterance: str,
        intent: OperatorIntent | None,
        plan: RequestPlan | None,
        graph: ReadinessGraph | None,
        compiler_name: str,
        pending_context: dict[str, Any],
    ) -> CorticalEnvelope:
        return CorticalEnvelope(
            envelope_id=f"{prefix}:{uuid.uuid4().hex}",
            utterance=utterance,
            intent=intent,
            request_plan=plan,
            readiness_graph=graph,
            provenance={"station": self.station_name, "compiler": compiler_name},
            pending_context=pending_context,
        )

    def record_result(
        self,
        utterance: str,
        message: str,
        *,
        intent: OperatorIntent | None,
        plan: RequestPlan | None,
        graph: ReadinessGraph | None,
        tickets: Iterable[AuthorityTicket | None],
        compiler_name: str,
        pending_context: dict[str, Any] | None = None,
        last_result: dict[str, Any] | None = None,
    ) -> CommandResult:
        envelope = self._make_envelope(
            "envelope", utterance, intent, plan, graph, compiler_name, dict(pending_context or {})
        )
        request_id = (
            graph.request_id
            if graph is not None
            else plan.request_id
            if plan is not None
            else envelope.envelope_id
        )
        command = ApprovedCommand(
            command_type=self.trace_command_type(message, graph=graph, plan=plan),
            request_id=request_id,
            source="station",
            utterance=utterance,
            payload={"first_line": message.splitlines()[0] if message else ""},
            request_plan=plan,
            readiness_graph=graph,
        )
        ticket = self.ticket_for_trace(request_id, tickets)
        result_payload: dict[str, Any] = {"message": message}
        if ticket is not None and last_result is not None:
            result_payload["last_result"] = dict(last_result)
        command_result = CommandResult(
            message,
            envelope=envelope,
            command=command,
            ticket=ticket,
            result=result_payload,
        )
        from .orpi import LabelledEpisode

        command_result.labelled_episode = LabelledEpisode.from_command_result(command_result)
        return command_result

    def pending_clarification_trace(
        self,
        utterance: str,
        clarification_type: str,
        *,
        intent: OperatorIntent | None,
        plan: RequestPlan | None,
        graph: ReadinessGraph | None,
        compiler_name: str,
    ) -> dict[str, Any]:
        envelope = self._make_envelope(
            "pending", utterance, intent, plan, graph, compiler_name,
            {"clarification": clarification_type},
        )
        return {
            "request_plan": plan,
            "readiness_graph": graph,
            "pending_envelope": envelope,
        }
