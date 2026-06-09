from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .schemas import (
    ApprovedCommand,
    CorticalEnvelope,
    MissionExecutionPlan,
    PrimitiveDefinitionRequest,
    ReadinessGraph,
    RequestPlan,
)


@dataclass
class PendingClarification:
    clarification_type: str
    original_utterance: str
    resume_kind: str
    partial_selector: dict[str, Any]
    missing_field: str
    supported_values: list[str]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    request_plan: RequestPlan | None = None
    readiness_graph: ReadinessGraph | None = None
    pending_envelope: CorticalEnvelope | None = None


@dataclass
class PendingSynthesisProposal:
    handle: str
    original_utterance: str
    intent: Any
    cap_match: Any
    similar_handles: list[str]
    proposed_description: str | None = None
    proposed_condition: dict[str, Any] | None = None


@dataclass
class PendingPrimitiveDefinition:
    request: PrimitiveDefinitionRequest
    request_plan: RequestPlan
    readiness_graph: ReadinessGraph
    mission_plan: MissionExecutionPlan | None = None


@dataclass
class TurnOrchestrator:
    """Top-level operator turn router for an OperatorStationSession facade.

    Owns the multi-turn pending state (clarification, synthesis, primitive definition)
    so that the station facade remains stateless with respect to turn flow.
    """

    classify_utterance: Callable[[str], ApprovedCommand]
    normalize_utterance: Callable[[str], str]
    looks_like_bare_label: Callable[[str], bool]
    pending_clarification: PendingClarification | None = None
    pending_synthesis_proposal: PendingSynthesisProposal | None = None
    pending_primitive_definition: PendingPrimitiveDefinition | None = None

    def handle_utterance_text(self, station: Any, utterance: str) -> str:
        station.last_repair_events = []
        station.last_operational_mismatches = []

        command = self.classify_utterance(utterance)
        if getattr(station, "pending_primitive_definition", None) is not None:
            return station.handle_pending_primitive_definition(utterance, command)
        if station.pending_synthesis_proposal is not None:
            return station.handle_pending_synthesis_proposal(utterance, command)
        if station.pending_clarification is not None:
            pending_response = self.handle_pending_clarification(station, utterance, command)
            if pending_response is not None:
                return pending_response

        if command.kind == "unresolved":
            command = station._command_from_active_claim_text(utterance) or command
            if command.kind == "unresolved":
                grounding_followup = station._command_from_grounding_followup(utterance)
                if grounding_followup is not None:
                    command = grounding_followup
            if command.kind == "unresolved":
                concept_command = station._command_from_concept(utterance)
                if concept_command is not None:
                    command = concept_command
            if command.kind == "unresolved" and self.looks_like_bare_label(
                self.normalize_utterance(utterance.strip())
            ):
                label = self.normalize_utterance(utterance.strip())
                return (
                    f"I don't recognise '{label}' as a command or a known concept.\n"
                    f"To teach it as a shorthand, say:\n"
                    f"  remember {label} means <full instruction>\n"
                    f"Example: remember {label} means go to the red door"
                )
            if command.kind == "unresolved":
                station.log("deterministic fast path unresolved; compiling operator intent")
                command = station.command_from_llm_intent(utterance)

        return self.execute_command(station, command)

    def handle_pending_clarification(
        self,
        station: Any,
        utterance: str,
        command: ApprovedCommand,
    ) -> str | None:
        normalized = self.normalize_utterance(utterance)
        if command.kind == "quit":
            station.pending_clarification = None
            return "QUIT"
        if command.kind == "cancel":
            station.pending_clarification = None
            return "CANCELLED: pending clarification cleared"
        if command.kind == "reset":
            return station.reset(clear_memory=bool(command.payload.get("clear_memory")))
        if command.kind == "cache_query":
            return station.cache_summary()
        if command.kind == "status_query":
            return station.status_summary(query=command.payload.get("query", "status"))

        pending = station.pending_clarification
        if pending is not None and pending.clarification_type == "arbitrator_offer":
            if station._is_acceptance(normalized):
                station.pending_clarification = None
                return station.resume_arbitration_offer(pending.resume_kind)

        if pending is not None and pending.clarification_type == "semantic_query_missing_field":
            if station.domain_helper.is_metric_answer(normalized, "manhattan"):
                station.pending_clarification = None
                clarified = f"{pending.original_utterance} using manhattan distance"
                station.log("resuming semantic query plan with distance_metric=manhattan")
                resumed = station.command_from_llm_intent(clarified)
                if resumed.kind == "clarification":
                    return resumed.payload["message"]
                return self.execute_command(station, resumed)
            if station.domain_helper.is_metric_answer(normalized, "euclidean"):
                station.pending_clarification = None
                return "I cannot use Euclidean distance yet. Supported: manhattan."

        if station.domain_helper.is_metric_answer(normalized, "manhattan"):
            return station.resume_pending_clarification("manhattan")
        if station.domain_helper.is_metric_answer(normalized, "euclidean"):
            station.pending_clarification = None
            return "I cannot use Euclidean distance yet. Supported: manhattan."

        color_answer = station.domain_helper.color_answer(normalized)
        if color_answer is not None:
            candidate_response = station.resume_candidate_clarification(color_answer)
            if candidate_response is not None:
                return candidate_response

        if command.kind == "unresolved":
            station.log("deterministic fast path unresolved; compiling operator intent")
            command = station.command_from_llm_intent(utterance)
        if command.kind in {
            "task_instruction",
            "task_selector",
            "knowledge_update",
            "ground_target_query",
            "unsupported",
            "ambiguous",
        }:
            station.pending_clarification = None
            station.log("new operator intent cancelled pending clarification")
            return self.execute_command(station, command)
        if command.kind == "synthesis_proposal":
            station.pending_clarification = None
            return command.payload["message"]
        if command.kind == "clarification":
            return command.payload["message"]
        if command.kind in {"missing_skills", "synthesizable"}:
            station.pending_clarification = None
            return command.payload.get("message", "I cannot fulfil that request.")

        pending = station.pending_clarification
        if pending is None:
            return None
        return station.clarification_prompt(
            pending.missing_field,
            pending.supported_values,
            candidates=pending.candidates,
        )

    def execute_command(self, station: Any, command: ApprovedCommand) -> str:
        station.log(f"classified utterance as {command.kind}")
        if command.kind == "quit":
            station.pending_clarification = None
            return "QUIT"
        if command.kind == "cancel":
            station.pending_clarification = None
            return "CANCELLED: pending clarification cleared"
        if command.kind == "reset":
            return station.reset(clear_memory=bool(command.payload.get("clear_memory")))
        if command.kind == "clarification":
            return command.payload.get("message", "")
        if command.kind == "concept_teach":
            return station.teach_concept(command.payload["name"], command.payload["utterance"])
        if command.kind == "concept_forget":
            return station.forget_concept(command.payload["name"])
        if command.kind == "procedure_execute":
            return station._run_procedure(command.payload["steps"], command.utterance)
        if command.kind == "sequence_execute":
            return station._run_sequence(command.payload["steps"], command.utterance)
        if command.kind == "motor_execute":
            return station._run_motor_command(
                command.payload["action"],
                command.payload["count"],
                command.utterance,
            )
        if command.kind == "motor_sequence_execute":
            return station._run_motor_sequence_command(
                command.payload["sequence"],
                command.utterance,
            )
        if command.kind == "mission_execute":
            return station._run_mission(command.payload["steps"], command.utterance)
        if command.kind == "cache_query":
            return station.cache_summary()
        if command.kind == "status_query":
            return station.status_summary(query=command.payload.get("query", "status"))
        if command.kind == "ground_target_query":
            return station.grounded_target_summary(command.payload)
        if command.kind == "metric_query":
            return station.metric_query_summary(command)
        if command.kind == "task_selector":
            return station.task_selector_summary(command)
        if command.kind == "primitive_definition":
            return station.propose_primitive_definition(
                command.payload["definition"],
                mission_request=command.payload.get("mission_request"),
            )
        if command.kind == "knowledge_update":
            return station._apply_knowledge_update_from_payload(
                command.utterance,
                command.payload,
                source="operator",
            )
        if command.kind == "claim_reference":
            return station.claim_reference_summary(command.payload.get("ref_type", ""))
        if command.kind == "synthesis_proposal":
            return command.payload["message"]
        if command.kind in {"accept_proposal", "reject_proposal"}:
            return station.status_summary(query="help")
        if command.kind == "missing_skills":
            return command.payload.get("message", "I do not have the required capabilities.")
        if command.kind == "synthesizable":
            return command.payload.get("message", "That capability is not yet implemented.")
        if command.kind == "task_instruction":
            instruction = station.resolve_task_instruction(command.utterance)
            if instruction is None:
                return station.missing_reference_summary(command.utterance)
            if instruction != command.utterance:
                station.log(f"resolved task instruction to: {instruction}")
            result = station._run_task_from_instruction(
                command.utterance,
                instruction,
                source="operator",
                record_plan=True,
            )
            return station.result_summary(result)
        if command.kind == "unsupported":
            return command.payload.get(
                "message",
                "I cannot safely execute that capability yet.",
            )
        if command.kind == "ambiguous":
            return command.payload.get(
                "message",
                "I could not safely resolve that instruction yet.",
            )
        return station.status_summary(query="help")
