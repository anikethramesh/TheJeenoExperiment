from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any, Callable

from .primitive_library import library_payload, primitive_names
from .schemas import (
    EvidenceFrame,
    ExecutionContract,
    ExecutionContext,
    MemoryUpdate,
    OperatorIntent,
    ProcedureRecipe,
    SchemaValidationError,
    SensePlanTemplate,
    SkillPlanTemplate,
    TaskRequest,
    memory_updates_json_schema,
    operator_intent_json_schema,
    procedure_recipe_json_schema,
    sense_plan_json_schema,
    skill_plan_json_schema,
    task_request_json_schema,
)


class CompilerBackend(ABC):
    name = "compiler"

    def __init__(self) -> None:
        self.logs: list[str] = []
        self.call_history: list[dict[str, Any]] = []

    @property
    def active_backend(self) -> str:
        return self.name

    def log(self, message: str) -> None:
        self.logs.append(message)

    def record_call(
        self,
        method_name: str,
        backend: str,
        success: bool,
        used_fallback: bool = False,
        reason: str | None = None,
        requested_max_tokens: int | None = None,
    ) -> None:
        self.call_history.append(
            {
                "method_name": method_name,
                "backend": backend,
                "success": success,
                "used_fallback": used_fallback,
                "reason": reason,
                "requested_max_tokens": requested_max_tokens,
            }
        )

    def usage_summary(self) -> dict[str, Any]:
        total_requested_max_tokens = sum(
            call["requested_max_tokens"] or 0 for call in self.call_history
        )
        return {
            "backend": self.active_backend,
            "llm_used": False,
            "llm_success_count": 0,
            "fallback_count": 0,
            "total_requested_max_tokens": total_requested_max_tokens,
            "call_history": list(self.call_history),
        }

    @abstractmethod
    def compile_task(self, instruction: str, available_task_primitives, memory) -> TaskRequest:
        raise NotImplementedError

    @abstractmethod
    def compile_procedure(self, task: TaskRequest, available_task_primitives, memory) -> ProcedureRecipe:
        raise NotImplementedError

    @abstractmethod
    def compile_sense_plan(
        self,
        evidence_frame: EvidenceFrame,
        execution_context: ExecutionContext,
        available_sensing_primitives,
        memory,
    ) -> SensePlanTemplate:
        raise NotImplementedError

    @abstractmethod
    def compile_skill_plan(
        self,
        execution_contract: ExecutionContract,
        percepts,
        available_action_primitives,
        memory,
    ) -> SkillPlanTemplate:
        raise NotImplementedError

    @abstractmethod
    def compile_memory_updates(
        self,
        final_state,
        final_claims,
        trace,
        memory,
    ) -> list[MemoryUpdate]:
        raise NotImplementedError

    @abstractmethod
    def compile_operator_intent(
        self,
        utterance: str,
        memory,
        scene_summary: dict[str, Any] | None = None,
        capability_manifest: dict[str, Any] | None = None,
        active_claims_summary: dict[str, Any] | None = None,
        pending_proposal: dict[str, Any] | None = None,
    ) -> OperatorIntent:
        raise NotImplementedError


def is_dataclass_instance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


def make_json_safe(value: Any) -> Any:
    if is_dataclass_instance(value):
        return make_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, set):
        return [make_json_safe(item) for item in sorted(value)]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    return value


def canonical_task_params(
    color: str | None = None,
    object_type: str | None = None,
    target_location: tuple[int, int] | None = None,
) -> dict[str, Any]:
    return {
        "color": color,
        "object_type": object_type,
        "target_location": target_location,
    }


class SmokeTestCompiler(CompilerBackend):
    name = "smoke_test_compiler"

    def compile_task(self, instruction: str, available_task_primitives, memory) -> TaskRequest:
        self.record_call(
            method_name="compile_task",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        normalized = instruction.strip().lower()

        door_match = re.search(
            r"go to the (?P<color>\w+) (?P<object_type>door)",
            normalized,
        )
        if door_match:
            return TaskRequest(
                instruction=instruction,
                task_type="go_to_object",
                params=canonical_task_params(
                    color=door_match.group("color"),
                    object_type=door_match.group("object_type"),
                ),
                source=self.name,
            )

        goal_match = re.search(
            r"(?:get to|go to|reach|find) the (?:(?P<color>\w+) )?(?P<object_type>goal)",
            normalized,
        )
        if goal_match:
            return TaskRequest(
                instruction=instruction,
                task_type="go_to_object",
                params=canonical_task_params(
                    color=goal_match.group("color"),
                    object_type=goal_match.group("object_type"),
                ),
                source=self.name,
            )

        return TaskRequest(
            instruction=instruction,
            task_type="search_for_object",
            params=canonical_task_params(object_type="goal"),
            source=self.name,
        )

    def compile_procedure(self, task: TaskRequest, available_task_primitives, memory) -> ProcedureRecipe:
        self.record_call(
            method_name="compile_procedure",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        recipe = memory.understanding.get(task.task_type, {}).get("default_recipe")
        if not recipe:
            raise ValueError(f"No default recipe for task type: {task.task_type}")

        self._validate_primitive_names(recipe, available_task_primitives, "task primitive")
        return ProcedureRecipe(
            task_type=task.task_type,
            steps=list(recipe),
            source=self.name,
            compiler_backend=self.name,
            validated=True,
            rationale="Deterministic smoke-test recipe from operational memory.",
        )

    def compile_sense_plan(
        self,
        evidence_frame: EvidenceFrame,
        execution_context: ExecutionContext,
        available_sensing_primitives,
        memory,
    ) -> SensePlanTemplate:
        self.record_call(
            method_name="compile_sense_plan",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        needs = set(evidence_frame.needs)
        primitives: list[str] = []
        if needs & {"object_location", "agent_pose", "adjacency_to_target", "occupancy_grid"}:
            primitives.append("parse_grid_objects")
            primitives.append("build_occupancy_grid")
        if "object_location" in needs:
            primitives.append("find_object_by_color_type")
        if "agent_pose" in needs:
            primitives.append("get_agent_pose")
        if "adjacency_to_target" in needs:
            primitives.append("check_adjacency")

        self._validate_primitive_names(primitives, available_sensing_primitives, "sensing primitive")

        required_inputs = ["observation"]
        merged_context = dict(execution_context.params)
        merged_context.update(evidence_frame.context)
        if merged_context.get("color") is not None:
            required_inputs.append("color")
        if merged_context.get("object_type") is not None:
            required_inputs.append("object_type")
        return SensePlanTemplate(
            primitives=primitives,
            required_inputs=required_inputs,
            produces=["world_sample", "operational_evidence", "percepts"],
            source=self.name,
            compiler_backend=self.name,
            validated=True,
            rationale="Deterministic sensing template based on evidence needs.",
        )

    def compile_skill_plan(
        self,
        execution_contract: ExecutionContract,
        percepts,
        available_action_primitives,
        memory,
    ) -> SkillPlanTemplate:
        self.record_call(
            method_name="compile_skill_plan",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        if execution_contract.skill == "navigate_to_object":
            primitives = ["plan_grid_path", "execute_next_path_action"]
            required_inputs = ["agent_pose", "target_location", "occupancy_grid", "direction"]
        elif execution_contract.skill == "done":
            primitives = ["done"]
            required_inputs = ["adjacency_to_target"]
        else:
            primitives = [execution_contract.skill]
            required_inputs = ["execution_contract"]

        self._validate_primitive_names(primitives, available_action_primitives, "action primitive")
        return SkillPlanTemplate(
            primitives=primitives,
            required_inputs=required_inputs,
            produces=["execution_report", "execution_context"],
            source=self.name,
            compiler_backend=self.name,
            validated=True,
            rationale="Deterministic skill template from the execution contract.",
        )

    def compile_memory_updates(
        self,
        final_state,
        final_claims,
        trace,
        memory,
    ) -> list[MemoryUpdate]:
        self.record_call(
            method_name="compile_memory_updates",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        task_request = final_state.get("task_request", {})
        resolved_task_params = final_state.get("resolved_task_params", {})
        updates = [
            MemoryUpdate(
                scope="knowledge",
                key="last_instruction",
                value=task_request.get("instruction"),
                reason="Remember the latest operator instruction.",
            ),
            MemoryUpdate(
                scope="knowledge",
                key="last_task_type",
                value=task_request.get("task_type"),
                reason="Remember the latest task type.",
            ),
        ]

        if resolved_task_params.get("color") is not None:
            updates.append(
                MemoryUpdate(
                    scope="knowledge",
                    key="target_color",
                    value=resolved_task_params.get("color"),
                    reason="Persist the active target color as durable knowledge.",
                )
            )
        if resolved_task_params.get("object_type") is not None:
            updates.append(
                MemoryUpdate(
                    scope="knowledge",
                    key="target_type",
                    value=resolved_task_params.get("object_type"),
                    reason="Persist the active target object type as durable knowledge.",
                )
            )
        if final_claims.get("target_location") is not None:
            updates.append(
                MemoryUpdate(
                    scope="episodic_memory",
                    key="known_target_location",
                    value=final_claims.get("target_location"),
                    reason="Store the last observed target location for this episode only.",
                )
            )
        if final_state.get("last_world_sample") is not None:
            updates.append(
                MemoryUpdate(
                    scope="episodic_memory",
                    key="last_world_sample",
                    value=final_state["last_world_sample"],
                    reason="Keep the last world model sample for debugging this run.",
                )
            )
        updates.append(
            MemoryUpdate(
                scope="episodic_memory",
                key="last_trace_length",
                value=len(trace),
                reason="Track how many trace events were emitted in the episode.",
            )
        )
        return updates

    def compile_operator_intent(
        self,
        utterance: str,
        memory,
        scene_summary: dict[str, Any] | None = None,
        capability_manifest: dict[str, Any] | None = None,
        active_claims_summary: dict[str, Any] | None = None,
        pending_proposal: dict[str, Any] | None = None,
    ) -> OperatorIntent:
        self.record_call(
            method_name="compile_operator_intent",
            backend=self.name,
            success=True,
            used_fallback=False,
        )
        # When a synthesis proposal is pending, classify acceptance/rejection first.
        if pending_proposal:
            normalized_quick = " ".join(utterance.lower().strip().split())
            _ACCEPT = {"yes", "ok", "okay", "sure", "go ahead", "do it", "yep", "yeah",
                       "please", "sounds good", "go for it", "correct", "that works"}
            _REJECT = {"no", "cancel", "stop", "nope", "don't", "skip", "never mind",
                       "nevermind", "abort"}
            if normalized_quick in _ACCEPT:
                return OperatorIntent(intent_type="accept_proposal", confidence=1.0, reason="")
            if normalized_quick in _REJECT:
                return OperatorIntent(intent_type="reject_proposal", confidence=1.0, reason="")
        normalized = " ".join(utterance.lower().strip().split())
        color_pattern = r"red|green|blue|yellow|purple|grey|gray"
        door_match = re.search(
            rf"\b(?:go to|go the|reach|find|get to|head to|navigate to)\s+"
            rf"(?:the )?(?P<color>{color_pattern}) (?P<object_type>door)\b",
            normalized,
        )
        _SUPERLATIVE_TERMS = frozenset([
            "farthest", "furthest", "most distant", "most far",
            "farthest away", "furthest away", "maximum distance",
        ])
        metric = (
            "euclidean" if "euclidean" in normalized
            else "manhattan"
        )
        ranked_handle = f"grounding.all_doors.ranked.{metric}.agent"
        is_navigation = re.search(
            r"\b(go to|go the|reach|find|get to|head to|navigate to)\b",
            normalized,
        ) is not None
        threshold_match = re.search(
            r"\b(?:distance\s+)?"
            r"(?P<comparison>above|greater than|more than|over|exceeds|at least|"
            r"below|less than|under|at most|within)\s+"
            r"(?P<value>\d+(?:\.\d+)?)\b",
            normalized,
        )
        if threshold_match and ("door" in normalized or active_claims_summary is not None):
            comparison_text = threshold_match.group("comparison")
            comparison = {
                "above": "above",
                "greater than": "above",
                "more than": "above",
                "over": "above",
                "exceeds": "above",
                "at least": "at_least",
                "below": "below",
                "less than": "below",
                "under": "below",
                "at most": "at_most",
                "within": "within",
            }[comparison_text]
            threshold = float(threshold_match.group("value"))
            metric = (
                "euclidean"
                if "euclidean" in normalized
                else "manhattan"
                if "manhattan" in normalized
                else None
            )
            if metric is None and active_claims_summary is not None:
                ranked = active_claims_summary.get("ranked_doors") or []
                # compact summaries are strings such as "red@7.62"; metric is not
                # always present, so default to manhattan if the operator omitted it.
                metric = "manhattan"
            metric = metric or "manhattan"
            claims_handle = f"claims.filter.threshold.{metric}"
            wants_highest = any(
                term in normalized
                for term in ("highest", "largest", "maximum", "farthest", "furthest")
            )
            wants_lowest = any(
                term in normalized
                for term in ("lowest", "smallest", "minimum", "closest", "nearest")
            )
            order = "descending" if wants_highest else "ascending" if wants_lowest else None
            ordinal = 1 if order is not None else None
            required = [claims_handle]
            if is_navigation:
                required.append("task.go_to_object.door")
            return OperatorIntent(
                intent_type="task_instruction" if is_navigation else "claim_reference",
                status_query=None,
                task_type="go_to_object" if is_navigation else None,
                target_selector=None,
                claim_reference="threshold_filter",
                grounding_query_plan={
                    "object_type": "door",
                    "operation": "filter",
                    "primitive_handle": claims_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": order,
                    "ordinal": ordinal,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": (
                        int(threshold) if threshold.is_integer() else threshold
                    ),
                    "comparison": comparison,
                    "tie_policy": "clarify",
                    "answer_fields": ["target", "distance"],
                    "required_capabilities": required,
                    "preserved_constraints": [
                        "door",
                        metric,
                        comparison,
                        str(threshold),
                        *(["highest"] if wants_highest else []),
                        *(["lowest"] if wants_lowest else []),
                    ],
                },
                capability_status="synthesizable",
                required_capabilities=required,
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a threshold claims-filter plan.",
            )
        ordinal_match = re.search(
            r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th)\s+"
            r"(closest|nearest|farthest|furthest)\b",
            normalized,
        )
        if (
            not is_navigation
            and
            ("closest" in normalized or "nearest" in normalized)
            and re.search(r"\b(second|2nd)\s+(closest|nearest)\b", normalized)
            and "door" in normalized
        ):
            return OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                target_selector=None,
                grounding_query_plan={
                    "object_type": "door",
                    "operation": "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "display",
                    "answer_fields": ["closest", "second_closest"],
                    "required_capabilities": [ranked_handle],
                    "preserved_constraints": ["closest", "second", "door", metric],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=[ranked_handle],
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a closest/second-closest answer plan.",
            )
        if ordinal_match and "door" in normalized:
            ordinal_map = {
                "second": 2,
                "2nd": 2,
                "third": 3,
                "3rd": 3,
                "fourth": 4,
                "4th": 4,
                "fifth": 5,
                "5th": 5,
            }
            ordinal = ordinal_map[ordinal_match.group(1)]
            direction = ordinal_match.group(2)
            order = "descending" if direction in {"farthest", "furthest"} else "ascending"
            return OperatorIntent(
                intent_type="task_instruction" if is_navigation else "status_query",
                status_query=None if is_navigation else "ground_target",
                task_type="go_to_object" if is_navigation else None,
                target_selector=None,
                grounding_query_plan={
                    "object_type": "door",
                    "operation": "select" if is_navigation else "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": order,
                    "ordinal": ordinal,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "clarify",
                    "answer_fields": ["target", "distance"],
                    "required_capabilities": (
                        [ranked_handle, "task.go_to_object.door"]
                        if is_navigation
                        else [ranked_handle]
                    ),
                    "preserved_constraints": [ordinal_match.group(1), direction, "door", metric],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=(
                    [ranked_handle, "task.go_to_object.door"]
                    if is_navigation
                    else [ranked_handle]
                ),
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a ranked ordinal query plan.",
            )

        distance_match = re.search(
            r"\b(?:distance\s+(?:of\s+)?|with\s+(?:a\s+)?distance\s+(?:of\s+)?)(\d+)\b",
            normalized,
        )
        if distance_match and "door" in normalized:
            distance_value = int(distance_match.group(1))
            return OperatorIntent(
                intent_type="task_instruction" if is_navigation else "status_query",
                status_query=None if is_navigation else "ground_target",
                task_type="go_to_object" if is_navigation else None,
                target_selector=None,
                grounding_query_plan={
                    "object_type": "door",
                    "operation": "select" if is_navigation else "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": distance_value,
                    "tie_policy": "clarify",
                    "answer_fields": ["target", "distance"],
                    "required_capabilities": (
                        [ranked_handle, "task.go_to_object.door"]
                        if is_navigation
                        else [ranked_handle]
                    ),
                    "preserved_constraints": ["distance", str(distance_value), "door", metric],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=(
                    [ranked_handle, "task.go_to_object.door"]
                    if is_navigation
                    else [ranked_handle]
                ),
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a distance-value query plan.",
            )

        color_mention = re.search(rf"\b(?P<color>{color_pattern})\s+door\b", normalized)
        if color_mention and (
            "how far" in normalized
            or "distance" in normalized
            or normalized.startswith("is there")
            or normalized.startswith("do you see")
        ):
            color = "grey" if color_mention.group("color") == "gray" else color_mention.group("color")
            wants_distance = "how far" in normalized or "distance" in normalized
            return OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                target_selector=None,
                grounding_query_plan={
                    "object_type": "door",
                    "operation": "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": color,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "display",
                    "answer_fields": ["distance"] if wants_distance else ["exists"],
                    "required_capabilities": [ranked_handle],
                    "preserved_constraints": [color, "door", "distance" if wants_distance else "exists"],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=[ranked_handle],
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a color-specific answer query plan.",
            )

        if (
            ("closest" in normalized or "nearest" in normalized)
            and ("farthest" in normalized or "furthest" in normalized)
            and "door" in normalized
        ):
            return OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                target_selector=None,
                grounding_query_plan={
                    "object_type": "door",
                    "operation": "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "display",
                    "answer_fields": ["closest", "farthest"],
                    "required_capabilities": [ranked_handle],
                    "preserved_constraints": ["closest", "farthest", "door", metric],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=[ranked_handle],
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a closest/farthest answer plan.",
            )

        if is_navigation and re.search(r"\b(that|it|that one|the one)\b", normalized):
            claim_handle = "grounding.claims.last_grounded_target"
            return OperatorIntent(
                intent_type="task_instruction",
                task_type="go_to_object",
                target_selector=None,
                grounding_query_plan={
                    "object_type": "door",
                    "operation": "select",
                    "primitive_handle": claim_handle,
                    "metric": None,
                    "reference": None,
                    "order": None,
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "clarify",
                    "answer_fields": ["target"],
                    "required_capabilities": [
                        claim_handle,
                        "task.go_to_object.door",
                    ],
                    "preserved_constraints": ["that"],
                },
                capability_status="executable",
                required_capabilities=[claim_handle, "task.go_to_object.door"],
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted an active-claim reference plan.",
            )

        if any(t in normalized for t in _SUPERLATIVE_TERMS) and "door" in normalized:
            return OperatorIntent(
                intent_type="task_instruction" if is_navigation else "status_query",
                status_query=None if is_navigation else "ground_target",
                task_type="go_to_object" if is_navigation else None,
                target_selector=None,
                grounding_query_plan={
                    "object_type": "door",
                    "operation": "select" if is_navigation else "answer",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "descending",
                    "ordinal": 1,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "clarify" if is_navigation else "display",
                    "answer_fields": ["farthest", "distance"],
                    "required_capabilities": (
                        [ranked_handle, "task.go_to_object.door"]
                        if is_navigation
                        else [ranked_handle]
                    ),
                    "preserved_constraints": ["farthest", "door", metric],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=(
                    [ranked_handle, "task.go_to_object.door"]
                    if is_navigation
                    else [ranked_handle]
                ),
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a farthest-door query plan.",
            )

        if any(
            phrase in normalized
            for phrase in ("next closest", "next one", "next door", "another door")
        ):
            return OperatorIntent(
                intent_type="claim_reference",
                claim_reference="next_closest",
                target_selector=None,
                required_capabilities=[],
                confidence=0.85,
                reason="Deterministic operator-intent fallback parsed a next-closest claim reference.",
            )

        if any(
            phrase in normalized
            for phrase in ("other door", "remaining door", "other one", "the other")
        ):
            return OperatorIntent(
                intent_type="claim_reference",
                claim_reference="other_door",
                target_selector=None,
                required_capabilities=[],
                confidence=0.85,
                reason="Deterministic operator-intent fallback parsed an other-door claim reference.",
            )

        is_ranked_query = any(
            phrase in normalized
            for phrase in (
                "in order", "in descending order", "in ascending order",
                "ranked", "ranking", "rank them", "rank the", "rank distances",
                "rank the distances", "list them",
                "all doors", "all of them", "list all", "each door", "each of these doors",
            )
        )

        if is_ranked_query and "door" in normalized:
            metric = (
                "euclidean"
                if "euclidean" in normalized
                else "manhattan"
                if "manhattan" in normalized
                else "manhattan"
            )
            ranked_handle = f"grounding.all_doors.ranked.{metric}.agent"
            return OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                target_selector=None,
                grounding_query_plan={
                    "object_type": "door",
                    "operation": "rank",
                    "primitive_handle": ranked_handle,
                    "metric": metric,
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "display",
                    "answer_fields": ["ranked_doors", "distance"],
                    "required_capabilities": [ranked_handle],
                    "preserved_constraints": ["rank", "door", metric],
                },
                capability_status="executable" if metric == "manhattan" else "synthesizable",
                required_capabilities=[ranked_handle],
                confidence=0.9,
                reason="Deterministic operator-intent fallback emitted a ranked-door query plan.",
            )

        if ("closest" in normalized or "nearest" in normalized or "shortest" in normalized) and "door" in normalized:
            metric = (
                "euclidean"
                if "euclidean" in normalized
                else "manhattan"
                if "manhattan" in normalized
                else None
            )
            if is_ranked_query:
                metric_suffix = metric or "manhattan"
                return OperatorIntent(
                    intent_type="status_query",
                    status_query="ground_target",
                    target_selector={
                        "object_type": "door",
                        "color": None,
                        "exclude_colors": [],
                        "relation": "closest",
                        "distance_metric": metric,
                        "distance_reference": "agent" if metric is not None else None,
                    },
                    capability_status="missing_skills",
                    required_capabilities=[f"grounding.all_doors.ranked.{metric_suffix}.agent"],
                    confidence=0.85,
                    reason="Ranked listing requires grounding.all_doors.ranked — not the same as closest.",
                )
            return OperatorIntent(
                intent_type=(
                    "knowledge_update"
                    if "delivery target" in normalized or "make" in normalized
                    else "task_instruction"
                    if re.search(r"\b(go to|go the|reach|find|get to|head to|navigate to)\b", normalized)
                    else "status_query"
                ),
                task_type=(
                    "go_to_object"
                    if re.search(r"\b(go to|go the|reach|find|get to|head to|navigate to)\b", normalized)
                    else None
                ),
                knowledge_update=(
                    {"delivery_target": None}
                    if "delivery target" in normalized or "make" in normalized
                    else None
                ),
                status_query=(
                    None
                    if re.search(r"\b(go to|go the|reach|find|get to|head to|navigate to)\b", normalized)
                    or "delivery target" in normalized
                    or "make" in normalized
                    else "ground_target"
                ),
                target_selector={
                    "object_type": "door",
                    "color": None,
                    "exclude_colors": [],
                    "relation": "closest",
                    "distance_metric": metric,
                    "distance_reference": "agent" if metric is not None else None,
                },
                capability_status=(
                    "synthesizable"
                    if metric == "euclidean"
                    else "needs_clarification"
                    if metric is None
                    else "executable"
                ),
                required_capabilities=(
                    ["grounding.closest_door.euclidean.agent"]
                    if metric == "euclidean"
                    else ["grounding.closest_door.manhattan.agent"]
                    if metric == "manhattan"
                    else []
                ),
                confidence=0.85,
                reason="Deterministic operator-intent fallback parsed a closest-door selector.",
            )

        not_color_matches = re.findall(
            rf"\b(?:not|other than|except|neither|nor)\s+(?:the )?({color_pattern})\b",
            normalized,
        )
        if not_color_matches:
            # Also capture colors joined with "or"/"nor" (e.g. "not purple or yellow")
            extra = re.findall(
                rf"\b(?:or|nor)\s+(?:the )?({color_pattern})\b",
                normalized,
            )
            not_color_matches.extend(extra)
        if "door" in normalized and not_color_matches:
            exclude_colors = [
                "grey" if c == "gray" else c
                for c in not_color_matches
            ]
            return OperatorIntent(
                intent_type="task_instruction",
                task_type="go_to_object",
                target_selector={
                    "object_type": "door",
                    "color": None,
                    "exclude_colors": exclude_colors,
                    "relation": "unique",
                    "distance_metric": None,
                    "distance_reference": None,
                },
                capability_status="executable",
                required_capabilities=[
                    "grounding.unique_door.color_filter",
                    "task.go_to_object.door",
                ],
                confidence=0.85,
                reason="Deterministic operator-intent fallback parsed an excluded-color door selector.",
            )

        if door_match:
            color = "grey" if door_match.group("color") == "gray" else door_match.group("color")
            return OperatorIntent(
                intent_type="task_instruction",
                canonical_instruction=f"go to the {color} door",
                task_type="go_to_object",
                target={"color": color, "object_type": "door"},
                target_selector=None,
                required_capabilities=["task.go_to_object.door"],
                confidence=1.0,
                reason="Deterministic operator-intent fallback parsed a door navigation task.",
            )

        delivery_match = re.search(
            rf"\b(?P<color>{color_pattern}) (?P<object_type>door)\b.*\bdelivery target\b",
            normalized,
        )
        if delivery_match:
            color = "grey" if delivery_match.group("color") == "gray" else delivery_match.group("color")
            return OperatorIntent(
                intent_type="knowledge_update",
                knowledge_update={
                    "delivery_target": {"color": color, "object_type": "door"}
                },
                target_selector=None,
                required_capabilities=[],
                confidence=1.0,
                reason="Deterministic operator-intent fallback parsed delivery target knowledge.",
            )

        if (
            re.search(r"\b(clear|delete|forget|remove)\b", normalized)
            and re.search(r"\b(target|delivery target)\b", normalized)
        ):
            return OperatorIntent(
                intent_type="knowledge_update",
                knowledge_update={"delivery_target": None},
                target_selector=None,
                required_capabilities=[],
                confidence=0.9,
                reason="Deterministic operator-intent fallback parsed target clearing.",
            )

        if "same" in normalized or "again" in normalized:
            return OperatorIntent(
                intent_type="task_instruction",
                task_type="go_to_object",
                reference="last_target",
                target_selector=None,
                required_capabilities=["task.go_to_object.door"],
                confidence=0.8,
                reason="Deterministic operator-intent fallback parsed a last-target reference.",
            )

        if "delivery target" in normalized and self._looks_like_question(normalized):
            return OperatorIntent(
                intent_type="status_query",
                status_query="delivery_target",
                target_selector=None,
                confidence=0.9,
                reason="Deterministic operator-intent fallback parsed a delivery-target query.",
            )

        if (
            "what do you see" in normalized
            or "what can you see" in normalized
            or "look around" in normalized
            or "around you" in normalized
        ):
            return OperatorIntent(
                intent_type="status_query",
                status_query="scene",
                target_selector=None,
                confidence=0.9,
                reason="Deterministic operator-intent fallback parsed a scene query.",
            )

        if (
            "what can you do" in normalized
            or "capabilit" in normalized
            or "what are you able" in normalized
            or "overview" in normalized
            or "skill" in normalized
        ):
            return OperatorIntent(
                intent_type="status_query",
                status_query="help",
                target_selector=None,
                confidence=0.8,
                reason="Deterministic operator-intent fallback parsed a capability query.",
            )

        if "last" in normalized or "previous" in normalized:
            return OperatorIntent(
                intent_type="status_query",
                status_query="last_run",
                target_selector=None,
                confidence=0.7,
                reason="Deterministic operator-intent fallback parsed a last-run query.",
            )

        return OperatorIntent(
            intent_type="unsupported",
            target_selector=None,
            confidence=0.0,
            reason="Deterministic operator-intent fallback could not resolve utterance.",
        )

    def _looks_like_question(self, normalized_utterance: str) -> bool:
        return (
            normalized_utterance.endswith("?")
            or normalized_utterance.startswith(
                (
                    "what",
                    "whats",
                    "what's",
                    "which",
                    "who",
                    "where",
                    "when",
                    "why",
                    "how",
                    "and",
                )
            )
        )

    def _validate_primitive_names(self, names, available_primitives, label: str) -> None:
        allowed = set(available_primitives)
        for name in names:
            if name not in allowed:
                raise ValueError(f"Unknown {label}: {name}")


class LLMCompiler(CompilerBackend):
    name = "llm_compiler"
    DEFAULT_METHOD_MAX_TOKENS = {
        "compile_operator_intent": 256,
        "compile_task": 256,
        "compile_procedure": 512,
        "compile_sense_plan": 384,
        "compile_skill_plan": 384,
        "compile_memory_updates": 512,
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        transport: Callable[[dict[str, Any]], Any] | None = None,
        fallback: SmokeTestCompiler | None = None,
        timeout: int = 30,
    ) -> None:
        super().__init__()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.timeout = timeout
        self.transport = transport or self._chat_completions_transport
        self.fallback = fallback or SmokeTestCompiler()
        self.fallback_reason: str | None = None
        self.default_max_tokens = int(os.getenv("OPENROUTER_MAX_TOKENS", "0") or "0")

        if not self.api_key:
            self.fallback_reason = "OPENROUTER_API_KEY not set; falling back to smoke_test_compiler."
            self.log(self.fallback_reason)

    @property
    def active_backend(self) -> str:
        if self.fallback_reason:
            return f"{self.name} -> {self.fallback.name}"
        return self.name

    def usage_summary(self) -> dict[str, Any]:
        llm_success_count = sum(
            1 for call in self.call_history if call["backend"] == self.name and call["success"]
        )
        fallback_count = sum(1 for call in self.call_history if call["used_fallback"])
        total_requested_max_tokens = sum(
            call["requested_max_tokens"] or 0 for call in self.call_history
        )
        return {
            "backend": self.active_backend,
            "llm_used": llm_success_count > 0,
            "llm_success_count": llm_success_count,
            "fallback_count": fallback_count,
            "total_requested_max_tokens": total_requested_max_tokens,
            "call_history": list(self.call_history),
        }

    def compile_task(self, instruction: str, available_task_primitives, memory) -> TaskRequest:
        task_types = sorted(memory.understanding)
        payload = {
            "instruction": instruction,
            "available_task_primitives": library_payload(available_task_primitives),
            "known_knowledge": memory.knowledge,
            "known_task_types": task_types,
        }
        return self._compile_or_fallback(
            method_name="compile_task",
            schema_name="jeenom_task_request",
            schema=task_request_json_schema(task_types),
            parser=TaskRequest.from_dict,
            system_prompt=(
                "You are the JEENOM cortical compiler. Parse the operator instruction into a "
                "typed TaskRequest. Choose only from the provided task types and never invent "
                "runtime actions or call the environment. Always include color, object_type, "
                "and target_location keys in params, using null when unknown."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_task(
                instruction, available_task_primitives, memory
            ),
        )

    def compile_operator_intent(
        self,
        utterance: str,
        memory,
        scene_summary: dict[str, Any] | None = None,
        capability_manifest: dict[str, Any] | None = None,
        active_claims_summary: dict[str, Any] | None = None,
        pending_proposal: dict[str, Any] | None = None,
    ) -> OperatorIntent:
        payload = {
            "utterance": utterance,
            "knowledge": memory.knowledge,
            "episodic_memory": memory.episodic_memory,
            "scene_summary": scene_summary,
            "capability_manifest": capability_manifest,
            "active_claims_summary": active_claims_summary,
            "pending_synthesis_proposal": pending_proposal,
            "supported": {
                "intent_types": [
                    "task_instruction",
                    "knowledge_update",
                    "status_query",
                    "cache_query",
                    "claim_reference",
                    "reset",
                    "quit",
                    "accept_proposal",
                    "reject_proposal",
                    "unsupported",
                    "ambiguous",
                ],
                "task_types": ["go_to_object"],
                "object_types": ["door"],
                "colors": ["red", "green", "blue", "yellow", "purple", "grey"],
                "references": ["delivery_target", "last_target", "last_task"],
                "status_queries": [
                    "status",
                    "scene",
                    "help",
                    "last_run",
                    "last_target",
                    "delivery_target",
                    "ground_target",
                    "cache",
                ],
            },
        }
        return self._compile_or_fallback(
            method_name="compile_operator_intent",
            schema_name="jeenom_operator_intent",
            schema=operator_intent_json_schema(),
            parser=OperatorIntent.from_dict,
            system_prompt=(
                "You are the JEENOM operator intent compiler. Convert the operator utterance "
                "into one typed OperatorIntent. You only describe intent; you do not update "
                "memory, execute tasks, or call tools. Current executable task scope is only "
                "go_to_object for door targets. For unsupported, ambiguous, non-door, pickup, "
                "open, unlock, exploration, correction, or replan requests, emit unsupported "
                "or ambiguous and do not fabricate a supported intent. Knowledge updates are "
                "narrowly limited to delivery_target. Question-shaped utterances must be "
                "status_query intents, not knowledge_update intents. Scene questions such as "
                "'what do you see around you' map to status_query=scene. Delivery-target "
                "questions map to status_query=delivery_target only when the utterance "
                "explicitly mentions delivery target. Capability or help questions such as "
                "'what can you do', 'what are your capabilities', 'what are your skills', "
                "or 'give me an overview of your capabilities' map to status_query=help "
                "and should be answered from the capability_manifest by the station. "
                "Relational target questions such as "
                "'what is the closest door to you', 'which door is nearest', or 'shortest "
                "distance to a door' must map to status_query=ground_target with a "
                "target_selector, never status_query=delivery_target. For relational target "
                "requests such as closest or not-yellow doors, emit a target_selector. Do not "
                "choose the closest object yourself; the station will ground the selector "
                "using current scene data. "
                "Use the capability_manifest to decide whether a request is executable, "
                "needs clarification, unsupported, or missing a primitive. Set "
                "capability_status explicitly: executable when the manifest says the needed "
                "primitive is implemented; needs_clarification when the request is supported "
                "but a slot like distance_metric is missing; synthesizable when the manifest "
                "says a missing primitive is safe_to_synthesize; missing_skills when a "
                "required primitive is missing and not synthesizable; unsupported when the "
                "request is outside the manifest. Closest-door Manhattan grounding is "
                "implemented. Closest-door Euclidean grounding is missing but marked safe to "
                "synthesize later; emit capability_status=synthesizable with "
                "distance_metric=euclidean rather than unsupported. If closest is requested "
                "without a metric, emit capability_status=needs_clarification and a closest "
                "selector with null distance fields so the station can ask a clarification. "
                "Example: utterance='I see. What is the closest door to you' -> "
                "intent_type=status_query, capability_status=needs_clarification, "
                "status_query=ground_target, target_selector={object_type: door, "
                "relation: closest, distance_metric: null, distance_reference: null}. "
                "If active_claims_summary is provided in the payload, the operator may be "
                "referring to a prior grounding result. Phrases like 'the next closest', "
                "'next one', 'the next door', 'the other door', 'the remaining door', or "
                "'another door' should be classified as intent_type=claim_reference. Set "
                "claim_reference=next_closest for sequential references (next one, next "
                "closest, next door) and claim_reference=other_door for residual references "
                "(the other door, the remaining door). Do NOT choose the object yourself; "
                "the station resolves claim references deterministically from active_claims. "
                "If active_claims_summary is provided and the utterance asks for a distance "
                "threshold over the prior ranked results (e.g. 'the door where euclidean "
                "distance is above 6', 'doors below 10', 'go to one with distance at least "
                "7'), emit intent_type=claim_reference, claim_reference=threshold_filter, "
                "and a grounding_query_plan with operation='filter', primitive_handle="
                "'claims.filter.threshold.<metric>', metric set to the named metric or the "
                "active claim metric, distance_value set to the numeric threshold, and "
                "comparison in ['above','below','within','at_least','at_most']. Include the "
                "claims.filter.threshold.<metric> handle in required_capabilities. Do not "
                "bake the threshold into the handle; the primitive is parametric. If the "
                "utterance asks for the highest/largest/farthest item inside the filtered "
                "set, set order='descending' and ordinal=1. If it asks for the lowest/"
                "smallest/closest item inside the filtered set, set order='ascending' and "
                "ordinal=1. "
                "REQUIRED_CAPABILITIES: Every intent must include a required_capabilities "
                "field listing the exact capability handles needed. Use the handle names "
                "exactly as they appear in the capability_manifest (e.g. "
                "'grounding.closest_door.manhattan.agent', 'task.go_to_object.door'). "
                "Include any handle the intent needs even if it is NOT in the manifest — the "
                "station's CapabilityMatcher will classify it as missing_skills. "
                "No weakening: grounding.closest_door does NOT satisfy legacy grounding.ranked_doors; "
                "grounding.closest_door does NOT satisfy grounding.nth_closest_door; "
                "task.go_to_object does NOT satisfy task.pickup. "
                "A request for ranked or ordered listing of objects (e.g. 'all doors in "
                "order', 'list doors by distance', 'doors closest to me ranked') requires "
                "'grounding.all_doors.ranked.manhattan.agent' — this is NOT the same as "
                "grounding.closest_door.manhattan.agent and must be listed separately. "
                "GROUNDING_QUERY_PLAN: For any operator request that asks about visible "
                "doors, distances, ordering, closest/farthest, ordinal ranks, color-specific "
                "existence, or a target selected from scene grounding, emit a "
                "grounding_query_plan. For non-grounding intents, set grounding_query_plan=null. "
                "Do not answer the question yourself and do not choose an object yourself. "
                "Describe the query. Use primitive_handle='grounding.all_doors.ranked.manhattan.agent' "
                "for Manhattan ranked-door plans. Use operation='rank' for ranked/list displays, "
                "operation='answer' for questions, and operation='select' for tasks that choose "
                "a grounded target. Use order='ascending' for closest/nearest and order='descending' "
                "for farthest/furthest/least-close. Convert ordinal words to integers: second=2, "
                "third=3. Example: 'can you navigate to the second farthest door' -> "
                "intent_type=task_instruction, task_type=go_to_object, grounding_query_plan={"
                "object_type:'door', operation:'select', primitive_handle:"
                "'grounding.all_doors.ranked.manhattan.agent', metric:'manhattan', "
                "reference:'agent', order:'descending', ordinal:2, color:null, "
                "exclude_colors:[], distance_value:null, tie_policy:'clarify', "
                "answer_fields:['target','distance'], required_capabilities:["
                "'grounding.all_doors.ranked.manhattan.agent','task.go_to_object.door'], "
                "preserved_constraints:['second','farthest','door','manhattan']}. "
                "Example: 'how far is the red door' -> status_query=ground_target and a "
                "grounding_query_plan with color='red', answer_fields=['distance']. "
                "Example: 'is there a green door' -> status_query=ground_target and a "
                "grounding_query_plan with color='green', answer_fields=['exists']. "
                "Example: 'which door is closest and which is farthest' -> answer_fields="
                "['closest','farthest']. "
                "Example: 'which door is closest and second closest' -> operation='answer', "
                "order='ascending', ordinal=null, answer_fields=['closest','second_closest'], "
                "preserved_constraints=['closest','second','door','manhattan']. "
                "Use answer fields first_closest, second_closest, third_closest, "
                "first_farthest, second_farthest, third_farthest for multi-answer "
                "ranked questions; do not collapse 'second closest' to ordinal=1. "
                "Reference requests such as 'go to that', 'go to it', or 'take me there' "
                "after a grounding answer must also use grounding_query_plan, not a plain "
                "target guess. Emit primitive_handle='grounding.claims.last_grounded_target', "
                "operation='select', metric=null, reference=null, order=null, ordinal=null, "
                "answer_fields=['target'], required_capabilities=["
                "'grounding.claims.last_grounded_target','task.go_to_object.door'], and "
                "preserved_constraints=['that'] or the actual pronoun used. "
                "Requests to clear/delete/forget only the current delivery target should be "
                "knowledge_update with knowledge_update={delivery_target:null}; this is not "
                "a motion task. "
                "For status queries and claim references with no grounding or task "
                "requirements, emit required_capabilities=[]."
                + (
                    " PENDING SYNTHESIS PROPOSAL: The station has proposed building a new "
                    "primitive and is awaiting operator approval. The pending proposal is in "
                    "pending_synthesis_proposal in the payload. If the operator's utterance "
                    "expresses agreement, confirmation, or approval of the proposal "
                    "(e.g. 'yes', 'yes build it', 'go ahead', 'do it', 'sure', 'yes please', "
                    "'I mean yes', 'build it', 'ok yes', 'yes that one'), emit "
                    "intent_type='accept_proposal' with required_capabilities=[]. "
                    "If the operator declines or cancels (e.g. 'no', 'cancel', 'stop', "
                    "'don't build it', 'no thanks'), emit intent_type='reject_proposal' "
                    "with required_capabilities=[]. "
                    "Only fall back to other intent types if the utterance is clearly a "
                    "new, unrelated instruction."
                    if pending_proposal else ""
                )
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_operator_intent(
                utterance,
                memory,
                scene_summary=scene_summary,
                capability_manifest=capability_manifest,
                active_claims_summary=active_claims_summary,
                pending_proposal=pending_proposal,
            ),
        )

    def compile_procedure(self, task: TaskRequest, available_task_primitives, memory) -> ProcedureRecipe:
        task_types = sorted(memory.understanding)
        primitive_list = primitive_names(available_task_primitives)
        payload = {
            "task_request": asdict(task),
            "available_task_primitives": library_payload(available_task_primitives),
            "understanding": memory.understanding,
            "knowledge": memory.knowledge,
        }
        return self._compile_or_fallback(
            method_name="compile_procedure",
            schema_name="jeenom_procedure_recipe",
            schema=procedure_recipe_json_schema(primitive_list, task_types),
            parser=ProcedureRecipe.from_dict,
            system_prompt=(
                "You are the JEENOM cortical compiler. Compose a typed high-level ProcedureRecipe "
                "using only task primitive names from the provided library. Output a reusable task "
                "recipe, not per-tick executable steps."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_procedure(
                task, available_task_primitives, memory
            ),
            allowed_primitive_names=set(primitive_list),
        )

    def compile_sense_plan(
        self,
        evidence_frame: EvidenceFrame,
        execution_context: ExecutionContext,
        available_sensing_primitives,
        memory,
    ) -> SensePlanTemplate:
        primitive_list = primitive_names(available_sensing_primitives)
        payload = {
            "evidence_frame": asdict(evidence_frame),
            "execution_context": asdict(execution_context),
            "available_sensing_primitives": library_payload(available_sensing_primitives),
            "knowledge": memory.knowledge,
            "episodic_memory": memory.episodic_memory,
        }
        return self._compile_or_fallback(
            method_name="compile_sense_plan",
            schema_name="jeenom_sense_plan",
            schema=sense_plan_json_schema(primitive_list),
            parser=SensePlanTemplate.from_dict,
            system_prompt=(
                "You are the JEENOM Sense compiler. Compose a reusable SensePlanTemplate using "
                "only the provided sensing primitive names. Do not emit per-tick observations or "
                "instant sensor results. Emit required_inputs and produces for a reusable template."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_sense_plan(
                evidence_frame,
                execution_context,
                available_sensing_primitives,
                memory,
            ),
            allowed_primitive_names=set(primitive_list),
        )

    def compile_skill_plan(
        self,
        execution_contract: ExecutionContract,
        percepts,
        available_action_primitives,
        memory,
    ) -> SkillPlanTemplate:
        primitive_list = primitive_names(available_action_primitives)
        payload = {
            "execution_contract": asdict(execution_contract),
            "percepts": asdict(percepts),
            "available_action_primitives": library_payload(available_action_primitives),
            "knowledge": memory.knowledge,
            "episodic_memory": memory.episodic_memory,
        }
        return self._compile_or_fallback(
            method_name="compile_skill_plan",
            schema_name="jeenom_skill_plan",
            schema=skill_plan_json_schema(primitive_list),
            parser=SkillPlanTemplate.from_dict,
            system_prompt=(
                "You are the JEENOM Spine compiler. Compose a reusable SkillPlanTemplate using "
                "only the provided action primitive names. Output a reusable template, not a "
                "per-tick action decision. Never execute MiniGrid actions directly."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_skill_plan(
                execution_contract,
                percepts,
                available_action_primitives,
                memory,
            ),
            allowed_primitive_names=set(primitive_list),
        )

    def compile_memory_updates(
        self,
        final_state,
        final_claims,
        trace,
        memory,
    ) -> list[MemoryUpdate]:
        payload = {
            "final_state": final_state,
            "final_claims": final_claims,
            "trace_events": trace,
            "current_memory": memory.serializable_snapshot(),
        }
        result = self._compile_or_fallback(
            method_name="compile_memory_updates",
            schema_name="jeenom_memory_updates",
            schema=memory_updates_json_schema(),
            parser=lambda data: [MemoryUpdate.from_dict(item) for item in data.get("updates", [])],
            system_prompt=(
                "You are the JEENOM memory compiler. Propose only typed MemoryUpdate records "
                "that separate durable knowledge from episodic run facts. Do not invent new "
                "memory scopes."
            ),
            user_payload=payload,
            fallback_call=lambda: self.fallback.compile_memory_updates(
                final_state, final_claims, trace, memory
            ),
        )
        return result

    def _compile_or_fallback(
        self,
        method_name: str,
        schema_name: str,
        schema: dict[str, Any],
        parser: Callable[[Any], Any],
        system_prompt: str,
        user_payload: dict[str, Any],
        fallback_call: Callable[[], Any],
        allowed_primitive_names: set[str] | None = None,
    ):
        if self.fallback_reason:
            self.record_call(
                method_name=method_name,
                backend=self.fallback.name,
                success=True,
                used_fallback=True,
                reason=self.fallback_reason,
            )
            return fallback_call()

        requested_max_tokens = self._max_tokens_for_method(method_name)
        request_payload = {
            "method_name": method_name,
            "system_prompt": system_prompt,
            "user_payload": make_json_safe(user_payload),
            "schema_name": schema_name,
            "schema": schema,
            "max_tokens": requested_max_tokens,
        }
        self.log(
            f"{method_name} requested max_tokens={requested_max_tokens} on model {self.model}"
        )

        try:
            raw_data = self.transport(request_payload)
            parsed = parser(raw_data)
            if allowed_primitive_names is not None:
                self._validate_compiler_primitives(parsed, allowed_primitive_names)
            parsed = self._normalize_compiler_output(parsed)
            self.record_call(
                method_name=method_name,
                backend=self.name,
                success=True,
                used_fallback=False,
                requested_max_tokens=requested_max_tokens,
            )
            return parsed
        except Exception as exc:  # noqa: BLE001
            message = f"{method_name} failed in llm_compiler, falling back: {exc}"
            self.log(message)
            self.record_call(
                method_name=method_name,
                backend=self.fallback.name,
                success=True,
                used_fallback=True,
                reason=str(exc),
                requested_max_tokens=requested_max_tokens,
            )
            return fallback_call()

    def _normalize_compiler_output(self, parsed):
        if isinstance(parsed, TaskRequest):
            parsed.source = self.name
            return parsed
        if isinstance(parsed, (ProcedureRecipe, SensePlanTemplate, SkillPlanTemplate)):
            parsed.source = self.name
            parsed.compiler_backend = self.name
            parsed.validated = True
            return parsed
        return parsed

    def _validate_compiler_primitives(self, parsed, allowed: set[str]) -> None:
        if isinstance(parsed, ProcedureRecipe):
            primitives = parsed.steps
        elif isinstance(parsed, SensePlanTemplate):
            primitives = parsed.primitives
        elif isinstance(parsed, SkillPlanTemplate):
            primitives = parsed.primitives
        else:
            return

        for primitive in primitives:
            if primitive not in allowed:
                raise SchemaValidationError(
                    f"Unknown primitive emitted by compiler: {primitive}"
                )

    def _max_tokens_for_method(self, method_name: str) -> int:
        if self.default_max_tokens > 0:
            return self.default_max_tokens
        return self.DEFAULT_METHOD_MAX_TOKENS.get(method_name, 384)

    def _chat_completions_transport(self, request_payload: dict[str, Any]) -> Any:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request_payload["system_prompt"]},
                {
                    "role": "user",
                    "content": json.dumps(request_payload["user_payload"], indent=2),
                },
            ],
            "max_tokens": request_payload["max_tokens"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request_payload["schema_name"],
                    "schema": request_payload["schema"],
                    "strict": True,
                },
            },
        }

        req = urllib.request.Request(
            url="https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP error {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter network error: {exc}") from exc

        message = raw_response["choices"][0]["message"]
        refusal = message.get("refusal")
        if refusal:
            raise RuntimeError(f"Model refusal: {refusal}")

        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Expected string JSON content from OpenRouter response")

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"Model did not return valid JSON: {exc}") from exc


def build_compiler(name: str) -> CompilerBackend:
    if name == "smoke_test":
        return SmokeTestCompiler()
    if name == "llm":
        return LLMCompiler()
    raise ValueError(f"Unknown compiler backend: {name}")
