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
    ProcedureRecipe,
    SchemaValidationError,
    SensePlanTemplate,
    SkillPlanTemplate,
    TaskRequest,
    memory_updates_json_schema,
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

    def _validate_primitive_names(self, names, available_primitives, label: str) -> None:
        allowed = set(available_primitives)
        for name in names:
            if name not in allowed:
                raise ValueError(f"Unknown {label}: {name}")


class LLMCompiler(CompilerBackend):
    name = "llm_compiler"
    DEFAULT_METHOD_MAX_TOKENS = {
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
