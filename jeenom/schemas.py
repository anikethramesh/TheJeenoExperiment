from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping


SENSE_TEMPLATE_ALLOWED_INPUTS = (
    "observation",
    "mission",
    "direction",
    "color",
    "object_type",
    "target_location",
    "agent_pose",
    "occupancy_grid",
    "grid_objects",
    "known_target_location",
    "execution_context",
)
SENSE_TEMPLATE_ALLOWED_OUTPUTS = (
    "world_sample",
    "operational_evidence",
    "percepts",
)
SKILL_TEMPLATE_ALLOWED_INPUTS = (
    "agent_pose",
    "target_location",
    "occupancy_grid",
    "direction",
    "object_type",
    "color",
    "adjacency_to_target",
    "passable_positions",
    "grid_size",
    "target_object",
    "execution_contract",
    "percepts",
)
SKILL_TEMPLATE_ALLOWED_OUTPUTS = (
    "execution_report",
    "execution_context",
)


class SchemaValidationError(ValueError):
    """Raised when compiler output does not satisfy the typed JEENOM schemas."""


def _ensure_mapping(data: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise SchemaValidationError(f"{label} must be an object, got {type(data).__name__}")
    return data


def _ensure_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError(f"{label} must be a string")
    return value


def _ensure_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{label} must be a boolean")
    return value


def _ensure_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{label} must be a list")
    return value


def _ensure_str_list(value: Any, label: str) -> list[str]:
    values = _ensure_list(value, label)
    result: list[str] = []
    for idx, item in enumerate(values):
        result.append(_ensure_str(item, f"{label}[{idx}]"))
    return result


def _ensure_dict(value: Any, label: str) -> dict[str, Any]:
    mapping = _ensure_mapping(value, label)
    return dict(mapping)


def _ensure_compiler_params(value: Any, label: str) -> dict[str, Any]:
    params = _ensure_dict(value, label)
    required_keys = ("color", "object_type", "target_location")
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise SchemaValidationError(
            f"{label} missing required keys: {', '.join(missing_keys)}"
        )

    extra_keys = sorted(set(params) - set(required_keys))
    if extra_keys:
        raise SchemaValidationError(
            f"{label} has unsupported keys: {', '.join(extra_keys)}"
        )

    for key in ("color", "object_type"):
        field_value = params.get(key)
        if field_value is not None and not isinstance(field_value, str):
            raise SchemaValidationError(f"{label}.{key} must be a string or null")

    target_location = params.get("target_location")
    if target_location is not None:
        if (
            not isinstance(target_location, list)
            or len(target_location) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in target_location)
        ):
            raise SchemaValidationError(
                f"{label}.target_location must be [int, int] or null"
            )
        params["target_location"] = tuple(target_location)

    return params


def _ensure_subset(
    values: list[str],
    allowed: tuple[str, ...],
    label: str,
) -> list[str]:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise SchemaValidationError(f"{label} contains unsupported values: {', '.join(unknown)}")
    return values


@dataclass
class PrimitiveCall:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> PrimitiveCall:
        mapping = _ensure_mapping(data, "PrimitiveCall")
        return cls(
            name=_ensure_str(mapping.get("name"), "PrimitiveCall.name"),
            params=_ensure_compiler_params(mapping.get("params", {}), "PrimitiveCall.params"),
        )


@dataclass
class TaskRequest:
    instruction: str
    task_type: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "compiler"

    @classmethod
    def from_dict(cls, data: Any) -> TaskRequest:
        mapping = _ensure_mapping(data, "TaskRequest")
        return cls(
            instruction=_ensure_str(mapping.get("instruction"), "TaskRequest.instruction"),
            task_type=_ensure_str(mapping.get("task_type"), "TaskRequest.task_type"),
            params=_ensure_compiler_params(mapping.get("params", {}), "TaskRequest.params"),
            source=_ensure_str(mapping.get("source", "compiler"), "TaskRequest.source"),
        )


@dataclass
class ProcedureRecipe:
    task_type: str
    steps: list[str]
    source: str = "compiler"
    compiler_backend: str = "compiler"
    validated: bool = False
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> ProcedureRecipe:
        mapping = _ensure_mapping(data, "ProcedureRecipe")
        return cls(
            task_type=_ensure_str(mapping.get("task_type"), "ProcedureRecipe.task_type"),
            steps=_ensure_str_list(mapping.get("steps"), "ProcedureRecipe.steps"),
            source=_ensure_str(mapping.get("source", "compiler"), "ProcedureRecipe.source"),
            compiler_backend=_ensure_str(
                mapping.get("compiler_backend", "compiler"),
                "ProcedureRecipe.compiler_backend",
            ),
            validated=_ensure_bool(mapping.get("validated"), "ProcedureRecipe.validated"),
            rationale=_ensure_str(mapping.get("rationale", ""), "ProcedureRecipe.rationale"),
        )


@dataclass
class EvidenceFrame:
    needs: list[str]
    context: dict[str, Any] = field(default_factory=dict)
    active_step: str | None = None
    step_index: int = 0


@dataclass
class SensePlanTemplate:
    primitives: list[str]
    required_inputs: list[str]
    produces: list[str]
    source: str = "compiler"
    compiler_backend: str = "compiler"
    validated: bool = False
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> SensePlanTemplate:
        mapping = _ensure_mapping(data, "SensePlanTemplate")
        return cls(
            primitives=_ensure_str_list(mapping.get("primitives"), "SensePlanTemplate.primitives"),
            required_inputs=_ensure_subset(
                _ensure_str_list(
                    mapping.get("required_inputs"),
                    "SensePlanTemplate.required_inputs",
                ),
                SENSE_TEMPLATE_ALLOWED_INPUTS,
                "SensePlanTemplate.required_inputs",
            ),
            produces=_ensure_subset(
                _ensure_str_list(mapping.get("produces"), "SensePlanTemplate.produces"),
                SENSE_TEMPLATE_ALLOWED_OUTPUTS,
                "SensePlanTemplate.produces",
            ),
            source=_ensure_str(mapping.get("source", "compiler"), "SensePlanTemplate.source"),
            compiler_backend=_ensure_str(
                mapping.get("compiler_backend", "compiler"),
                "SensePlanTemplate.compiler_backend",
            ),
            validated=_ensure_bool(mapping.get("validated"), "SensePlanTemplate.validated"),
            rationale=_ensure_str(mapping.get("rationale", ""), "SensePlanTemplate.rationale"),
        )


@dataclass
class WorldModelSample:
    mission: str | None = None
    direction: int | None = None
    step_count: int = 0
    raw_image: Any | None = None
    grid_size: tuple[int, int] | None = None
    grid_objects: list[dict[str, Any]] = field(default_factory=list)
    occupancy_grid: list[list[bool]] = field(default_factory=list)
    passable_positions: set[tuple[int, int]] = field(default_factory=set)
    agent_pose: dict[str, Any] | None = None
    target_visible: bool = False
    target_location: tuple[int, int] | None = None
    target_object: dict[str, Any] | None = None
    adjacency_to_target: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "mission": self.mission,
            "direction": self.direction,
            "step_count": self.step_count,
            "grid_size": self.grid_size,
            "agent_pose": self.agent_pose,
            "target_visible": self.target_visible,
            "target_location": self.target_location,
            "target_object": self.target_object,
            "adjacency_to_target": self.adjacency_to_target,
        }


@dataclass
class OperationalEvidence:
    claims: dict[str, Any]
    confidence: float = 1.0
    source: str = "sense"


@dataclass
class Percepts:
    cues: dict[str, Any] = field(default_factory=dict)
    source: str = "sense"


@dataclass
class ExecutionContract:
    skill: str
    params: dict[str, Any] = field(default_factory=dict)
    stop_conditions: list[str] = field(default_factory=list)
    source: str = "cortex"


@dataclass
class SkillPlanTemplate:
    primitives: list[str]
    required_inputs: list[str]
    produces: list[str]
    source: str = "compiler"
    compiler_backend: str = "compiler"
    validated: bool = False
    rationale: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> SkillPlanTemplate:
        mapping = _ensure_mapping(data, "SkillPlanTemplate")
        return cls(
            primitives=_ensure_str_list(mapping.get("primitives"), "SkillPlanTemplate.primitives"),
            required_inputs=_ensure_subset(
                _ensure_str_list(
                    mapping.get("required_inputs"),
                    "SkillPlanTemplate.required_inputs",
                ),
                SKILL_TEMPLATE_ALLOWED_INPUTS,
                "SkillPlanTemplate.required_inputs",
            ),
            produces=_ensure_subset(
                _ensure_str_list(mapping.get("produces"), "SkillPlanTemplate.produces"),
                SKILL_TEMPLATE_ALLOWED_OUTPUTS,
                "SkillPlanTemplate.produces",
            ),
            source=_ensure_str(mapping.get("source", "compiler"), "SkillPlanTemplate.source"),
            compiler_backend=_ensure_str(
                mapping.get("compiler_backend", "compiler"),
                "SkillPlanTemplate.compiler_backend",
            ),
            validated=_ensure_bool(mapping.get("validated"), "SkillPlanTemplate.validated"),
            rationale=_ensure_str(mapping.get("rationale", ""), "SkillPlanTemplate.rationale"),
        )


@dataclass
class ExecutionReport:
    status: str
    progress: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    source: str = "spine"


@dataclass
class ExecutionContext:
    active_skill: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryUpdate:
    scope: str
    key: str
    value: Any
    reason: str

    @classmethod
    def from_dict(cls, data: Any) -> MemoryUpdate:
        mapping = _ensure_mapping(data, "MemoryUpdate")
        return cls(
            scope=_ensure_str(mapping.get("scope"), "MemoryUpdate.scope"),
            key=_ensure_str(mapping.get("key"), "MemoryUpdate.key"),
            value=mapping.get("value"),
            reason=_ensure_str(mapping.get("reason", ""), "MemoryUpdate.reason"),
        )


@dataclass
class ReadinessReport:
    status: str
    task_type: str
    missing_task_primitives: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    missing_actions: list[str] = field(default_factory=list)
    recipe_steps: list[str] = field(default_factory=list)


@dataclass
class TraceEvent:
    event: str
    payload: dict[str, Any] = field(default_factory=dict)
    loop_index: int | None = None
    step_name: str | None = None


@dataclass
class PlanCacheEntry:
    key: str
    template_type: Literal["procedure", "sense", "skill"]
    template: ProcedureRecipe | SensePlanTemplate | SkillPlanTemplate
    compiler_backend: str
    source: str
    created_at_loop: int
    hit_count: int = 0


@dataclass
class PlanCacheStats:
    hits: int = 0
    misses: int = 0
    llm_calls_saved: int = 0


SensePrimitivePlan = SensePlanTemplate
SpineSkillPlan = SkillPlanTemplate


def task_request_json_schema(task_types: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "instruction": {"type": "string"},
            "task_type": {"type": "string", "enum": task_types},
            "params": primitive_params_json_schema(),
            "source": {"type": "string"},
        },
        "required": ["instruction", "task_type", "params", "source"],
        "additionalProperties": False,
    }


def procedure_recipe_json_schema(primitive_names: list[str], task_types: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "task_type": {"type": "string", "enum": task_types},
            "steps": {
                "type": "array",
                "items": {"type": "string", "enum": primitive_names},
            },
            "source": {"type": "string"},
            "compiler_backend": {"type": "string"},
            "validated": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": [
            "task_type",
            "steps",
            "source",
            "compiler_backend",
            "validated",
            "rationale",
        ],
        "additionalProperties": False,
    }


def sense_plan_json_schema(primitive_names: list[str]) -> dict[str, Any]:
    return template_json_schema(
        primitive_names=primitive_names,
        allowed_inputs=list(SENSE_TEMPLATE_ALLOWED_INPUTS),
        allowed_outputs=list(SENSE_TEMPLATE_ALLOWED_OUTPUTS),
    )


def skill_plan_json_schema(primitive_names: list[str]) -> dict[str, Any]:
    return template_json_schema(
        primitive_names=primitive_names,
        allowed_inputs=list(SKILL_TEMPLATE_ALLOWED_INPUTS),
        allowed_outputs=list(SKILL_TEMPLATE_ALLOWED_OUTPUTS),
    )


def template_json_schema(
    primitive_names: list[str],
    allowed_inputs: list[str],
    allowed_outputs: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "primitives": {
                "type": "array",
                "items": {"type": "string", "enum": primitive_names},
            },
            "required_inputs": {
                "type": "array",
                "items": {"type": "string", "enum": allowed_inputs},
            },
            "produces": {
                "type": "array",
                "items": {"type": "string", "enum": allowed_outputs},
            },
            "source": {"type": "string"},
            "compiler_backend": {"type": "string"},
            "validated": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
        "required": [
            "primitives",
            "required_inputs",
            "produces",
            "source",
            "compiler_backend",
            "validated",
            "rationale",
        ],
        "additionalProperties": False,
    }


def memory_updates_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "updates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["knowledge", "episodic_memory"],
                        },
                        "key": {"type": "string"},
                        "value": {
                            "type": ["string", "number", "boolean", "object", "array", "null"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["scope", "key", "value", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["updates"],
        "additionalProperties": False,
    }


def primitive_call_json_schema(primitive_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": primitive_names},
            "params": primitive_params_json_schema(),
        },
        "required": ["name", "params"],
        "additionalProperties": False,
    }


def primitive_params_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "color": {"type": ["string", "null"]},
            "object_type": {"type": ["string", "null"]},
            "target_location": {
                "type": ["array", "null"],
                "items": {"type": "integer"},
            },
        },
        "required": ["color", "object_type", "target_location"],
        "additionalProperties": False,
    }
