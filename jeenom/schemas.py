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
OPERATOR_INTENT_TYPES = (
    "task_instruction",
    "knowledge_update",
    "status_query",
    "claim_reference",
    "cache_query",
    "reset",
    "quit",
    "accept_proposal",
    "reject_proposal",
    "unsupported",
    "ambiguous",
)
OPERATOR_CLAIM_REFERENCES = ("next_closest", "other_door", "threshold_filter")
GROUNDING_QUERY_COMPARISONS = ("above", "below", "within", "at_least", "at_most")
OPERATOR_TASK_TYPES = ("go_to_object",)
OPERATOR_OBJECT_TYPES = ("door",)
OPERATOR_COLORS = ("red", "green", "blue", "yellow", "purple", "grey")
OPERATOR_REFERENCES = ("delivery_target", "last_target", "last_task")
OPERATOR_SELECTOR_RELATIONS = ("closest", "unique")
OPERATOR_DISTANCE_METRICS = ("manhattan", "euclidean")
OPERATOR_DISTANCE_REFERENCES = ("agent",)
GROUNDING_QUERY_OPERATIONS = ("list", "filter", "rank", "select", "answer")
GROUNDING_QUERY_ORDERS = ("ascending", "descending")
GROUNDING_QUERY_TIE_POLICIES = ("clarify", "display")
OPERATOR_STATUS_QUERIES = (
    "status",
    "scene",
    "help",
    "last_run",
    "last_target",
    "delivery_target",
    "ground_target",
    "cache",
)
OPERATOR_CONTROLS = ("reset", "quit")
OPERATOR_CAPABILITY_STATUSES = (
    "executable",
    "needs_clarification",
    "missing_skills",
    "synthesizable",
    "unsupported",
)
REQUEST_OBJECTIVE_TYPES = (
    "task",
    "query",
    "knowledge_update",
    "control",
    "unsupported",
)
REQUEST_PLAN_LAYERS = (
    "task",
    "grounding",
    "claims",
    "sensing",
    "action",
    "memory",
    "answer",
    "control",
)
REQUEST_PLAN_OPERATIONS = (
    "rank",
    "filter",
    "select",
    "ground",
    "execute",
    "answer",
    "read",
    "update",
    "reset",
    "refuse",
)
REQUEST_EXPECTED_RESPONSES = (
    "execute_task",
    "answer_query",
    "ask_clarification",
    "propose_synthesis",
    "update_memory",
    "refuse",
)
REQUEST_TIE_POLICIES = ("clarify", "display_ties", "fail")
READINESS_NODE_STATUSES = (
    "executable",
    "needs_clarification",
    "synthesizable",
    "missing_skills",
    "unsupported",
    "stale_claims",
    "blocked_by_dependency",
)
READINESS_NEXT_ACTIONS = (
    "execute_task",
    "answer_query",
    "ask_clarification",
    "propose_synthesis",
    "refresh_claims",
    "update_memory",
    "refuse",
)
ARBITRATION_DECISION_TYPES = (
    "substitute",
    "clarify",
    "synthesize",
    "refuse",
)
PRIMITIVE_SPEC_TYPES = ("task", "grounding", "sensing", "action", "claims")
PRIMITIVE_IMPLEMENTATION_STATUSES = (
    "implemented",
    "unsupported",
    "synthesizable",
    "planned",
    "missing",
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


def _ensure_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{label} must be a number")
    result = float(value)
    if result < 0.0 or result > 1.0:
        raise SchemaValidationError(f"{label} must be between 0.0 and 1.0")
    return result


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


def _ensure_optional_str_enum(
    value: Any,
    allowed: tuple[str, ...],
    label: str,
) -> str | None:
    if value is None:
        return None
    string_value = _ensure_str(value, label)
    if string_value not in allowed:
        raise SchemaValidationError(
            f"{label} must be one of: {', '.join(allowed)}"
        )
    return string_value


def _ensure_optional_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{label} must be an integer or null")
    return value


def _ensure_operator_target(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    target = _ensure_dict(value, label)
    required_keys = ("color", "object_type")
    missing = [key for key in required_keys if key not in target]
    if missing:
        raise SchemaValidationError(f"{label} missing required keys: {', '.join(missing)}")
    extra = sorted(set(target) - set(required_keys))
    if extra:
        raise SchemaValidationError(f"{label} has unsupported keys: {', '.join(extra)}")
    return {
        "color": _ensure_optional_str_enum(target.get("color"), OPERATOR_COLORS, f"{label}.color"),
        "object_type": _ensure_optional_str_enum(
            target.get("object_type"),
            OPERATOR_OBJECT_TYPES,
            f"{label}.object_type",
        ),
    }


def _ensure_operator_knowledge_update(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    update = _ensure_dict(value, label)
    required_keys = ("delivery_target",)
    missing = [key for key in required_keys if key not in update]
    if missing:
        raise SchemaValidationError(f"{label} missing required keys: {', '.join(missing)}")
    extra = sorted(set(update) - set(required_keys))
    if extra:
        raise SchemaValidationError(f"{label} has unsupported keys: {', '.join(extra)}")
    raw_delivery_target = update.get("delivery_target")
    if raw_delivery_target is None:
        return {"delivery_target": None}
    delivery_target = _ensure_operator_target(raw_delivery_target, f"{label}.delivery_target")
    if delivery_target.get("color") is None or delivery_target.get("object_type") is None:
        raise SchemaValidationError(f"{label}.delivery_target must be fully specified")
    return {"delivery_target": delivery_target}


def _ensure_primitive_spec(value: Any, label: str) -> dict[str, Any]:
    spec = _ensure_dict(value, label)
    required_keys = (
        "name",
        "primitive_type",
        "layer",
        "description",
        "inputs",
        "outputs",
        "side_effects",
        "implementation_status",
        "safe_to_synthesize",
        "runtime_binding",
    )
    missing = [key for key in required_keys if key not in spec]
    if missing:
        raise SchemaValidationError(f"{label} missing required keys: {', '.join(missing)}")
    extra = sorted(set(spec) - set(required_keys))
    if extra:
        raise SchemaValidationError(f"{label} has unsupported keys: {', '.join(extra)}")
    primitive_type = _ensure_str(spec["primitive_type"], f"{label}.primitive_type")
    if primitive_type not in PRIMITIVE_SPEC_TYPES:
        raise SchemaValidationError(
            f"{label}.primitive_type must be one of: {', '.join(PRIMITIVE_SPEC_TYPES)}"
        )
    implementation_status = _ensure_str(
        spec["implementation_status"],
        f"{label}.implementation_status",
    )
    if implementation_status not in PRIMITIVE_IMPLEMENTATION_STATUSES:
        raise SchemaValidationError(
            f"{label}.implementation_status must be one of: "
            + ", ".join(PRIMITIVE_IMPLEMENTATION_STATUSES)
        )
    return {
        "name": _ensure_str(spec["name"], f"{label}.name"),
        "primitive_type": primitive_type,
        "layer": _ensure_str(spec["layer"], f"{label}.layer"),
        "description": _ensure_str(spec["description"], f"{label}.description"),
        "inputs": _ensure_str_list(spec["inputs"], f"{label}.inputs"),
        "outputs": _ensure_str_list(spec["outputs"], f"{label}.outputs"),
        "side_effects": _ensure_str_list(spec["side_effects"], f"{label}.side_effects"),
        "implementation_status": implementation_status,
        "safe_to_synthesize": _ensure_bool(
            spec["safe_to_synthesize"],
            f"{label}.safe_to_synthesize",
        ),
        "runtime_binding": spec["runtime_binding"],
    }


def _ensure_primitive_manifest(value: Any, label: str) -> dict[str, Any]:
    manifest = _ensure_dict(value, label)
    required_keys = ("name", "primitives")
    missing = [key for key in required_keys if key not in manifest]
    if missing:
        raise SchemaValidationError(f"{label} missing required keys: {', '.join(missing)}")
    extra = sorted(set(manifest) - set(required_keys))
    if extra:
        raise SchemaValidationError(f"{label} has unsupported keys: {', '.join(extra)}")
    primitives = _ensure_list(manifest["primitives"], f"{label}.primitives")
    return {
        "name": _ensure_str(manifest["name"], f"{label}.name"),
        "primitives": [
            _ensure_primitive_spec(item, f"{label}.primitives[{idx}]")
            for idx, item in enumerate(primitives)
        ],
    }


def _migrate_exclude_color(selector: dict[str, Any]) -> None:
    """Migrate legacy exclude_color (str) to exclude_colors (list[str]) in-place."""
    if "exclude_color" in selector and "exclude_colors" not in selector:
        val = selector.pop("exclude_color")
        selector["exclude_colors"] = [val] if val else []
    elif "exclude_color" in selector:
        selector.pop("exclude_color")
    if "exclude_colors" not in selector:
        selector["exclude_colors"] = []


def _ensure_target_selector(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    selector = _ensure_dict(value, label)
    # Support both legacy exclude_color (single) and new exclude_colors (list)
    _migrate_exclude_color(selector)
    required_keys = (
        "object_type",
        "color",
        "exclude_colors",
        "relation",
        "distance_metric",
        "distance_reference",
    )
    missing = [key for key in required_keys if key not in selector]
    if missing:
        raise SchemaValidationError(f"{label} missing required keys: {', '.join(missing)}")
    extra = sorted(set(selector) - set(required_keys))
    if extra:
        raise SchemaValidationError(f"{label} has unsupported keys: {', '.join(extra)}")

    raw_exclude = selector.get("exclude_colors") or []
    if not isinstance(raw_exclude, list):
        raw_exclude = [raw_exclude] if raw_exclude else []
    validated_exclude = []
    for c in raw_exclude:
        validated = _ensure_optional_str_enum(c, OPERATOR_COLORS, f"{label}.exclude_colors[]")
        if validated:
            validated_exclude.append(validated)

    result = {
        "object_type": _ensure_optional_str_enum(
            selector.get("object_type"),
            OPERATOR_OBJECT_TYPES,
            f"{label}.object_type",
        ),
        "color": _ensure_optional_str_enum(
            selector.get("color"),
            OPERATOR_COLORS,
            f"{label}.color",
        ),
        "exclude_colors": validated_exclude,
        "relation": _ensure_optional_str_enum(
            selector.get("relation"),
            OPERATOR_SELECTOR_RELATIONS,
            f"{label}.relation",
        ),
        "distance_metric": _ensure_optional_str_enum(
            selector.get("distance_metric"),
            OPERATOR_DISTANCE_METRICS,
            f"{label}.distance_metric",
        ),
        "distance_reference": _ensure_optional_str_enum(
            selector.get("distance_reference"),
            OPERATOR_DISTANCE_REFERENCES,
            f"{label}.distance_reference",
        ),
    }
    if result["object_type"] != "door":
        raise SchemaValidationError(f"{label}.object_type must be door")
    return result


def _ensure_grounding_query_plan(value: Any, label: str) -> dict[str, Any] | None:
    if value is None:
        return None
    plan = _ensure_dict(value, label)
    required_keys = (
        "object_type",
        "operation",
        "primitive_handle",
        "metric",
        "reference",
        "order",
        "ordinal",
        "color",
        "exclude_colors",
        "distance_value",
        "tie_policy",
        "answer_fields",
        "required_capabilities",
        "preserved_constraints",
    )
    optional_keys = ("comparison",)
    missing = [key for key in required_keys if key not in plan]
    if missing:
        raise SchemaValidationError(f"{label} missing required keys: {', '.join(missing)}")
    extra = sorted(set(plan) - set(required_keys) - set(optional_keys))
    if extra:
        raise SchemaValidationError(f"{label} has unsupported keys: {', '.join(extra)}")

    raw_exclude = plan.get("exclude_colors") or []
    if not isinstance(raw_exclude, list):
        raise SchemaValidationError(f"{label}.exclude_colors must be a list")
    exclude_colors: list[str] = []
    for idx, color in enumerate(raw_exclude):
        validated = _ensure_optional_str_enum(
            color,
            OPERATOR_COLORS,
            f"{label}.exclude_colors[{idx}]",
        )
        if validated is not None:
            exclude_colors.append(validated)

    primitive_handle = plan.get("primitive_handle")
    if primitive_handle is not None:
        primitive_handle = _ensure_str(primitive_handle, f"{label}.primitive_handle")

    result = {
        "object_type": _ensure_optional_str_enum(
            plan.get("object_type"),
            OPERATOR_OBJECT_TYPES,
            f"{label}.object_type",
        ),
        "operation": _ensure_optional_str_enum(
            plan.get("operation"),
            GROUNDING_QUERY_OPERATIONS,
            f"{label}.operation",
        ),
        "primitive_handle": primitive_handle,
        "metric": _ensure_optional_str_enum(
            plan.get("metric"),
            OPERATOR_DISTANCE_METRICS,
            f"{label}.metric",
        ),
        "reference": _ensure_optional_str_enum(
            plan.get("reference"),
            OPERATOR_DISTANCE_REFERENCES,
            f"{label}.reference",
        ),
        "order": _ensure_optional_str_enum(
            plan.get("order"),
            GROUNDING_QUERY_ORDERS,
            f"{label}.order",
        ),
        "ordinal": _ensure_optional_int(plan.get("ordinal"), f"{label}.ordinal"),
        "color": _ensure_optional_str_enum(
            plan.get("color"),
            OPERATOR_COLORS,
            f"{label}.color",
        ),
        "exclude_colors": exclude_colors,
        "distance_value": _ensure_optional_int(
            plan.get("distance_value"),
            f"{label}.distance_value",
        ),
        "comparison": _ensure_optional_str_enum(
            plan.get("comparison"),
            GROUNDING_QUERY_COMPARISONS,
            f"{label}.comparison",
        ),
        "tie_policy": _ensure_optional_str_enum(
            plan.get("tie_policy"),
            GROUNDING_QUERY_TIE_POLICIES,
            f"{label}.tie_policy",
        ),
        "answer_fields": _ensure_str_list(
            plan.get("answer_fields"),
            f"{label}.answer_fields",
        ),
        "required_capabilities": _ensure_str_list(
            plan.get("required_capabilities"),
            f"{label}.required_capabilities",
        ),
        "preserved_constraints": _ensure_str_list(
            plan.get("preserved_constraints"),
            f"{label}.preserved_constraints",
        ),
    }
    if result["object_type"] != "door":
        raise SchemaValidationError(f"{label}.object_type must be door")
    if result["operation"] is None:
        raise SchemaValidationError(f"{label}.operation must not be null")
    ordinal = result["ordinal"]
    if ordinal is not None and ordinal < 1:
        raise SchemaValidationError(f"{label}.ordinal must be >= 1")
    distance_value = result["distance_value"]
    if distance_value is not None and distance_value < 0:
        raise SchemaValidationError(f"{label}.distance_value must be >= 0")
    if result["metric"] is not None and result["reference"] is None:
        raise SchemaValidationError(f"{label}.reference is required when metric is set")
    if result["operation"] in {"rank", "select"} and result["metric"] is not None:
        handle = result["primitive_handle"]
        if handle is None:
            raise SchemaValidationError(f"{label}.primitive_handle is required for ranked plans")
    return result


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
class TargetSelector:
    object_type: str
    color: str | None = None
    exclude_colors: list[str] = field(default_factory=list)
    relation: str | None = None
    distance_metric: str | None = None
    distance_reference: str | None = None

    @classmethod
    def from_dict(cls, data: Any) -> TargetSelector:
        selector = _ensure_target_selector(data, "TargetSelector")
        if selector is None:
            raise SchemaValidationError("TargetSelector must not be null")
        return cls(**selector)


@dataclass
class GroundingQueryPlan:
    object_type: str
    operation: str
    primitive_handle: str | None = None
    metric: str | None = None
    reference: str | None = None
    order: str | None = None
    ordinal: int | None = None
    color: str | None = None
    exclude_colors: list[str] = field(default_factory=list)
    distance_value: int | None = None
    comparison: str | None = None
    tie_policy: str | None = "clarify"
    answer_fields: list[str] = field(default_factory=list)
    required_capabilities: list[str] = field(default_factory=list)
    preserved_constraints: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> GroundingQueryPlan:
        plan = _ensure_grounding_query_plan(data, "GroundingQueryPlan")
        if plan is None:
            raise SchemaValidationError("GroundingQueryPlan must not be null")
        return cls(**plan)


@dataclass
class RequestPlanStep:
    step_id: str
    layer: str
    operation: str
    required_handle: str | None = None
    implementation_status: str | None = None
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    tie_policy: str = "clarify"
    memory_reads: list[str] = field(default_factory=list)
    memory_writes: list[str] = field(default_factory=list)
    scene_fingerprint_required: bool = False

    @classmethod
    def from_dict(cls, data: Any) -> RequestPlanStep:
        mapping = _ensure_mapping(data, "RequestPlanStep")
        layer = _ensure_optional_str_enum(
            mapping.get("layer"),
            REQUEST_PLAN_LAYERS,
            "RequestPlanStep.layer",
        )
        operation = _ensure_optional_str_enum(
            mapping.get("operation"),
            REQUEST_PLAN_OPERATIONS,
            "RequestPlanStep.operation",
        )
        implementation_status = _ensure_optional_str_enum(
            mapping.get("implementation_status"),
            PRIMITIVE_IMPLEMENTATION_STATUSES,
            "RequestPlanStep.implementation_status",
        )
        tie_policy = _ensure_optional_str_enum(
            mapping.get("tie_policy", "clarify"),
            REQUEST_TIE_POLICIES,
            "RequestPlanStep.tie_policy",
        )
        required_handle = mapping.get("required_handle")
        if required_handle is not None:
            required_handle = _ensure_str(required_handle, "RequestPlanStep.required_handle")
        return cls(
            step_id=_ensure_str(mapping.get("step_id"), "RequestPlanStep.step_id"),
            layer=layer or "control",
            operation=operation or "refuse",
            required_handle=required_handle,
            implementation_status=implementation_status,
            inputs=_ensure_dict(mapping.get("inputs", {}), "RequestPlanStep.inputs"),
            outputs=_ensure_str_list(mapping.get("outputs", []), "RequestPlanStep.outputs"),
            depends_on=_ensure_str_list(
                mapping.get("depends_on", []),
                "RequestPlanStep.depends_on",
            ),
            constraints=_ensure_dict(
                mapping.get("constraints", {}),
                "RequestPlanStep.constraints",
            ),
            tie_policy=tie_policy or "clarify",
            memory_reads=_ensure_str_list(
                mapping.get("memory_reads", []),
                "RequestPlanStep.memory_reads",
            ),
            memory_writes=_ensure_str_list(
                mapping.get("memory_writes", []),
                "RequestPlanStep.memory_writes",
            ),
            scene_fingerprint_required=_ensure_bool(
                mapping.get("scene_fingerprint_required", False),
                "RequestPlanStep.scene_fingerprint_required",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "layer": self.layer,
            "operation": self.operation,
            "required_handle": self.required_handle,
            "implementation_status": self.implementation_status,
            "inputs": dict(self.inputs),
            "outputs": list(self.outputs),
            "depends_on": list(self.depends_on),
            "constraints": dict(self.constraints),
            "tie_policy": self.tie_policy,
            "memory_reads": list(self.memory_reads),
            "memory_writes": list(self.memory_writes),
            "scene_fingerprint_required": self.scene_fingerprint_required,
        }


@dataclass
class RequestPlan:
    request_id: str
    original_utterance: str
    objective_type: str
    objective_summary: str
    steps: list[RequestPlanStep] = field(default_factory=list)
    preservation_signals: list[str] = field(default_factory=list)
    expected_response: str = "refuse"

    @classmethod
    def from_dict(cls, data: Any) -> RequestPlan:
        mapping = _ensure_mapping(data, "RequestPlan")
        objective_type = _ensure_optional_str_enum(
            mapping.get("objective_type"),
            REQUEST_OBJECTIVE_TYPES,
            "RequestPlan.objective_type",
        )
        expected_response = _ensure_optional_str_enum(
            mapping.get("expected_response", "refuse"),
            REQUEST_EXPECTED_RESPONSES,
            "RequestPlan.expected_response",
        )
        raw_steps = _ensure_list(mapping.get("steps", []), "RequestPlan.steps")
        return cls(
            request_id=_ensure_str(mapping.get("request_id"), "RequestPlan.request_id"),
            original_utterance=_ensure_str(
                mapping.get("original_utterance"),
                "RequestPlan.original_utterance",
            ),
            objective_type=objective_type or "unsupported",
            objective_summary=_ensure_str(
                mapping.get("objective_summary"),
                "RequestPlan.objective_summary",
            ),
            steps=[
                RequestPlanStep.from_dict(item)
                for item in raw_steps
            ],
            preservation_signals=_ensure_str_list(
                mapping.get("preservation_signals", []),
                "RequestPlan.preservation_signals",
            ),
            expected_response=expected_response or "refuse",
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "original_utterance": self.original_utterance,
            "objective_type": self.objective_type,
            "objective_summary": self.objective_summary,
            "steps": [step.as_dict() for step in self.steps],
            "preservation_signals": list(self.preservation_signals),
            "expected_response": self.expected_response,
        }


@dataclass
class ReadinessNode:
    step_id: str
    status: str
    layer: str
    operation: str
    required_handle: str | None = None
    reason: str = ""
    blocking_dependencies: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> ReadinessNode:
        mapping = _ensure_mapping(data, "ReadinessNode")
        status = _ensure_optional_str_enum(
            mapping.get("status"),
            READINESS_NODE_STATUSES,
            "ReadinessNode.status",
        )
        required_handle = mapping.get("required_handle")
        if required_handle is not None:
            required_handle = _ensure_str(required_handle, "ReadinessNode.required_handle")
        return cls(
            step_id=_ensure_str(mapping.get("step_id"), "ReadinessNode.step_id"),
            status=status or "unsupported",
            layer=_ensure_str(mapping.get("layer"), "ReadinessNode.layer"),
            operation=_ensure_str(mapping.get("operation"), "ReadinessNode.operation"),
            required_handle=required_handle,
            reason=_ensure_str(mapping.get("reason", ""), "ReadinessNode.reason"),
            blocking_dependencies=_ensure_str_list(
                mapping.get("blocking_dependencies", []),
                "ReadinessNode.blocking_dependencies",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "status": self.status,
            "layer": self.layer,
            "operation": self.operation,
            "required_handle": self.required_handle,
            "reason": self.reason,
            "blocking_dependencies": list(self.blocking_dependencies),
        }


@dataclass
class ReadinessGraph:
    request_id: str
    nodes: list[ReadinessNode] = field(default_factory=list)
    graph_status: str = "unsupported"
    next_action: str = "refuse"
    blocking_step_id: str | None = None
    explanation: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> ReadinessGraph:
        mapping = _ensure_mapping(data, "ReadinessGraph")
        graph_status = _ensure_optional_str_enum(
            mapping.get("graph_status", "unsupported"),
            READINESS_NODE_STATUSES,
            "ReadinessGraph.graph_status",
        )
        next_action = _ensure_optional_str_enum(
            mapping.get("next_action", "refuse"),
            READINESS_NEXT_ACTIONS,
            "ReadinessGraph.next_action",
        )
        blocking_step_id = mapping.get("blocking_step_id")
        if blocking_step_id is not None:
            blocking_step_id = _ensure_str(
                blocking_step_id,
                "ReadinessGraph.blocking_step_id",
            )
        raw_nodes = _ensure_list(mapping.get("nodes", []), "ReadinessGraph.nodes")
        return cls(
            request_id=_ensure_str(mapping.get("request_id"), "ReadinessGraph.request_id"),
            nodes=[ReadinessNode.from_dict(item) for item in raw_nodes],
            graph_status=graph_status or "unsupported",
            next_action=next_action or "refuse",
            blocking_step_id=blocking_step_id,
            explanation=_ensure_str(
                mapping.get("explanation", ""),
                "ReadinessGraph.explanation",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "nodes": [node.as_dict() for node in self.nodes],
            "graph_status": self.graph_status,
            "next_action": self.next_action,
            "blocking_step_id": self.blocking_step_id,
            "explanation": self.explanation,
        }


@dataclass
class PrimitiveSpec:
    name: str
    primitive_type: str
    layer: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    implementation_status: str = "implemented"
    safe_to_synthesize: bool = False
    runtime_binding: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: Any) -> PrimitiveSpec:
        return cls(**_ensure_primitive_spec(data, "PrimitiveSpec"))


@dataclass
class PrimitiveManifest:
    name: str
    primitives: list[PrimitiveSpec] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> PrimitiveManifest:
        manifest = _ensure_primitive_manifest(data, "PrimitiveManifest")
        return cls(
            name=manifest["name"],
            primitives=[
                PrimitiveSpec.from_dict(item)
                for item in manifest["primitives"]
            ],
        )


@dataclass
class OperatorIntent:
    intent_type: str
    canonical_instruction: str | None = None
    task_type: str | None = None
    target: dict[str, Any] | None = None
    knowledge_update: dict[str, Any] | None = None
    reference: str | None = None
    status_query: str | None = None
    claim_reference: str | None = None
    control: str | None = None
    target_selector: dict[str, Any] | None = None
    grounding_query_plan: dict[str, Any] | None = None
    capability_status: str = "executable"
    required_capabilities: list[str] = field(default_factory=list)
    clear_memory: bool = False
    confidence: float = 0.0
    reason: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> OperatorIntent:
        mapping = _ensure_mapping(data, "OperatorIntent")
        intent_type = _ensure_str(mapping.get("intent_type"), "OperatorIntent.intent_type")
        if intent_type not in OPERATOR_INTENT_TYPES:
            raise SchemaValidationError(
                "OperatorIntent.intent_type must be one of: "
                + ", ".join(OPERATOR_INTENT_TYPES)
            )

        target = _ensure_operator_target(mapping.get("target"), "OperatorIntent.target")
        knowledge_update = _ensure_operator_knowledge_update(
            mapping.get("knowledge_update"),
            "OperatorIntent.knowledge_update",
        )
        task_type = _ensure_optional_str_enum(
            mapping.get("task_type"),
            OPERATOR_TASK_TYPES,
            "OperatorIntent.task_type",
        )
        reference = _ensure_optional_str_enum(
            mapping.get("reference"),
            OPERATOR_REFERENCES,
            "OperatorIntent.reference",
        )
        status_query = _ensure_optional_str_enum(
            mapping.get("status_query"),
            OPERATOR_STATUS_QUERIES,
            "OperatorIntent.status_query",
        )
        claim_reference = _ensure_optional_str_enum(
            mapping.get("claim_reference"),
            OPERATOR_CLAIM_REFERENCES,
            "OperatorIntent.claim_reference",
        )
        control = _ensure_optional_str_enum(
            mapping.get("control"),
            OPERATOR_CONTROLS,
            "OperatorIntent.control",
        )
        capability_status = _ensure_optional_str_enum(
            mapping.get("capability_status", "executable"),
            OPERATOR_CAPABILITY_STATUSES,
            "OperatorIntent.capability_status",
        )
        target_selector = _ensure_target_selector(
            mapping.get("target_selector"),
            "OperatorIntent.target_selector",
        )
        grounding_query_plan = _ensure_grounding_query_plan(
            mapping.get("grounding_query_plan"),
            "OperatorIntent.grounding_query_plan",
        )

        canonical_instruction = mapping.get("canonical_instruction")
        if canonical_instruction is not None:
            canonical_instruction = _ensure_str(
                canonical_instruction,
                "OperatorIntent.canonical_instruction",
            )

        raw_required = mapping.get("required_capabilities")
        if raw_required is None:
            required_capabilities: list[str] = []
        elif isinstance(raw_required, list):
            required_capabilities = [str(h) for h in raw_required if h is not None]
        else:
            required_capabilities = []

        intent = cls(
            intent_type=intent_type,
            canonical_instruction=canonical_instruction,
            task_type=task_type,
            target=target,
            knowledge_update=knowledge_update,
            reference=reference,
            status_query=status_query,
            claim_reference=claim_reference,
            control=control,
            target_selector=target_selector,
            grounding_query_plan=grounding_query_plan,
            capability_status=capability_status or "executable",
            required_capabilities=required_capabilities,
            clear_memory=_ensure_bool(
                mapping.get("clear_memory"),
                "OperatorIntent.clear_memory",
            ),
            confidence=_ensure_float(mapping.get("confidence"), "OperatorIntent.confidence"),
            reason=_ensure_str(mapping.get("reason", ""), "OperatorIntent.reason"),
        )
        intent._validate_supported_shape()
        return intent

    def _validate_supported_shape(self) -> None:
        if self.intent_type == "task_instruction":
            if self.task_type != "go_to_object":
                raise SchemaValidationError("task_instruction requires task_type=go_to_object")
            has_target = (
                isinstance(self.target, dict)
                and self.target.get("color") is not None
                and self.target.get("object_type") == "door"
            )
            if (
                not has_target
                and self.reference not in OPERATOR_REFERENCES
                and self.target_selector is None
                and self.grounding_query_plan is None
            ):
                raise SchemaValidationError(
                    "task_instruction requires a supported target, reference, target_selector, or grounding_query_plan"
                )
        elif self.intent_type == "knowledge_update":
            if self.knowledge_update is None:
                raise SchemaValidationError("knowledge_update requires knowledge_update payload")
            if (
                self.knowledge_update.get("delivery_target") is None
                and self.target_selector is None
                and self.grounding_query_plan is None
            ):
                raise SchemaValidationError(
                    "selector-based knowledge_update requires target_selector or grounding_query_plan"
                )
        elif self.intent_type == "status_query":
            if self.status_query is None:
                raise SchemaValidationError("status_query requires status_query")
        elif self.intent_type == "cache_query":
            if self.status_query not in {None, "cache"}:
                raise SchemaValidationError("cache_query status_query must be cache or null")
        elif self.intent_type == "reset":
            if self.control not in {None, "reset"}:
                raise SchemaValidationError("reset control must be reset or null")
        elif self.intent_type == "quit":
            if self.control not in {None, "quit"}:
                raise SchemaValidationError("quit control must be quit or null")


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
class SceneObject:
    object_type: str
    color: str | None
    x: int
    y: int
    state: int | None = None


@dataclass
class SceneModel:
    """Structured snapshot of the last sensed scene, projected from WorldModelSample."""

    agent_x: int
    agent_y: int
    agent_dir: int
    grid_width: int
    grid_height: int
    objects: list[SceneObject]
    source: str  # "task_sense" | "idle_sense"
    env_id: str | None = None
    seed: int | None = None
    step_count: int = 0

    def find(
        self,
        *,
        object_type: str | None = None,
        color: str | None = None,
        exclude_colors: list[str] | None = None,
    ) -> list[SceneObject]:
        result: list[SceneObject] = self.objects
        if object_type is not None:
            result = [o for o in result if o.object_type == object_type]
        if color is not None:
            result = [o for o in result if o.color == color]
        if exclude_colors:
            result = [o for o in result if o.color not in exclude_colors]
        return result

    def manhattan_distance_from_agent(self, obj: SceneObject) -> int:
        return abs(obj.x - self.agent_x) + abs(obj.y - self.agent_y)

    @classmethod
    def from_world_model_sample(
        cls,
        sample: WorldModelSample,
        *,
        source: str,
        env_id: str | None = None,
        seed: int | None = None,
    ) -> SceneModel:
        agent_pose = sample.agent_pose or {}
        grid_w, grid_h = sample.grid_size if sample.grid_size else (0, 0)
        objects = [
            SceneObject(
                object_type=obj["type"],
                color=obj.get("color"),
                x=int(obj["x"]),
                y=int(obj["y"]),
                state=obj.get("state"),
            )
            for obj in (sample.grid_objects or [])
        ]
        return cls(
            agent_x=int(agent_pose.get("x", 0)),
            agent_y=int(agent_pose.get("y", 0)),
            agent_dir=int(agent_pose.get("dir", 0)),
            grid_width=int(grid_w),
            grid_height=int(grid_h),
            objects=objects,
            source=source,
            env_id=env_id,
            seed=seed,
            step_count=sample.step_count,
        )


@dataclass
class GroundedDoorEntry:
    color: str | None
    x: int
    y: int
    distance: float  # float for Euclidean and other non-integer metrics
    object_type: str = "door"
    metric: str | None = None       # e.g. "manhattan", "euclidean"
    provenance: str | None = None   # primitive handle that produced this entry

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.object_type,
            "color": self.color,
            "x": self.x,
            "y": self.y,
            "distance": self.distance,
            "metric": self.metric,
            "provenance": self.provenance,
        }


@dataclass
class StationActiveClaims:
    """Session-scoped claims produced by grounding queries.

    Tied to a SceneModel fingerprint (agent_x, agent_y, step_count).
    Cleared on reset and at task start. Never written to durable memory.
    """

    scene_fingerprint: tuple[int, int, int]  # (agent_x, agent_y, step_count)
    ranked_scene_doors: list[GroundedDoorEntry]
    last_grounded_target: GroundedDoorEntry
    last_grounded_rank: int
    last_grounding_query: dict[str, Any]

    def is_valid_for(self, scene: SceneModel) -> bool:
        return self.scene_fingerprint == (scene.agent_x, scene.agent_y, scene.step_count)

    def next_ranked(self) -> tuple[GroundedDoorEntry, int] | tuple[None, None]:
        rank = self.last_grounded_rank + 1
        if rank < len(self.ranked_scene_doors):
            return self.ranked_scene_doors[rank], rank
        return None, None

    def other_doors(self) -> list[GroundedDoorEntry]:
        t = self.last_grounded_target
        return [
            d for d in self.ranked_scene_doors
            if not (d.x == t.x and d.y == t.y)
        ]

    def compact_summary(self) -> dict[str, Any]:
        return {
            "last_grounded_target": (
                f"{self.last_grounded_target.color} door @ distance {self.last_grounded_target.distance}"
            ),
            "ranked_doors": [
                f"{d.color}@{d.distance}" for d in self.ranked_scene_doors
            ],
            "last_rank": self.last_grounded_rank,
        }


@dataclass
class ArbitrationDecision:
    """Typed arbitration decision produced by CapabilityArbitrator when a capability gap is detected.

    decision_type must be one of ARBITRATION_DECISION_TYPES.
    safe_to_execute must be False for refuse and synthesize decision types.
    """

    decision_type: str
    safe_to_execute: bool
    reason: str
    suggested_handle: str | None = None
    clarification_prompt: str | None = None
    operator_message: str = ""
    proposed_handle: str | None = None
    proposed_description: str | None = None
    proposed_condition: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.decision_type not in ARBITRATION_DECISION_TYPES:
            raise SchemaValidationError(
                f"ArbitrationDecision.decision_type must be one of: "
                + ", ".join(ARBITRATION_DECISION_TYPES)
            )
        if self.decision_type in {"refuse", "synthesize"} and self.safe_to_execute:
            raise SchemaValidationError(
                f"ArbitrationDecision with decision_type={self.decision_type} "
                "must have safe_to_execute=False"
            )


@dataclass
class ArbitrationTrace:
    """Provenance record for one arbitration event."""

    utterance: str
    intent_type: str
    required_capabilities: list[str]
    missing_handles: list[str]
    synthesizable_handles: list[str]
    decision: ArbitrationDecision

    def compact(self) -> dict[str, Any]:
        return {
            "utterance": self.utterance,
            "intent_type": self.intent_type,
            "required_capabilities": self.required_capabilities,
            "missing_handles": self.missing_handles,
            "synthesizable_handles": self.synthesizable_handles,
            "decision_type": self.decision.decision_type,
            "safe_to_execute": self.decision.safe_to_execute,
            "reason": self.decision.reason,
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


def operator_intent_json_schema() -> dict[str, Any]:
    target_schema = {
        "type": ["object", "null"],
        "properties": {
            "color": {"type": ["string", "null"], "enum": [*OPERATOR_COLORS, None]},
            "object_type": {"type": ["string", "null"], "enum": [*OPERATOR_OBJECT_TYPES, None]},
        },
        "required": ["color", "object_type"],
        "additionalProperties": False,
    }
    target_selector_schema = {
        "type": ["object", "null"],
        "properties": {
            "object_type": {"type": ["string", "null"], "enum": [*OPERATOR_OBJECT_TYPES, None]},
            "color": {"type": ["string", "null"], "enum": [*OPERATOR_COLORS, None]},
            "exclude_colors": {
                "type": "array",
                "items": {"type": "string", "enum": list(OPERATOR_COLORS)},
                "description": "Colors to exclude. Use [] when no exclusion. Supports multiple: ['purple', 'yellow'].",
            },
            "relation": {"type": ["string", "null"], "enum": [*OPERATOR_SELECTOR_RELATIONS, None]},
            "distance_metric": {"type": ["string", "null"], "enum": [*OPERATOR_DISTANCE_METRICS, None]},
            "distance_reference": {
                "type": ["string", "null"],
                "enum": [*OPERATOR_DISTANCE_REFERENCES, None],
            },
        },
        "required": [
            "object_type",
            "color",
            "exclude_colors",
            "relation",
            "distance_metric",
            "distance_reference",
        ],
        "additionalProperties": False,
    }
    grounding_query_plan_schema = {
        "type": ["object", "null"],
        "properties": {
            "object_type": {"type": ["string", "null"], "enum": [*OPERATOR_OBJECT_TYPES, None]},
            "operation": {"type": ["string", "null"], "enum": [*GROUNDING_QUERY_OPERATIONS, None]},
            "primitive_handle": {"type": ["string", "null"]},
            "metric": {"type": ["string", "null"], "enum": [*OPERATOR_DISTANCE_METRICS, None]},
            "reference": {"type": ["string", "null"], "enum": [*OPERATOR_DISTANCE_REFERENCES, None]},
            "order": {"type": ["string", "null"], "enum": [*GROUNDING_QUERY_ORDERS, None]},
            "ordinal": {"type": ["integer", "null"]},
            "color": {"type": ["string", "null"], "enum": [*OPERATOR_COLORS, None]},
            "exclude_colors": {
                "type": "array",
                "items": {"type": "string", "enum": list(OPERATOR_COLORS)},
            },
            "distance_value": {"type": ["integer", "null"]},
            "comparison": {
                "type": ["string", "null"],
                "enum": [*GROUNDING_QUERY_COMPARISONS, None],
            },
            "tie_policy": {
                "type": ["string", "null"],
                "enum": [*GROUNDING_QUERY_TIE_POLICIES, None],
            },
            "answer_fields": {
                "type": "array",
                "items": {"type": "string"},
            },
            "required_capabilities": {
                "type": "array",
                "items": {"type": "string"},
            },
            "preserved_constraints": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "object_type",
            "operation",
            "primitive_handle",
            "metric",
            "reference",
            "order",
            "ordinal",
            "color",
            "exclude_colors",
            "distance_value",
            "comparison",
            "tie_policy",
            "answer_fields",
            "required_capabilities",
            "preserved_constraints",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "intent_type": {"type": "string", "enum": list(OPERATOR_INTENT_TYPES)},
            "canonical_instruction": {"type": ["string", "null"]},
            "task_type": {"type": ["string", "null"], "enum": [*OPERATOR_TASK_TYPES, None]},
            "target": target_schema,
            "knowledge_update": {
                "type": ["object", "null"],
                "properties": {
                    "delivery_target": target_schema,
                },
                "required": ["delivery_target"],
                "additionalProperties": False,
            },
            "reference": {"type": ["string", "null"], "enum": [*OPERATOR_REFERENCES, None]},
            "status_query": {
                "type": ["string", "null"],
                "enum": [*OPERATOR_STATUS_QUERIES, None],
            },
            "claim_reference": {
                "type": ["string", "null"],
                "enum": [*OPERATOR_CLAIM_REFERENCES, None],
            },
            "control": {"type": ["string", "null"], "enum": [*OPERATOR_CONTROLS, None]},
            "target_selector": target_selector_schema,
            "grounding_query_plan": grounding_query_plan_schema,
            "capability_status": {
                "type": "string",
                "enum": list(OPERATOR_CAPABILITY_STATUSES),
            },
            "required_capabilities": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exact capability handles this intent requires. List every handle "
                    "needed, including any not yet in the registry — the station will "
                    "classify missing ones as missing_skills. No weakening: do not "
                    "substitute a broader capability for a specific one."
                ),
            },
            "clear_memory": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reason": {"type": "string"},
        },
        "required": [
            "intent_type",
            "canonical_instruction",
            "task_type",
            "target",
            "knowledge_update",
            "reference",
            "status_query",
            "claim_reference",
            "control",
            "target_selector",
            "grounding_query_plan",
            "capability_status",
            "required_capabilities",
            "clear_memory",
            "confidence",
            "reason",
        ],
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
