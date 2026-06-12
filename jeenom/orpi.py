from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from .schemas import (
    CommandResult,
    OperationalContext,
    PrimitiveSpec,
    SchemaValidationError,
    orpi_primitive_type_for,
)


ORPI_VERSION = "0"


def _obj_as_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return as_dict()
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=repr)
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return _json_safe(as_dict())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    return repr(value)


def _plan_required_handles(plan: dict[str, Any] | None) -> list[str]:
    if plan is None:
        return []
    handles: list[str] = []
    for step in plan.get("steps", []):
        if not isinstance(step, dict):
            continue
        handle = step.get("required_handle")
        if isinstance(handle, str) and handle not in handles:
            handles.append(handle)
    return handles


def _readiness_node_summary(graph: dict[str, Any] | None) -> list[dict[str, Any]]:
    if graph is None:
        return []
    nodes = graph.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    summary: list[dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        summary.append(
            {
                "step_id": node.get("step_id"),
                "status": node.get("status"),
                "layer": node.get("layer"),
                "operation": node.get("operation"),
                "required_handle": node.get("required_handle"),
                "reason": node.get("reason"),
            }
        )
    return summary


def _last_runtime_result(result_payload: dict[str, Any]) -> dict[str, Any]:
    last_result = result_payload.get("last_result")
    return dict(last_result) if isinstance(last_result, dict) else {}


def _postcondition_results(
    final_state: dict[str, Any],
    final_claims: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if "task_complete" in final_state:
        results.append(
            {
                "name": "task_complete",
                "passed": final_state.get("task_complete") is True,
                "source": "final_state",
            }
        )
    for claim_name, claim_value in sorted(final_claims.items()):
        if isinstance(claim_value, bool):
            results.append(
                {
                    "name": claim_name,
                    "passed": claim_value is True,
                    "source": "final_claims",
                }
            )
    return results


@dataclass(frozen=True)
class OrpiContract:
    """ORPI-v0 contract view over the existing JEENOM PrimitiveSpec."""

    primitive: PrimitiveSpec

    @property
    def name(self) -> str:
        return self.primitive.name

    @property
    def primitive_type(self) -> str:
        return orpi_primitive_type_for(self.primitive.primitive_type)

    @classmethod
    def from_primitive_spec(cls, primitive: PrimitiveSpec) -> "OrpiContract":
        return cls(primitive=primitive)

    def as_dict(self) -> dict[str, Any]:
        spec = self.primitive
        return {
            "name": spec.name,
            "primitive_type": self.primitive_type,
            "source_primitive_type": spec.primitive_type,
            "layer": spec.layer,
            "description": spec.description,
            "inputs": list(spec.inputs),
            "outputs": list(spec.outputs),
            "side_effects": list(spec.side_effects),
            "implementation_status": spec.implementation_status,
            "safe_to_synthesize": spec.safe_to_synthesize,
            "runtime_binding": spec.runtime_binding,
            "preconditions": list(spec.preconditions),
            "postconditions": list(spec.postconditions),
            "postcondition_primitive": spec.postcondition_primitive,
            "required_claims": list(spec.required_claims),
            "produced_claims": list(spec.produced_claims),
            "units": dict(spec.units),
            "frame_id": spec.frame_id,
            "required_frames": list(spec.required_frames),
            "safety_class": spec.safety_class,
            "authority_level": spec.authority_level,
            "failure_modes": list(spec.failure_modes),
            "validation_hooks": list(spec.validation_hooks),
            "substrate_fingerprint": spec.substrate_fingerprint,
            "mode": spec.mode,
            "cadence": spec.cadence,
            "invariant_level": spec.invariant_level,
        }


@dataclass(frozen=True)
class OrpiManifest:
    """Substrate manifest: context meaning plus published primitive contracts."""

    substrate_id: str
    substrate_fingerprint: str
    object_vocabulary: list[str]
    primitives: list[OrpiContract]
    orpi_version: str = ORPI_VERSION
    symbol_mappings: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, Any] = field(default_factory=dict)
    units: dict[str, Any] = field(default_factory=dict)
    risk_policy: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_context_and_registry(
        cls,
        context: OperationalContext,
        registry: Any,
    ) -> "OrpiManifest":
        metadata = dict(context.metadata)
        primitives = [
            OrpiContract.from_primitive_spec(spec)
            for spec in getattr(getattr(registry, "manifest", None), "primitives", [])
        ]
        units: dict[str, Any] = dict(metadata.get("units", {}))
        for contract in primitives:
            units.update(contract.as_dict().get("units", {}))
        substrate_fingerprint = metadata.get("substrate_fingerprint") or context.fingerprint()
        return cls(
            substrate_id=context.substrate_id,
            substrate_fingerprint=str(substrate_fingerprint),
            object_vocabulary=list(context.object_vocabulary),
            symbol_mappings=dict(metadata.get("symbol_mappings", {})),
            frames=dict(metadata.get("frames", {})),
            units=units,
            risk_policy=dict(metadata.get("risk_policy", {})),
            primitives=primitives,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "substrate_id": self.substrate_id,
            "substrate_fingerprint": self.substrate_fingerprint,
            "orpi_version": self.orpi_version,
            "object_vocabulary": list(self.object_vocabulary),
            "symbol_mappings": dict(self.symbol_mappings),
            "frames": dict(self.frames),
            "units": dict(self.units),
            "risk_policy": dict(self.risk_policy),
            "primitives": [contract.as_dict() for contract in self.primitives],
        }


@dataclass(frozen=True)
class LabelledEpisode:
    """Outbound ORPI supervision artifact for one operator turn."""

    intent: dict[str, Any] | None
    grounding: dict[str, Any]
    plan: dict[str, Any] | None
    authority: dict[str, Any]
    execution: dict[str, Any]
    verification: dict[str, Any]
    attribution: dict[str, Any]
    steering: dict[str, Any]

    @classmethod
    def from_command_result(cls, command_result: CommandResult) -> "LabelledEpisode":
        envelope = command_result.envelope
        command = command_result.command
        ticket = command_result.ticket
        failure = command_result.failure_outcome
        intent = _obj_as_dict(getattr(envelope, "intent", None))
        plan = _obj_as_dict(getattr(envelope, "request_plan", None))
        graph = _obj_as_dict(getattr(envelope, "readiness_graph", None))
        ticket_payload = _obj_as_dict(ticket)
        result_payload = dict(command_result.result)
        runtime_result = _last_runtime_result(result_payload)
        final_claims = runtime_result.get("final_claims", {})
        final_state = runtime_result.get("final_state", {})
        trace_events = runtime_result.get("trace_events", [])
        postcondition_results = _postcondition_results(
            final_state if isinstance(final_state, dict) else {},
            final_claims if isinstance(final_claims, dict) else {},
        )
        required_handles = _plan_required_handles(plan)
        readiness_nodes = _readiness_node_summary(graph)
        return cls(
            intent=intent,
            grounding={
                "required_handles": required_handles,
                "final_claim_keys": (
                    sorted(final_claims)
                    if isinstance(final_claims, dict)
                    else []
                ),
                "trace_event_names": [
                    event.get("event")
                    for event in trace_events
                    if isinstance(event, dict) and isinstance(event.get("event"), str)
                ],
                "readiness_graph": graph,
            },
            plan={
                "request_plan": plan,
                "readiness_graph": graph,
                "required_handles": required_handles,
                "readiness_nodes": readiness_nodes,
                "graph_status": graph.get("graph_status") if graph is not None else None,
                "next_action": graph.get("next_action") if graph is not None else None,
            },
            authority={
                "command": _obj_as_dict(command),
                "ticket": ticket_payload,
            },
            execution={
                "message": command_result.message,
                "result": result_payload,
                "failure_outcome": _obj_as_dict(failure),
                "trace_event_count": len(trace_events) if isinstance(trace_events, list) else 0,
                "runtime_llm_calls_during_render": runtime_result.get(
                    "runtime_llm_calls_during_render"
                ),
                "cache_miss_during_render": runtime_result.get(
                    "cache_miss_during_render"
                ),
            },
            verification={
                "task_complete": (
                    final_state.get("task_complete")
                    if isinstance(final_state, dict)
                    else None
                ),
                "final_state": dict(final_state) if isinstance(final_state, dict) else {},
                "final_claim_keys": (
                    sorted(final_claims)
                    if isinstance(final_claims, dict)
                    else []
                ),
                "postcondition_results": postcondition_results,
            },
            attribution={
                "failure_category": getattr(failure, "category", None),
                "blocking_claim_handle": getattr(failure, "blocking_claim_handle", None),
                "readiness_status": graph.get("graph_status") if graph is not None else None,
            },
            steering={
                "pending_context": dict(getattr(envelope, "pending_context", {}) or {}),
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return _json_safe({
            "intent": self.intent,
            "grounding": dict(self.grounding),
            "plan": self.plan,
            "authority": dict(self.authority),
            "execution": dict(self.execution),
            "verification": dict(self.verification),
            "attribution": dict(self.attribution),
            "steering": dict(self.steering),
        })


def assert_no_deliberative_meta_plan_references(
    plan: Any,
    registry: Any,
) -> None:
    """Reject compiled plans that reference deliberative meta-primitives."""

    if plan is None:
        return
    for step in getattr(plan, "steps", []):
        handle = getattr(step, "required_handle", None)
        if handle is None:
            continue
        primitive = registry.lookup(handle) if hasattr(registry, "lookup") else None
        if primitive is None:
            continue
        if (
            orpi_primitive_type_for(primitive.primitive_type) == "meta"
            and primitive.mode == "deliberative"
        ):
            raise SchemaValidationError(
                f"Compiled plan references deliberative meta-primitive: {handle}"
            )
