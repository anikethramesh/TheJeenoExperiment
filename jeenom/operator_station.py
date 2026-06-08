from __future__ import annotations

import argparse
import inspect
import math
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .capability_arbitrator import ArbitratorBackend, build_arbitrator
from .capability_matcher import CapabilityMatchResult, CapabilityMatcher, default_matcher
from .capability_registry import CapabilityRegistry
from .command_authority import CommandAuthority
from .primitive_synthesizer import SynthesizerBackend, build_synthesizer
from .primitive_validator import PrimitiveValidator, default_validator
from .readiness_graph import evaluate_request_plan
from .request_planner import build_request_plan
from .intent_verifier import IntentVerificationResult, IntentVerifier
from .cortex import Cortex
from .llm_compiler import CompilerBackend, SmokeTestCompiler, build_compiler, canonical_task_params
from .memory import OperationalMemory
from .mission_cortex import (
    InlineMetricMissionRequest,
    MissionCortex,
    parse_inline_metric_request,
)
from .minigrid_runtime_package import (
    build_minigrid_runtime_package,
    default_minigrid_domain_helper,
)
from .plan_cache import PlanCache
from .planning_semantics import PlanningSemantics
from .knowledge_base import KnowledgeBase, NamedConcept
from .mismatch import MismatchDetector, OperationalMismatch, default_detector
from .plan_reuse import PlanReuseCache, ReuseVerdict, plan_semantic_key
from .repair_loop import RepairEvent, RepairLoop
from .representation import RepresentationStore
from .runtime_package import RuntimePackage
from .semantic_normalizer import infer_direction_from_utterance, normalize_distance_ordinal
from .side_effect_authority import SideEffectAuthority
from .substrate_adapter import SubstrateAdapter
from .turn_orchestrator import TurnOrchestrator
from .schemas import (
    ApprovedCommand,
    ArbitrationTrace,
    CommandResult,
    CorticalEnvelope,
    EnvironmentIdentity,
    ExecutionTicket,
    GroundedDoorEntry,
    MemoryUpdate,
    MemoryWriteTicket,
    MissionExecutionPlan,
    OperatorIntent,
    PrimitiveDefinitionRequest,
    ProcedureRecipe,
    RawMotorTicket,
    ReadinessGraph,
    RequestPlanStep,
    RequestPlan,
    SceneModel,
    SceneObject,
    SchemaValidationError,
    StationActiveClaims,
    TargetSelector,
    TaskRequest,
)


_DEFAULT_DOMAIN_HELPER = default_minigrid_domain_helper()


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


_ACTION_LEAK_TERMS = (
    "move",
    "moves",
    "moving",
    "turn",
    "turns",
    "pickup",
    "pick up",
    "grab",
    "toggle",
    "open",
    "unlock",
    "navigate",
    "go forward",
    "env step",
    "env.step",
    "controller",
    "motor",
    "actuate",
)


def _scene_object_to_dict(obj: SceneObject) -> dict[str, Any]:
    return {"type": obj.object_type, "color": obj.color, "x": obj.x, "y": obj.y}


def _normalize_utterance(utterance: str) -> str:
    text = " ".join(utterance.lower().strip().split())
    text = re.sub(r"[?!]+$", "", text)
    text = re.sub(r"[.,;:]+", " ", text)
    text = " ".join(text.split())
    while True:
        stripped = re.sub(
            r"^(?:ok|okay|alright|right|so|well|now|first|then)\s+",
            "",
            text,
        )
        if stripped == text:
            return text
        text = stripped


def _looks_like_bare_label(normalized: str) -> bool:
    """True for single-word utterances that look like an operator concept label.

    Must be long enough to be distinctive (>=4 chars), purely alphabetic, and
    not a common English word that the LLM could meaningfully interpret.
    """
    _common_words = {
        "quit", "exit", "bye", "cancel", "reset", "help", "status",
        "cache", "done", "back", "next", "stop", "show", "list",
        "what", "which", "where", "when", "find", "goto", "move",
        "go", "get", "run", "yes", "okay", "sure", "red", "blue",
        "green", "yellow", "purple", "grey", "gray",
    }
    return (
        " " not in normalized
        and len(normalized) >= 4
        and normalized.isalpha()
        and normalized not in _common_words
    )


def _normalize_metric_name(name: str) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name.strip())
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def _metric_dependency_handle(metric: str) -> str:
    return f"grounding.all_doors.ranked.{metric}.agent"


def _metric_dependencies(formula: str) -> list[str]:
    normalized = _normalize_utterance(formula)
    dependencies: list[str] = []
    for metric in ("euclidean", "manhattan"):
        if re.search(rf"\b{metric}\b", normalized) and metric not in dependencies:
            dependencies.append(metric)
    if (
        not dependencies
        and re.search(r"\bboth\s+(?:distance\s+)?metrics?\b", normalized)
    ):
        dependencies.extend(["euclidean", "manhattan"])
    return dependencies


def _parse_metric_expression(formula: str) -> dict[str, Any] | None:
    normalized = _normalize_utterance(formula)
    dependencies = _metric_dependencies(formula)
    if any(term in normalized for term in _ACTION_LEAK_TERMS):
        return {
            "op": "unsafe",
            "reason": "Metric definitions must be query-only and cannot include actuation.",
            "metrics": dependencies,
        }
    if not dependencies:
        return None

    number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", normalized)
    constant = float(number_match.group(1)) if number_match else None

    if len(dependencies) >= 2 and (
        "sum" in normalized
        or "total" in normalized
        or "combined" in normalized
        or "plus" in normalized
        or "+" in formula
    ):
        return {"op": "sum", "metrics": dependencies}
    if "minimum" in normalized or "min(" in normalized or "min of" in normalized:
        return {"op": "min", "metrics": dependencies}
    if "maximum" in normalized or "max(" in normalized or "max of" in normalized:
        return {"op": "max", "metrics": dependencies}
    if "mod" in normalized or "modulo" in normalized:
        if constant is None:
            return None
        return {"op": "mod", "metric": dependencies[0], "constant": constant}
    if "abs" in normalized and "-" in formula and len(dependencies) >= 2:
        expression: dict[str, Any] = {
            "op": "abs_diff",
            "metrics": dependencies[:2],
        }
        if constant is not None and ("+" in formula or " plus " in normalized):
            expression["op"] = "abs_diff_plus"
            expression["constant"] = constant
        return expression
    if " plus " in normalized or "+" in formula:
        if constant is None:
            return None
        return {"op": "add", "metric": dependencies[0], "constant": constant}
    if " minus " in normalized or "-" in formula:
        if constant is None:
            return None
        return {"op": "subtract", "metric": dependencies[0], "constant": constant}
    if len(dependencies) == 1:
        return {"op": "alias", "metric": dependencies[0]}
    return None


def _metric_expression_name(expression: dict[str, Any]) -> str:
    op = str(expression.get("op") or "metric")
    if op in {"sum", "min", "max"}:
        metrics = [
            _normalize_metric_name(str(metric))
            for metric in expression.get("metrics", [])
        ]
        return _normalize_metric_name("_".join([op, *metrics]))
    if op in {"alias", "mod", "add", "subtract"}:
        metric = _normalize_metric_name(str(expression.get("metric") or "metric"))
        suffix = ""
        if expression.get("constant") is not None:
            suffix = "_" + str(expression["constant"]).replace(".", "_")
        return _normalize_metric_name(f"{op}_{metric}{suffix}")
    if op in {"abs_diff", "abs_diff_plus"}:
        metrics = [
            _normalize_metric_name(str(metric))
            for metric in expression.get("metrics", [])
        ]
        return _normalize_metric_name("_".join([op, *metrics]))
    return _normalize_metric_name(op)


def _parse_primitive_definition_request(text: str) -> PrimitiveDefinitionRequest | str | None:
    patterns = [
        re.compile(
            r"^(?:please\s+)?(?:define|make|create|synthesize)\s+"
            r"(?:a\s+|an\s+)?(?:new\s+)?(?:distance\s+)?metric\s+"
            r"(?:(?:called|named)\s+)?(?P<name>[A-Za-z][A-Za-z0-9_]*)\s*=\s*"
            r"(?P<formula>.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:please\s+)?(?:define|make|create|synthesize)\s+"
            r"(?:a\s+|an\s+)?(?:new\s+)?(?:distance\s+)?metric\s+"
            r"(?:called|named)\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)\s+"
            r"(?:as|to be|which is|that is|that)\s+(?P<formula>.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:please\s+)?(?:define|make|create|synthesize)\s+"
            r"(?:a\s+|an\s+)?(?:new\s+)?distance\s+metric\s+"
            r"(?:which is|that is|as)\s+(?P<formula>.+?)\s+"
            r"(?:and\s+)?call\s+it\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)$",
            re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.match(text.strip())
        if not match:
            continue
        name = match.group("name")
        formula = match.group("formula").strip()
        normalized_name = _normalize_metric_name(name)
        expression = _parse_metric_expression(formula)
        if expression is None:
            return (
                "I could not parse that metric formula. Use a query-only formula "
                "such as min(euclidean, manhattan), euclidean mod 5, or manhattan plus 3."
            )
        if expression.get("op") == "unsafe":
            return (
                "REFUSE\n"
                "Metric definitions must be query-only. I will not build a metric "
                "that contains actuation, movement, controller, or motor side effects."
            )
        dependencies = list(dict.fromkeys(_metric_dependencies(formula)))
        dependency_handles = [_metric_dependency_handle(metric) for metric in dependencies]
        return PrimitiveDefinitionRequest(
            definition_type="distance_metric",
            name=name,
            normalized_name=normalized_name,
            expression=expression,
            dependencies=dependencies,
            dependency_handles=dependency_handles,
            proposed_handle=_metric_dependency_handle(normalized_name),
            safety_class="query",
            authority_level="operator",
            provenance={
                "operator_utterance": text,
                "formula": formula,
            },
        )
    return None


def _parse_metric_query(text: str) -> str | None:
    raw = text.strip()
    patterns = [
        re.compile(
            r"\b(?:rank|list|show)\s+(?:all\s+)?(?:the\s+)?doors\s+by\s+"
            r"(?P<metric>[A-Za-z][A-Za-z0-9_]*)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:what\s+is|whats|what's|show|list)\s+(?:the\s+)?"
            r"(?P<metric>[A-Za-z][A-Za-z0-9_]*)\s+"
            r"(?:distance\s+)?(?:to|for|of)\s+all\s+(?:the\s+)?doors\b",
            re.IGNORECASE,
        ),
    ]
    stopwords = {
        "distance",
        "distances",
        "closest",
        "farthest",
        "furthest",
        "door",
        "doors",
        "all",
        "the",
        "manhattan",
        "euclidean",
    }
    for pattern in patterns:
        match = pattern.search(raw)
        if not match:
            continue
        metric = match.group("metric")
        normalized = _normalize_metric_name(metric)
        if normalized and normalized not in stopwords:
            return normalized
    return None


def _looks_like_question(normalized_utterance: str) -> bool:
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


def _approved(
    command_type: str,
    utterance: str = "",
    message: str | None = None,
    payload: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ApprovedCommand:
    p = payload if payload is not None else ({"message": message} if message is not None else {})
    return ApprovedCommand(command_type=command_type, utterance=utterance, payload=p, **kwargs)


def classify_utterance(utterance: str) -> ApprovedCommand:
    text = utterance.strip()
    normalized = _normalize_utterance(text)

    if normalized in {"quit", "exit", "bye"}:
        return ApprovedCommand(command_type="quit", utterance=text)

    if normalized in {"cancel", "never mind", "nevermind"}:
        return ApprovedCommand(command_type="cancel", utterance=text)

    if normalized in {"reset", "reset episode"}:
        return ApprovedCommand(command_type="reset", utterance=text, payload={"clear_memory": False})
    if normalized in {"clear memory", "forget memory", "forget everything", "reset memory"}:
        return ApprovedCommand(command_type="reset", utterance=text, payload={"clear_memory": True})

    if normalized in {"show cache", "cache", "cache status", "what is cached?"}:
        return ApprovedCommand(command_type="cache_query", utterance=text)

    if normalized in {
        "what do you see?",
        "what do you see",
        "what can you see?",
        "what can you see",
        "what doors are available?",
        "what doors are available",
        "which doors are available?",
        "which doors are available",
        "which doors are visible?",
        "which doors are visible",
        "which doors do you see?",
        "which doors do you see",
        "describe the scene",
        "describe what you see",
        "show scene",
    }:
        return ApprovedCommand(command_type="status_query", utterance=text, payload={"query": "scene"})

    if normalized in {
        "what was the last target?",
        "what was the last target",
        "what was the previous target?",
        "what was the previous target",
        "previous target",
        "last target",
    }:
        return ApprovedCommand(command_type="status_query", utterance=text, payload={"query": "last_target"})

    if normalized in {
        "help",
        "what can you do?",
        "what can you do",
        "status",
        "what do you know?",
        "what do you know",
        "what happened last run?",
        "what happened last run",
        "last run",
    }:
        if normalized in {"help", "what can you do?", "what can you do"}:
            query = "help"
        elif "last run" in normalized or "happened" in normalized:
            query = "last_run"
        else:
            query = "status"
        return ApprovedCommand(command_type="status_query", utterance=text, payload={"query": query})

    if normalized in {
        "go there again",
        "go to the same door",
        "go to same door",
        "repeat the last task",
        "repeat the previous task",
    }:
        return ApprovedCommand(command_type="task_instruction", utterance=text)

    if "delivery target" in normalized and re.search(
        r"^(go to|reach|find|get to|head to|navigate to) (the )?delivery target$",
        normalized,
    ):
        return ApprovedCommand(command_type="task_instruction", utterance=text)

    primitive_definition = _parse_primitive_definition_request(text)
    if isinstance(primitive_definition, PrimitiveDefinitionRequest):
        return _approved(
            "primitive_definition",
            text,
            payload={"definition": primitive_definition.as_dict()},
        )
    if isinstance(primitive_definition, str):
        return _approved("unsupported", text, primitive_definition)

    inline_metric = parse_inline_metric_request(text)
    if isinstance(inline_metric, InlineMetricMissionRequest):
        return _approved(
            "primitive_definition",
            text,
            payload={
                "definition": inline_metric.primitive_definition.as_dict(),
                "mission_request": inline_metric,
            },
        )
    if isinstance(inline_metric, str):
        return _approved("unsupported", text, inline_metric)

    custom_metric = _parse_metric_query(text)
    if custom_metric is not None:
        return _approved("metric_query", text, payload={"metric": custom_metric})

    # Concept teach — explicit syntax: "remember X means Y" / "define X as Y"
    _concept_teach_m = re.match(
        r"^(?:remember|teach|define)\s+(.+?)\s+(?:means|as|is shorthand for)\s+(.+)$",
        normalized,
    )
    if _concept_teach_m:
        cname = _concept_teach_m.group(1).strip().strip("'\"")
        # Re-match against raw text (commas intact) to preserve CSV procedure sequences.
        # normalized strips [.,;:] so "bingo, scout, bingo" becomes "bingo scout bingo".
        _raw_m = re.match(
            r"^(?:remember|teach|define)\s+(.+?)\s+(?:means|as|is shorthand for)\s+(.+)$",
            text,
            re.IGNORECASE,
        )
        cutterance = (
            _raw_m.group(2).strip().strip("'\"")
            if _raw_m is not None
            else _concept_teach_m.group(2).strip().strip("'\"")
        )
        return _approved("concept_teach", text, payload={"name": cname, "utterance": cutterance})


    # Natural-language concept teach patterns ("when I say X, Y" / "X means Y") are
    # intentionally omitted here — they route through the LLM / SmokeTestCompiler which
    # now emit concept_teach as a first-class intent type (Phase 8.4.5).

    # Concept forget: "forget concept X" / "forget X"
    _concept_forget_m = re.match(r"^forget(?:\s+concept)?\s+(.+)$", normalized)
    if _concept_forget_m:
        cname = _concept_forget_m.group(1).strip().strip("'\"")
        return _approved("concept_forget", text, payload={"name": cname})


    # Concept list
    if normalized in {
        "what concepts do you know",
        "list concepts",
        "show concepts",
        "concepts",
        "what have you learned",
        "what do you remember",
    }:
        return ApprovedCommand(command_type="status_query", utterance=text, payload={"query": "concepts"})

    target_fact = _parse_target_fact(normalized)
    if target_fact is not None:
        return ApprovedCommand(command_type="knowledge_update", utterance=text, payload=target_fact)

    if _parse_exact_go_to_object_utterance(normalized) is not None:
        return ApprovedCommand(command_type="task_instruction", utterance=text)

    return ApprovedCommand(command_type="unresolved", utterance=text)


def _parse_target_fact(normalized: str) -> dict[str, str] | None:
    return _DEFAULT_DOMAIN_HELPER.parse_target_fact(normalized)


def _canonicalize_task_instruction(utterance: str) -> str:
    return _DEFAULT_DOMAIN_HELPER.canonicalize_task_instruction(utterance)


def _parse_go_to_object_utterance(utterance: str) -> dict[str, str] | None:
    return _DEFAULT_DOMAIN_HELPER.parse_go_to_object_utterance(utterance)


def _parse_exact_go_to_object_utterance(utterance: str) -> dict[str, str] | None:
    return _DEFAULT_DOMAIN_HELPER.parse_exact_go_to_object_utterance(utterance)


class OperatorStationSession:
    def __init__(
        self,
        *,
        env_id: str = "MiniGrid-GoToDoor-8x8-v0",
        seed: int = 42,
        compiler_name: str = "llm",
        compiler: CompilerBackend | None = None,
        memory_root: Path | None = None,
        render_mode: str = "human",
        max_loops: int = 128,
        use_cache: bool = True,
        verbose: bool = False,
        request_plan_reuse_cache: PlanReuseCache | None = None,
        runtime_package: RuntimePackage | None = None,
    ) -> None:
        self.env_id = env_id
        self.seed = seed
        self.compiler_name = compiler_name
        self.compiler = compiler or build_compiler(compiler_name)
        self.memory = OperationalMemory(root=memory_root)
        self.plan_cache = PlanCache(enabled=use_cache)
        self.render_mode = render_mode
        self.max_loops = max_loops
        self.verbose = verbose
        self.last_result: dict[str, Any] | None = None
        self.runtime_package = runtime_package or build_minigrid_runtime_package(
            env_id=self.env_id,
            render_mode=self.render_mode,
        )
        self.operational_context = self.runtime_package.operational_context
        self.context_fingerprint = self.operational_context.fingerprint()
        self.domain_helper = self.runtime_package.domain_helper
        self.planning_semantics = PlanningSemantics(self.operational_context)
        self.intent_verifier = IntentVerifier(planning_semantics=self.planning_semantics)
        self.turn_orchestrator = TurnOrchestrator(
            classify_utterance=classify_utterance,
            normalize_utterance=_normalize_utterance,
            looks_like_bare_label=_looks_like_bare_label,
        )
        self.substrate: SubstrateAdapter = self.runtime_package.substrate
        self.capability_registry = self.runtime_package.resolve_capability_registry()
        self.mission_cortex = MissionCortex(
            planning_semantics=self.planning_semantics,
            registry=self.capability_registry,
        )
        # KnowledgeBase persists alongside knowledge.yaml in memory_dir.
        self.knowledge_base = KnowledgeBase(
            storage_path=self.memory.memory_dir / "knowledge_base.json"
        )
        self.representation = RepresentationStore(
            memory=self.memory,
            knowledge_base=self.knowledge_base,
        )
        self.command_authority = CommandAuthority(station_name="OperatorStationSession")
        self.side_effect_authority = SideEffectAuthority(source_name="OperatorStationSession")
        self.cortex = Cortex(self.memory, self.compiler, plan_cache=self.plan_cache)
        self.sense = self.substrate.create_sense(
            self.memory,
            self.compiler,
            self.plan_cache,
        )
        self.spine = self.substrate.create_spine(
            self.memory,
            self.compiler,
            self.plan_cache,
        )
        self.prewarm_compiler = SmokeTestCompiler()
        self.prewarm_cortex = Cortex(
            self.memory,
            self.prewarm_compiler,
            plan_cache=self.plan_cache,
        )
        self.prewarm_sense = self.substrate.create_sense(
            self.memory,
            self.prewarm_compiler,
            self.plan_cache,
        )
        self.prewarm_spine = self.substrate.create_spine(
            self.memory,
            self.prewarm_compiler,
            self.plan_cache,
        )
        self.startup_prewarm_summary: dict[str, Any] | None = None
        self.pending_clarification: PendingClarification | None = None
        self.pending_synthesis_proposal: PendingSynthesisProposal | None = None
        self.pending_primitive_definition: PendingPrimitiveDefinition | None = None
        self.last_mission_execution_plan: MissionExecutionPlan | None = None
        self.last_execution_ticket: ExecutionTicket | None = None
        self.last_memory_write_ticket: MemoryWriteTicket | None = None
        self.last_raw_motor_ticket: RawMotorTicket | None = None
        self.last_operator_intent: OperatorIntent | None = None
        self.last_cortical_envelope: CorticalEnvelope | None = None
        self.last_approved_command: ApprovedCommand | None = None
        self.last_command_result: CommandResult | None = None
        self.current_environment_identity: EnvironmentIdentity | None = None
        self.last_environment_invalidation_reason: str | None = None
        self.last_request_plan: RequestPlan | None = None
        self.last_readiness_graph: ReadinessGraph | None = None
        self.request_plan_reuse_cache: PlanReuseCache = (
            request_plan_reuse_cache if request_plan_reuse_cache is not None else PlanReuseCache()
        )
        self.last_plan_reuse_verdict: ReuseVerdict | None = None
        self.last_operational_mismatches: list[OperationalMismatch] = []
        self.last_repair_events: list[RepairEvent] = []
        self.arbitrator: ArbitratorBackend = build_arbitrator(compiler_name)
        self.last_arbitration_trace: ArbitrationTrace | None = None
        self.synthesizer: SynthesizerBackend = build_synthesizer(compiler_name)
        self.validator: PrimitiveValidator = default_validator

    @property
    def active_claims(self) -> StationActiveClaims | None:
        return self.representation.get_active_claims()

    @active_claims.setter
    def active_claims(self, claims: StationActiveClaims | None) -> None:
        if claims is None:
            self.representation.clear_active_claims(reason="station cleared active claims")
            return
        self.representation.set_active_claims(claims)

    @property
    def preview_adapter(self) -> Any:
        return getattr(self.substrate, "preview_adapter", None)

    @property
    def task_adapter(self) -> Any:
        return getattr(self.substrate, "task_adapter", None)

    def log(self, message: str) -> None:
        if self.verbose:
            print(f"[station] {message}", flush=True)

    def startup(self) -> str:
        self.log(
            f"initializing env={self.env_id} seed={self.seed} "
            f"compiler={self.compiler.active_backend} render={self.render_mode}"
        )
        self.open_preview()
        self.startup_prewarm_summary = self.prewarm_known_task_family()
        if self.startup_prewarm_summary is not None:
            self.log(
                "startup prewarm complete: "
                f"cache_entries={self.startup_prewarm_summary['cache_entries']}"
            )
        self.log("operator station ready")
        return "READY"

    def prewarm_known_task_family(self) -> dict[str, Any] | None:
        if not self.plan_cache.enabled:
            self.log("cache disabled; skipping startup prewarm")
            return None

        instruction = self._startup_warmup_instruction()
        self.log("warming up known go_to_object task family")
        self.log(f"startup warmup target: {instruction}")
        self._pump_render_window()
        task = self.compose_known_task(instruction)
        procedure = self.compose_known_procedure(task)
        self.prewarm_cortex.onboard_task(task, procedure)
        summary = self.substrate.prewarm_templates(
            task_request=task,
            procedure_recipe=procedure,
            cortex=self.prewarm_cortex,
            sense=self.prewarm_sense,
            spine=self.prewarm_spine,
            plan_cache=self.plan_cache,
            progress_callback=self._progress_callback,
        )
        return {
            "compiled_templates": summary["compiled_templates"],
            "cache_entries": len(self.plan_cache.entries),
        }

    def _startup_warmup_instruction(self) -> str:
        delivery_target = self.memory.knowledge.get("delivery_target")
        if isinstance(delivery_target, dict):
            color = delivery_target.get("color")
            object_type = delivery_target.get("object_type")
            if color in self.domain_helper.supported_colors and object_type == "door":
                return f"go to the {self.domain_helper.normalize_color(color)} door"

        target_color = self.memory.knowledge.get("target_color")
        target_type = self.memory.knowledge.get("target_type")
        if target_color in self.domain_helper.supported_colors and target_type == "door":
            return f"go to the {self.domain_helper.normalize_color(target_color)} door"

        return "go to the red door"

    def _record_command_result(
        self,
        utterance: str,
        message: str,
        *,
        plan: RequestPlan | None,
        graph: ReadinessGraph | None,
    ) -> CommandResult:
        command_result = self.command_authority.record_result(
            utterance,
            message,
            intent=self.last_operator_intent,
            plan=plan,
            graph=graph,
            tickets=(
                self.last_execution_ticket,
                self.last_memory_write_ticket,
                self.last_raw_motor_ticket,
            ),
            compiler_name=self.compiler_name,
            pending_context={
                "clarification": self.pending_clarification.clarification_type
                if self.pending_clarification is not None
                else None,
                "synthesis": self.pending_synthesis_proposal.handle
                if self.pending_synthesis_proposal is not None
                else None,
                "primitive_definition": (
                    self.pending_primitive_definition.request.proposed_handle
                    if self.pending_primitive_definition is not None
                    else None
                ),
            },
            last_result=self.last_result,
        )
        self.last_cortical_envelope = command_result.envelope
        self.last_approved_command = command_result.command
        self.last_command_result = command_result
        return command_result

    def _pending_clarification_trace(
        self,
        utterance: str,
        clarification_type: str,
    ) -> dict[str, Any]:
        return self.command_authority.pending_clarification_trace(
            utterance,
            clarification_type,
            intent=self.last_operator_intent,
            plan=self.last_request_plan,
            graph=self.last_readiness_graph,
            compiler_name=self.compiler_name,
        )

    def handle_utterance(self, utterance: str) -> CommandResult:
        self.last_cortical_envelope = None
        self.last_approved_command = None
        self.last_command_result = None
        previous_plan = self.last_request_plan
        previous_graph = self.last_readiness_graph
        message = self.turn_orchestrator.handle_utterance_text(self, utterance)
        plan = self.last_request_plan if self.last_request_plan is not previous_plan else None
        graph = (
            self.last_readiness_graph
            if self.last_readiness_graph is not previous_graph
            else None
        )
        return self._record_command_result(
            utterance,
            str(message),
            plan=plan,
            graph=graph,
        )

    def command_from_llm_intent(
        self,
        utterance: str,
        *,
        pending_proposal: dict[str, Any] | None = None,
    ) -> ApprovedCommand:
        intent = self.compiler.compile_operator_intent(
            utterance,
            memory=self.memory,
            scene_summary=None,
            capability_manifest=self.capability_registry.compact_summary(),
            active_claims_summary=(
                self.active_claims.compact_summary() if self.active_claims is not None else None
            ),
            pending_proposal=pending_proposal,
        )
        self.log(
            "operator intent: "
            f"type={intent.intent_type} confidence={intent.confidence:.2f} "
            f"reason={intent.reason}"
        )
        return self.command_from_operator_intent(intent, utterance)

    def command_from_operator_intent(
        self,
        intent: OperatorIntent,
        utterance: str,
    ) -> ApprovedCommand:
        # Reset per-turn so stale values from previous turns are never observed.
        self.last_readiness_graph = None
        self.last_repair_events = []
        self.last_operational_mismatches = []

        # ── Proactive Intent Signal Verification (Phase 7.595) ───────────────
        # Runs unconditionally on ALL LLM outputs before any routing.
        # Blueprint Rule 9: deterministic gate between compiler output and
        # CapabilityMatcher — regardless of what the LLM declared.
        intent, verif_result = self.intent_verifier.enrich(utterance, intent)
        promoted_intent = self._promote_verified_query_intent(
            utterance,
            intent,
            verif_result,
        )
        if promoted_intent is not intent:
            self.log(
                "intent verifier promoted unresolved utterance to "
                f"{promoted_intent.intent_type}"
            )
            intent = promoted_intent
        self.last_operator_intent = intent
        if verif_result.injected_handles:
            self.log(f"intent verifier injected: {verif_result.summary()}")
        if verif_result.inversion_detected:
            self.log(f"intent verifier blocked inversion: {verif_result.inversion_reason}")
            return _approved("ambiguous", utterance, (
                    "Semantic inversion detected in compiled plan: "
                    f"{verif_result.inversion_reason} "
                    "Please rephrase."
                ))


        request_plan_recorded = False
        if intent.grounding_query_plan is not None:
            self._record_request_plan(utterance, intent)
            request_plan_recorded = True
            plan_command = self._command_from_grounding_query_plan(utterance, intent)
            if plan_command is not None:
                return plan_command

        if not request_plan_recorded:
            self._record_request_plan(utterance, intent)

        # ── Intent Readiness Requirement Matching (Phase 7.59) ────────────────
        # Runs every turn, deterministically. Matcher verdict overrides LLM's
        # capability_status when required_capabilities are declared. No weakening.
        cap_match = default_matcher.match(intent, self.capability_registry)
        composition_command = self._try_compose_grounding_result(
            utterance,
            intent,
            cap_match,
            verif_result,
        )
        if composition_command is not None:
            return composition_command
        if cap_match.verdict in {"missing_skills", "synthesizable", "unsupported"}:
            return self._arbitrate_gap(utterance, intent, cap_match)

        if intent.target_selector is not None and intent.capability_status in {
            "needs_clarification",
            "missing_skills",
            "synthesizable",
        }:
            if intent.capability_status == "needs_clarification":
                grounded = self.ground_target_selector(intent.target_selector)
                clarification = self.maybe_start_selector_clarification(
                    utterance=utterance,
                    resume_kind=(
                        "task_instruction"
                        if intent.intent_type == "task_instruction"
                        else "knowledge_update"
                        if intent.intent_type == "knowledge_update"
                        else "ground_target_query"
                    ),
                    grounded=grounded,
                )
                if clarification is not None:
                    return clarification
            readiness_command = self.command_from_selector_readiness(
                intent.target_selector,
                utterance,
            )
            if readiness_command is not None:
                return readiness_command

        if intent.intent_type in {"accept_proposal", "reject_proposal"}:
            return ApprovedCommand(command_type=intent.intent_type, utterance=utterance)

        if intent.intent_type == "concept_teach":
            name = (intent.concept_name or "").strip()
            expansion = (intent.concept_utterance or "").strip()
            if not name or not expansion:
                return _approved("clarification", utterance, "Please specify both a name and an instruction for the concept.")

            return _approved("concept_teach", utterance, payload={"name": name, "utterance": expansion})


        if intent.intent_type == "concept_recall":
            name = (intent.concept_name or "").strip()
            if not name:
                return _approved("clarification", utterance, "Please specify the concept name to recall.")

            concept = self.knowledge_base.recall(name)
            if concept is None:
                return ApprovedCommand(
                    kind="clarification",
                    utterance=utterance,
                    payload={
                        "message": (
                            f"I don't know a concept named '{name}'.\n"
                            f"To teach it, say: remember {name} means <full instruction>"
                        )
                    },
                )
            # Procedure-type concepts route to _run_procedure, not task_instruction
            if concept.concept_type == "procedure":
                return _approved("procedure_execute", utterance, payload={"steps": list(concept.steps)})

            expanded = concept.utterance
            self.log(f"concept_recall: '{name}' → '{expanded}'")
            return ApprovedCommand(command_type="task_instruction", utterance=expanded)

        if intent.intent_type == "procedure_recall":
            steps = list(intent.concept_steps or [])
            if not steps:
                return _approved("clarification", utterance, "Please specify the concept names to execute in sequence.")

            self.log(f"procedure_recall: steps={steps}")
            return _approved("procedure_execute", utterance, payload={"steps": steps})


        if intent.intent_type == "sequence_instruction":
            usteps = list(intent.utterance_steps or [])
            if not usteps:
                return _approved("clarification", utterance, "Please specify the task steps to execute in sequence.")

            self.log(f"sequence_instruction: steps={usteps}")
            return _approved("sequence_execute", utterance, payload={"steps": usteps})


        if intent.intent_type == "motor_command":
            action = (intent.action_name or "").strip()
            count = max(1, intent.repeat_count or 1)
            if not action:
                return _approved("clarification", utterance, "Please specify which motor action to perform.")

            self.log(f"motor_command: action={action} count={count}")
            return _approved("motor_execute", utterance, payload={"action": action, "count": count})


        if intent.intent_type == "motor_sequence":
            sequence: list[dict[str, Any]] = []
            for step in (intent.utterance_steps or []):
                parts = step.split(":", 1)
                if len(parts) != 2:
                    continue
                action_name, count_str = parts
                try:
                    action_count = max(1, int(count_str))
                except ValueError:
                    continue
                sequence.append({"action": action_name, "count": action_count})
            if len(sequence) < 2:
                return _approved("clarification", utterance, "Could not parse motor sequence steps.")

            self.log(f"motor_sequence: {len(sequence)} actions")
            return _approved("motor_sequence_execute", utterance, payload={"sequence": sequence})


        if intent.intent_type == "mission_contract":
            steps = list(intent.mission_steps or [])
            if len(steps) < 2:
                return _approved("clarification", utterance, "A mission requires at least 2 task steps.")

            self.log(f"mission_contract: {len(steps)} steps")
            return _approved("mission_execute", utterance, payload={"steps": steps})


        if intent.intent_type == "claim_reference":
            if intent.claim_reference == "threshold_filter":
                # Route through capability matching → arbitration → synthesis pipeline.
                # The plan carries threshold, comparison, and metric for execution.
                if cap_match.verdict in {"synthesizable", "missing_skills"}:
                    return self._arbitrate_gap(utterance, intent, cap_match)
                # Already registered — execute directly.
                return self._dispatch_claims_filter(utterance, intent, cap_match)
            return _approved("claim_reference", utterance, payload={"ref_type": intent.claim_reference or ""})


        if intent.intent_type in {"unsupported", "ambiguous"}:
            # Pure semantic/schema failure (no declared capabilities) → plain error.
            # Reserve arbitration for intents where the LLM identified a capability but
            # the registry says it is missing or synthesizable.
            if not intent.required_capabilities and not cap_match.missing:
                reason = intent.reason or "I could not understand that request."
                if intent.intent_type == "unsupported" and "unsupported" in reason.lower():
                    return ApprovedCommand(
                        kind="unsupported",
                        utterance=utterance,
                        payload={
                            "message": (
                                "I cannot safely execute that capability yet. "
                                f"{reason}"
                            )
                        },
                    )
                reason = intent.reason or "I could not understand that request."
                return _approved("clarification", utterance, f"I didn't understand that: {reason}")

            # Route through the arbitrator — it has access to the full capability manifest
            # and SceneModel API surface, so it can recognise synthesisable spatial
            # computations (e.g. threshold filtering) that the LLM compiler missed.
            return self._arbitrate_gap(utterance, intent, cap_match)

        if intent.intent_type == "quit":
            return ApprovedCommand(command_type="quit", utterance=utterance)

        if intent.intent_type == "reset":
            return _approved("reset", utterance, payload={"clear_memory": bool(intent.clear_memory)})


        if intent.intent_type == "cache_query":
            return ApprovedCommand(command_type="cache_query", utterance=utterance)

        if intent.intent_type == "status_query":
            if intent.status_query == "ground_target" and intent.target_selector is not None:
                return _approved("ground_target_query", utterance, payload={"target_selector": intent.target_selector})

            return _approved("status_query", utterance, payload={"query": intent.status_query or "status"})


        if intent.intent_type == "knowledge_update":
            question_command = self._question_override_command(utterance)
            if question_command is not None:
                self.log("question-shaped utterance overrode knowledge_update intent")
                return question_command
            update = intent.knowledge_update or {}
            delivery_target = update.get("delivery_target")
            if delivery_target is None and intent.target_selector is not None:
                readiness_command = self.command_from_selector_readiness(
                    intent.target_selector,
                    utterance,
                )
                if readiness_command is not None:
                    return readiness_command
                grounded = self.ground_target_selector(intent.target_selector)
                if not grounded["ok"]:
                    clarification = self.maybe_start_selector_clarification(
                        utterance=utterance,
                        resume_kind="knowledge_update",
                        grounded=grounded,
                    )
                    if clarification is not None:
                        return clarification
                    return _approved("ambiguous", utterance, grounded["message"])

                target = grounded["target"]
                return ApprovedCommand(
                    kind="knowledge_update",
                    utterance=utterance,
                    payload={
                        "target_color": target["color"],
                        "target_type": target["type"],
                        "delivery_target": {
                            "color": target["color"],
                            "object_type": target["type"],
                        },
                    },
                )
            if delivery_target is None and intent.knowledge_update is not None:
                return ApprovedCommand(
                    kind="knowledge_update",
                    utterance=utterance,
                    payload={
                        "target_color": None,
                        "target_type": None,
                        "delivery_target": None,
                    },
                )
            if not isinstance(delivery_target, dict):
                return ApprovedCommand(command_type="unsupported", utterance=utterance)
            return ApprovedCommand(
                kind="knowledge_update",
                utterance=utterance,
                payload={
                    "target_color": delivery_target["color"],
                    "target_type": delivery_target["object_type"],
                    "delivery_target": {
                        "color": delivery_target["color"],
                        "object_type": delivery_target["object_type"],
                    },
                },
            )

        if intent.intent_type == "task_instruction":
            if intent.reference == "delivery_target":
                instruction = "go to the delivery target"
            elif intent.reference == "last_target":
                instruction = "go there again"
            elif intent.reference == "last_task":
                instruction = "repeat the last task"
            elif intent.target_selector is not None:
                readiness_command = self.command_from_selector_readiness(
                    intent.target_selector,
                    utterance,
                )
                if readiness_command is not None:
                    return readiness_command
                grounded = self.ground_target_selector(intent.target_selector)
                if not grounded["ok"]:
                    # Before asking the operator to disambiguate, let the arbitrator
                    # decide whether the unresolved constraint can be synthesised.
                    # This fires when the operator specifies a condition (e.g. a
                    # distance threshold) that the current primitives cannot handle.
                    if cap_match.verdict in {"missing_skills", "synthesizable", "unsupported"}:
                        arb_command = self._arbitrate_gap(utterance, intent, cap_match)
                        if arb_command.kind in {
                            "synthesis_proposal", "missing_skills", "synthesizable",
                        }:
                            return arb_command
                    clarification = self.maybe_start_selector_clarification(
                        utterance=utterance,
                        resume_kind="task_instruction",
                        grounded=grounded,
                    )
                    if clarification is not None:
                        return clarification
                    return _approved("ambiguous", utterance, grounded["message"])

                target = grounded["target"]
                instruction = f"go to the {target['color']} {target['type']}"
            elif isinstance(intent.target, dict):
                if "closest" in _normalize_utterance(utterance):
                    return ApprovedCommand(
                        kind="ambiguous",
                        utterance=utterance,
                        payload={
                            "message": (
                                "I need a valid target selector to ground closest. "
                                "I did not execute."
                            )
                        },
                    )
                color = intent.target.get("color")
                object_type = intent.target.get("object_type")
                if not color or object_type != "door" or intent.task_type != "go_to_object":
                    return ApprovedCommand(command_type="unsupported", utterance=utterance)
                instruction = intent.canonical_instruction or f"go to the {color} {object_type}"
            else:
                return ApprovedCommand(command_type="unsupported", utterance=utterance)
            return ApprovedCommand(command_type="task_instruction", utterance=instruction)

        return ApprovedCommand(command_type="unsupported", utterance=utterance)

    def _record_request_state(
        self,
        *,
        request_plan: RequestPlan | None = None,
        readiness_graph: ReadinessGraph | None = None,
        reason: str = "station_request_state",
    ) -> None:
        self.representation.record_turn_graph(
            request_plan=request_plan,
            readiness_graph=readiness_graph,
            reason=reason,
        )

    def _request_plan_is_reusable(
        self,
        plan: RequestPlan,
        readiness_graph: ReadinessGraph | None = None,
    ) -> bool:
        if plan.objective_type == "unsupported":
            return False
        if plan.expected_response == "refuse":
            return False
        if any(step.operation == "refuse" for step in plan.steps):
            return False
        if readiness_graph is not None and readiness_graph.graph_status != "executable":
            return False
        return True

    def _record_request_plan(self, utterance: str, intent: OperatorIntent) -> None:
        claims_valid = self._claims_valid_for_current_environment()
        active_summary = (
            self.active_claims.compact_summary()
            if self.active_claims is not None and claims_valid
            else None
        )

        # Build a fresh plan to compute the structural key and check against cache.
        # Plan construction is deterministic and cheap (no LLM calls).
        fresh_plan = build_request_plan(
            utterance,
            intent,
            active_claims_summary=active_summary,
            environment_identity=self.current_environment_identity,
            planning_semantics=self.planning_semantics,
        )

        cacheable_plan = self._request_plan_is_reusable(fresh_plan)
        cached_entry = (
            self.request_plan_reuse_cache.lookup(fresh_plan)
            if cacheable_plan
            else None
        )
        if cached_entry is not None:
            verdict = self.request_plan_reuse_cache.can_reuse(
                cached_entry,
                registry=self.capability_registry,
                environment_identity=self.current_environment_identity,
            )
            self.last_plan_reuse_verdict = verdict
            self.request_plan_reuse_cache.record_reuse(
                cached_entry.key,
                verdict.verdict,
                verdict.reason,
                env_id=self.env_id,
                seed=self.seed,
            )
            if verdict.verdict == "reuse":
                # Structure is proven valid — use the fresh plan (which carries the
                # current environment_assumptions) as last_request_plan so that the
                # plan is always up-to-date.  Update the cache entry so future
                # lookups also get the current plan.
                cached_entry.plan = fresh_plan
                readiness_graph = evaluate_request_plan(
                    fresh_plan,
                    registry=self.capability_registry,
                    active_claims=self.active_claims,
                    claims_valid=claims_valid,
                    environment_identity=self.current_environment_identity,
                )
                self.last_request_plan = fresh_plan
                self.last_readiness_graph = readiness_graph
                self._record_request_state(
                    request_plan=fresh_plan,
                    readiness_graph=readiness_graph,
                    reason="request_plan_reused",
                )
                self._run_mismatch_detection(fresh_plan)
                self.log(
                    f"request plan reused: key={cached_entry.key} "
                    f"steps={len(fresh_plan.steps)} "
                    f"graph_status={readiness_graph.graph_status}"
                )
                return
        else:
            self.last_plan_reuse_verdict = None

        # Fresh compile path — no matching cache entry or reuse was rejected.
        readiness_graph = evaluate_request_plan(
            fresh_plan,
            registry=self.capability_registry,
            active_claims=self.active_claims,
            claims_valid=claims_valid,
            environment_identity=self.current_environment_identity,
        )
        self.last_request_plan = fresh_plan
        self.last_readiness_graph = readiness_graph
        self._record_request_state(
            request_plan=fresh_plan,
            readiness_graph=readiness_graph,
            reason="request_plan_recorded",
        )

        # Store plans that are immediately executable so future turns can reuse them.
        if self._request_plan_is_reusable(fresh_plan, readiness_graph):
            self.request_plan_reuse_cache.store(fresh_plan)

        self._run_mismatch_detection(fresh_plan)
        
        should_repair = (
            readiness_graph.graph_status != "executable"
            or any(
                mismatch.mismatch_type == "STALE_CLAIMS"
                for mismatch in self.last_operational_mismatches
            )
        )
        if self.last_operational_mismatches and should_repair:
            repair_loop = RepairLoop(self)
            repair_events = repair_loop.attempt_repair(self.last_operational_mismatches)
            self.last_repair_events.extend(repair_events)
            
            if any(event.success for event in repair_events):
                self.log("repair successful, re-evaluating readiness graph")
                claims_valid = self._claims_valid_for_current_environment()
                readiness_graph = evaluate_request_plan(
                    fresh_plan,
                    registry=self.capability_registry,
                    active_claims=self.active_claims,
                    claims_valid=claims_valid,
                    environment_identity=self.current_environment_identity,
                )
                self.last_readiness_graph = readiness_graph
                self._record_request_state(
                    request_plan=fresh_plan,
                    readiness_graph=readiness_graph,
                    reason="request_plan_repaired",
                )
                if self._request_plan_is_reusable(fresh_plan, readiness_graph):
                    self.request_plan_reuse_cache.store(fresh_plan)

        self.log(
            "request plan: "
            f"steps={len(fresh_plan.steps)} "
            f"graph_status={readiness_graph.graph_status} "
            f"next_action={readiness_graph.next_action}"
        )

    def _record_task_instruction_request_plan(
        self,
        utterance: str,
        instruction: str,
    ) -> None:
        parsed = _parse_exact_go_to_object_utterance(instruction)
        if parsed is None:
            return
        intent = OperatorIntent(
            intent_type="task_instruction",
            canonical_instruction=instruction,
            task_type="go_to_object",
            target={
                "color": parsed["color"],
                "object_type": parsed["object_type"],
            },
            capability_status="executable",
            required_capabilities=["task.go_to_object.door"],
            confidence=1.0,
            reason="Deterministic go-to-object task instruction.",
        )
        self._record_request_plan(utterance, intent)

    def _task_intent_for_instruction(self, instruction: str) -> OperatorIntent:
        parsed = _parse_exact_go_to_object_utterance(instruction)
        if parsed is None:
            raise ValueError(f"Cannot build task execution ticket for {instruction!r}")
        return OperatorIntent(
            intent_type="task_instruction",
            canonical_instruction=instruction,
            task_type="go_to_object",
            target={
                "color": parsed["color"],
                "object_type": parsed["object_type"],
            },
            capability_status="executable",
            required_capabilities=["task.go_to_object.door"],
            confidence=1.0,
            reason="Deterministic go-to-object task instruction.",
        )

    def _local_task_plan_and_graph(
        self,
        utterance: str,
        instruction: str,
    ) -> tuple[RequestPlan, ReadinessGraph]:
        intent = self._task_intent_for_instruction(instruction)
        claims_valid = self._claims_valid_for_current_environment()
        active_summary = (
            self.active_claims.compact_summary()
            if self.active_claims is not None and claims_valid
            else None
        )
        plan = build_request_plan(
            utterance,
            intent,
            active_claims_summary=active_summary,
            environment_identity=self.current_environment_identity,
            planning_semantics=self.planning_semantics,
        )
        graph = evaluate_request_plan(
            plan,
            registry=self.capability_registry,
            active_claims=self.active_claims,
            claims_valid=claims_valid,
            environment_identity=self.current_environment_identity,
        )
        return plan, graph

    def _execution_ticket_from_plan(
        self,
        instruction: str,
        plan: RequestPlan,
        graph: ReadinessGraph,
        *,
        source: str,
        mission_plan: MissionExecutionPlan | None = None,
    ) -> ExecutionTicket:
        task = self.compose_known_task(instruction)
        provenance: dict[str, Any] = {}
        parent_request_id = None
        mission_id = None
        if mission_plan is not None:
            mission_id = mission_plan.mission_id
            parent_request_id = mission_plan.request_plan.request_id
            provenance = {
                "mission_id": mission_plan.mission_id,
                "parent_request_id": mission_plan.request_plan.request_id,
                "original_utterance": mission_plan.provenance.get(
                    "original_utterance",
                    mission_plan.description,
                ),
                "primitive_handle": (
                    mission_plan.primitive_definition.proposed_handle
                    if mission_plan.primitive_definition is not None
                    else None
                ),
                "mission_description": mission_plan.description,
            }
        return self.side_effect_authority.issue_execution_ticket(
            instruction=instruction,
            task_type=task.task_type,
            params=task.params,
            request_plan=plan,
            readiness_graph=graph,
            source=source,
            mission_id=mission_id,
            parent_request_id=parent_request_id,
            provenance=provenance,
        )

    def _execution_ticket_for_instruction(
        self,
        utterance: str,
        instruction: str,
        *,
        source: str,
        record_plan: bool,
    ) -> ExecutionTicket:
        if record_plan:
            self._record_task_instruction_request_plan(utterance, instruction)
            plan = self.last_request_plan
            graph = self.last_readiness_graph
            if plan is None or graph is None:
                raise RuntimeError("Task execution requires a recorded RequestPlan and ReadinessGraph")
        else:
            plan, graph = self._local_task_plan_and_graph(utterance, instruction)
        return self._execution_ticket_from_plan(
            instruction,
            plan,
            graph,
            source=source,
        )

    def _run_task_from_instruction(
        self,
        utterance: str,
        instruction: str,
        *,
        source: str,
        record_plan: bool,
    ) -> dict[str, Any]:
        ticket = self._execution_ticket_for_instruction(
            utterance,
            instruction,
            source=source,
            record_plan=record_plan,
        )
        return self._run_task_with_ticket(ticket)

    def _motor_intent_for_action(self, action_name: str, count: int) -> OperatorIntent:
        return OperatorIntent(
            intent_type="motor_command",
            action_name=action_name,
            repeat_count=count,
            capability_status="executable",
            required_capabilities=[],
            confidence=1.0,
            reason=f"Explicit low-level motor command: {action_name} × {count}.",
        )

    def _local_motor_plan_and_graph(
        self,
        utterance: str,
        action_name: str,
        count: int,
    ) -> tuple[RequestPlan, ReadinessGraph]:
        intent = self._motor_intent_for_action(action_name, count)
        plan = build_request_plan(
            utterance,
            intent,
            active_claims_summary=None,
            environment_identity=self.current_environment_identity,
            planning_semantics=self.planning_semantics,
        )
        graph = evaluate_request_plan(
            plan,
            registry=self.capability_registry,
            active_claims=self.active_claims,
            claims_valid=self._claims_valid_for_current_environment(),
            environment_identity=self.current_environment_identity,
        )
        return plan, graph

    def _raw_motor_ticket_from_plan(
        self,
        action_name: str,
        count: int,
        plan: RequestPlan,
        graph: ReadinessGraph,
        *,
        source: str,
    ) -> RawMotorTicket:
        return self.side_effect_authority.issue_raw_motor_ticket(
            action_name=action_name,
            repeat_count=count,
            request_plan=plan,
            readiness_graph=graph,
            source=source,
        )

    def _raw_motor_ticket_for_command(
        self,
        utterance: str,
        action_name: str,
        count: int,
        *,
        source: str,
    ) -> RawMotorTicket:
        plan, graph = self._local_motor_plan_and_graph(utterance, action_name, count)
        self.last_request_plan = plan
        self.last_readiness_graph = graph
        self._record_request_state(
            request_plan=plan,
            readiness_graph=graph,
            reason="raw_motor_ticket_created",
        )
        return self._raw_motor_ticket_from_plan(
            action_name,
            count,
            plan,
            graph,
            source=source,
        )

    def _knowledge_update_from_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "delivery_target" in payload:
            delivery_target = payload["delivery_target"]
        elif payload.get("target_color") is not None and payload.get("target_type") is not None:
            delivery_target = {
                "color": payload["target_color"],
                "object_type": payload["target_type"],
            }
        else:
            delivery_target = None
        return {"delivery_target": delivery_target}

    def _memory_writes_from_payload(self, payload: dict[str, Any]) -> list[MemoryUpdate]:
        writes: list[MemoryUpdate] = []
        for key in (
            "target_color",
            "target_type",
            "delivery_target",
            "primitive_definitions",
        ):
            if key in payload:
                writes.append(
                    MemoryUpdate(
                        scope="knowledge",
                        key=key,
                        value=payload[key],
                        reason="operator knowledge update",
                    )
                )
        if not writes and "delivery_target" in payload:
            writes.append(
                MemoryUpdate(
                    scope="knowledge",
                    key="delivery_target",
                    value=payload["delivery_target"],
                    reason="operator knowledge update",
                )
            )
        return writes

    def _memory_write_ticket_for_payload(
        self,
        utterance: str,
        payload: dict[str, Any],
        *,
        source: str,
    ) -> MemoryWriteTicket:
        intent = OperatorIntent(
            intent_type="knowledge_update",
            knowledge_update=self._knowledge_update_from_payload(payload),
            capability_status="executable",
            required_capabilities=[],
            confidence=1.0,
            reason="Deterministic durable knowledge update.",
        )
        plan = build_request_plan(
            utterance,
            intent,
            active_claims_summary=None,
            environment_identity=self.current_environment_identity,
            planning_semantics=self.planning_semantics,
        )
        graph = evaluate_request_plan(
            plan,
            registry=self.capability_registry,
            active_claims=self.active_claims,
            claims_valid=self._claims_valid_for_current_environment(),
            environment_identity=self.current_environment_identity,
        )
        self.last_request_plan = plan
        self.last_readiness_graph = graph
        self._record_request_state(
            request_plan=plan,
            readiness_graph=graph,
            reason="memory_write_ticket_created",
        )
        return self.side_effect_authority.issue_memory_write_ticket(
            writes=self._memory_writes_from_payload(payload),
            request_plan=plan,
            readiness_graph=graph,
            source=source,
        )

    def _apply_knowledge_update_from_payload(
        self,
        utterance: str,
        payload: dict[str, Any],
        *,
        source: str,
    ) -> str:
        ticket = self._memory_write_ticket_for_payload(
            utterance,
            payload,
            source=source,
        )
        return self.apply_knowledge_update(ticket)

    def _run_mismatch_detection(self, plan: RequestPlan) -> None:
        """Run mismatch detection and store results in last_operational_mismatches."""
        mismatches = default_detector.detect(
            plan,
            registry=self.capability_registry,
            scene_model=self.memory.scene_model,
            active_claims=self.active_claims,
            environment_identity=self.current_environment_identity,
        )
        self.last_operational_mismatches = mismatches
        if mismatches:
            types = [m.mismatch_type for m in mismatches]
            self.log(f"mismatch detection: {len(mismatches)} mismatch(es): {types}")

    def propose_primitive_definition(
        self,
        definition: dict[str, Any],
        *,
        mission_request: InlineMetricMissionRequest | None = None,
    ) -> str:
        request = PrimitiveDefinitionRequest.from_dict(definition)
        existing = self.capability_registry.lookup(request.proposed_handle)
        if existing is not None and existing.implementation_status == "implemented":
            if mission_request is not None:
                mission_plan = self.mission_cortex.plan_inline_metric_request(
                    mission_request,
                    active_claims=self.active_claims,
                    claims_valid=self._claims_valid_for_current_environment(),
                    environment_identity=self.current_environment_identity,
                )
                resumed = self._execute_approved_mission_plan(mission_plan)
                return (
                    "PRIMITIVE DEFINITION ALREADY REGISTERED\n"
                    f"handle={request.proposed_handle}\n\n"
                    "RESUMING ORIGINAL REQUEST\n"
                    f"{resumed}"
                )
            return (
                "PRIMITIVE DEFINITION NOT STORED\n"
                f"handle={request.proposed_handle}\n"
                "reason=already_implemented"
            )
        mission_plan = None
        if mission_request is not None:
            mission_plan = self.mission_cortex.plan_inline_metric_request(
                mission_request,
                active_claims=self.active_claims,
                claims_valid=self._claims_valid_for_current_environment(),
                environment_identity=self.current_environment_identity,
            )
            plan = mission_plan.request_plan
            graph = mission_plan.readiness_graph
        else:
            plan, graph = self.mission_cortex.plan_primitive_definition(
                request,
                utterance=request.provenance.get("operator_utterance", request.name),
                active_claims=self.active_claims,
                claims_valid=self._claims_valid_for_current_environment(),
                environment_identity=self.current_environment_identity,
            )
        self.last_request_plan = plan
        self.last_readiness_graph = graph
        self.pending_primitive_definition = PendingPrimitiveDefinition(
            request=request,
            request_plan=plan,
            readiness_graph=graph,
            mission_plan=mission_plan,
        )
        dependency_lines = []
        for metric, handle in zip(request.dependencies, request.dependency_handles):
            spec = self.capability_registry.lookup(handle)
            status = spec.implementation_status if spec is not None else "missing"
            dependency_lines.append(f"  - {metric}: {handle} ({status})")
        expression = request.expression
        formula = request.provenance.get("formula")
        return (
            "PRIMITIVE DEFINITION PROPOSAL\n"
            f"name={request.name}\n"
            f"normalized_name={request.normalized_name}\n"
            f"handle={request.proposed_handle}\n"
            f"definition_type={request.definition_type}\n"
            f"safety_class={request.safety_class}\n"
            f"formula={formula}\n"
            f"expression={expression}\n"
            "dependencies:\n"
            + ("\n".join(dependency_lines) if dependency_lines else "  - none")
            + (
                "\nAfter registration I will resume the original request."
                if mission_plan is not None
                else ""
            )
            + "\nShould I build it now? (yes / no)"
        )

    def handle_pending_primitive_definition(
        self,
        utterance: str,
        command: ApprovedCommand,
    ) -> str:
        pending = self.pending_primitive_definition
        if pending is None:
            return "No primitive definition is pending."
        normalized = _normalize_utterance(utterance)
        if command.kind == "quit":
            self.pending_primitive_definition = None
            return "QUIT"
        if command.kind == "cancel":
            self.pending_primitive_definition = None
            return "CANCELLED: primitive definition proposal cleared"
        if command.kind == "reset":
            self.pending_primitive_definition = None
            return self.reset(clear_memory=bool(command.payload.get("clear_memory")))
        if normalized in {"no", "nope", "nah", "do not", "don't", "reject", "cancel it"}:
            self.pending_primitive_definition = None
            return (
                "PRIMITIVE DEFINITION REJECTED\n"
                f"handle={pending.request.proposed_handle}\n"
                "registered=false"
            )
        if self._is_acceptance(normalized):
            self.pending_primitive_definition = None
            result = self._approve_primitive_definition(pending.request, utterance)
            if pending.mission_plan is not None and "registered=true" in result:
                resumed = self._execute_approved_mission_plan(pending.mission_plan)
                return f"{result}\n\nRESUMING ORIGINAL REQUEST\n{resumed}"
            return result
        return (
            "CLARIFY\n"
            f"Primitive definition pending: {pending.request.proposed_handle}\n"
            "Please answer yes or no."
        )

    def _execute_approved_mission_plan(
        self,
        mission_plan: MissionExecutionPlan,
    ) -> str:
        mission_plan = self.mission_cortex.resume_after_approval(
            mission_plan,
            active_claims=self.active_claims,
            claims_valid=self._claims_valid_for_current_environment(),
            environment_identity=self.current_environment_identity,
        )
        if mission_plan.continuation_intent is None:
            raise RuntimeError("Approved mission plan requires continuation_intent")
        utterance = mission_plan.provenance.get("original_utterance", mission_plan.description)
        intent = mission_plan.continuation_intent
        command = self.command_from_operator_intent(intent, utterance)
        if command.kind == "task_instruction":
            instruction = self.resolve_task_instruction(command.utterance)
            if instruction is None:
                return self.missing_reference_summary(command.utterance)
            plan = self.last_request_plan
            graph = self.last_readiness_graph
            if plan is None or graph is None:
                raise RuntimeError("Approved mission execution requires a recorded RequestPlan")
            mission_plan.continuation_request_plan = plan
            mission_plan.continuation_readiness_graph = graph
            ticket = self._execution_ticket_from_plan(
                instruction,
                plan,
                graph,
                source="operator_mission_flow",
                mission_plan=mission_plan,
            )
            mission_plan.child_tickets.append(ticket)
            self.last_mission_execution_plan = mission_plan
            run_result = self._run_task_with_ticket(ticket)
            result = self.result_summary(run_result)
            if self.last_result is not None:
                instruction = self.last_result.get("task", {}).get("instruction")
                if instruction:
                    return f"resolved_instruction={instruction}\n{result}"
            return result

        result = self.execute_command(command)
        self.last_mission_execution_plan = mission_plan
        if self.last_result is not None:
            instruction = self.last_result.get("task", {}).get("instruction")
            if instruction:
                return f"resolved_instruction={instruction}\n{result}"
        return result

    def _approve_primitive_definition(
        self,
        request: PrimitiveDefinitionRequest,
        approval_utterance: str,
    ) -> str:
        if request.safety_class != "query":
            return (
                "PRIMITIVE DEFINITION FAILED\n"
                "reason=non_query_safety_class\n"
                "registered=false"
            )
        for metric in request.dependencies:
            ok, message = self._ensure_metric_dependency(metric)
            if not ok:
                return (
                    "PRIMITIVE DEFINITION FAILED\n"
                    f"handle={request.proposed_handle}\n"
                    f"dependency={metric}\n"
                    f"reason={message}\n"
                    "registered=false"
                )

        fn = self._ranker_for_metric_expression(request.expression)
        validation_error = self._validate_metric_ranker(fn)
        if validation_error is not None:
            return (
                "PRIMITIVE DEFINITION FAILED\n"
                f"handle={request.proposed_handle}\n"
                f"reason=validation_failed:{validation_error}\n"
                "registered=false"
            )

        description = (
            f"Operator-defined ranked-door distance metric '{request.normalized_name}' "
            f"from expression {request.expression}."
        )
        registered = self.capability_registry.register_dynamic(
            request.proposed_handle,
            description,
            fn,
            inputs=["scene.door_candidates", "agent_pose"],
            outputs=["ranked_door_list", "distances"],
            side_effects=[],
            safety_class="query",
            authority_level="operator",
            validation_hooks=["operator_metric_fixture_validation"],
        )
        if not registered:
            return (
                "PRIMITIVE DEFINITION FAILED\n"
                f"handle={request.proposed_handle}\n"
                "reason=registration_conflict\n"
                "registered=false"
            )
        self._install_metric_semantics(request.normalized_name)
        self._record_primitive_definition(request, approval_utterance)
        return (
            "PRIMITIVE DEFINITION REGISTERED\n"
            f"name={request.name}\n"
            f"normalized_name={request.normalized_name}\n"
            f"handle={request.proposed_handle}\n"
            "safety_class=query\n"
            "registered=true"
        )

    def _ensure_metric_dependency(self, metric: str) -> tuple[bool, str]:
        handle = _metric_dependency_handle(metric)
        spec = self.capability_registry.lookup(handle)
        if spec is None:
            return False, f"dependency handle {handle} is missing"
        if spec.implementation_status == "implemented":
            return True, "implemented"
        if not (spec.implementation_status == "synthesizable" or spec.safe_to_synthesize):
            return False, f"dependency handle {handle} is not safe to synthesize"
        fn = self._ranker_for_metric_expression({"op": "alias", "metric": metric})
        validation_error = self._validate_metric_ranker(fn)
        if validation_error is not None:
            return False, f"dependency validation failed: {validation_error}"
        if not self.capability_registry.register_synthesized(handle, fn):
            return False, f"dependency handle {handle} could not be registered"
        return True, "synthesized"

    def _install_metric_semantics(self, metric: str) -> None:
        metrics = self.operational_context.grounding_semantics.setdefault(
            "distance_metrics",
            [],
        )
        if not isinstance(metrics, list):
            metrics = []
            self.operational_context.grounding_semantics["distance_metrics"] = metrics
        if metric not in metrics:
            metrics.append(metric)

    def _record_primitive_definition(
        self,
        request: PrimitiveDefinitionRequest,
        approval_utterance: str,
    ) -> None:
        definitions = self.memory.knowledge.get("primitive_definitions")
        if not isinstance(definitions, dict):
            definitions = {}
        record = request.as_dict()
        record["provenance"] = {
            **record.get("provenance", {}),
            "approval_utterance": approval_utterance,
            "registered_handle": request.proposed_handle,
        }
        definitions[request.normalized_name] = record
        ticket = self._memory_write_ticket_for_payload(
            request.provenance.get("operator_utterance", request.name),
            {"primitive_definitions": definitions},
            source="operator_primitive_definition",
        )
        self.apply_knowledge_update(ticket)

    def _base_metric_distance(
        self,
        scene: SceneModel,
        obj: SceneObject,
        metric: str,
    ) -> float:
        if metric == "manhattan":
            return float(scene.manhattan_distance_from_agent(obj))
        if metric == "euclidean":
            return math.sqrt((obj.x - scene.agent_x) ** 2 + (obj.y - scene.agent_y) ** 2)
        fn = self.capability_registry.get_synthesized_callable(_metric_dependency_handle(metric))
        if fn is not None:
            ranked = fn(scene, {"object_type": "door", "color": None, "exclude_colors": []})
            for dist, candidate in ranked:
                if candidate is obj or (
                    candidate.x == obj.x
                    and candidate.y == obj.y
                    and candidate.color == obj.color
                    and candidate.object_type == obj.object_type
                ):
                    return float(dist)
        raise ValueError(f"Unknown metric dependency: {metric}")

    def _evaluate_metric_expression(
        self,
        expression: dict[str, Any],
        scene: SceneModel,
        obj: SceneObject,
    ) -> float:
        op = expression.get("op")
        if op == "alias":
            return self._base_metric_distance(scene, obj, str(expression.get("metric")))
        if op == "min":
            values = [
                self._base_metric_distance(scene, obj, str(metric))
                for metric in expression.get("metrics", [])
            ]
            return min(values)
        if op == "max":
            values = [
                self._base_metric_distance(scene, obj, str(metric))
                for metric in expression.get("metrics", [])
            ]
            return max(values)
        if op == "sum":
            values = [
                self._base_metric_distance(scene, obj, str(metric))
                for metric in expression.get("metrics", [])
            ]
            return sum(values)
        if op == "mod":
            base = self._base_metric_distance(scene, obj, str(expression.get("metric")))
            constant = float(expression.get("constant"))
            if constant == 0:
                raise ValueError("mod constant must be non-zero")
            return base % constant
        if op == "add":
            return self._base_metric_distance(
                scene,
                obj,
                str(expression.get("metric")),
            ) + float(expression.get("constant"))
        if op == "subtract":
            return self._base_metric_distance(
                scene,
                obj,
                str(expression.get("metric")),
            ) - float(expression.get("constant"))
        if op in {"abs_diff", "abs_diff_plus"}:
            metrics = list(expression.get("metrics", []))
            if len(metrics) < 2:
                raise ValueError("abs_diff requires two metrics")
            value = abs(
                self._base_metric_distance(scene, obj, str(metrics[0]))
                - self._base_metric_distance(scene, obj, str(metrics[1]))
            )
            if op == "abs_diff_plus":
                value += float(expression.get("constant"))
            return value
        raise ValueError(f"Unsupported metric expression op: {op}")

    def _ranker_for_metric_expression(self, expression: dict[str, Any]) -> Any:
        def _rank(scene: SceneModel, selector: dict[str, Any]) -> list[tuple[float, SceneObject]]:
            doors = scene.find(
                object_type=selector.get("object_type", "door"),
                color=selector.get("color"),
                exclude_colors=selector.get("exclude_colors") or [],
            )
            ranked = [
                (self._evaluate_metric_expression(expression, scene, door), door)
                for door in doors
            ]
            return sorted(
                ranked,
                key=lambda pair: (pair[0], pair[1].color or "", pair[1].x, pair[1].y),
            )

        return _rank

    def _validate_metric_ranker(self, fn: Any) -> str | None:
        scene = SceneModel(
            agent_x=1,
            agent_y=1,
            agent_dir=0,
            grid_width=6,
            grid_height=6,
            objects=[
                SceneObject("door", "red", 1, 4),
                SceneObject("door", "blue", 5, 1),
                SceneObject("door", "green", 4, 5),
            ],
            source="fixture",
        )
        try:
            ranked = fn(scene, {"object_type": "door", "color": None, "exclude_colors": []})
        except Exception as exc:  # noqa: BLE001
            return f"{type(exc).__name__}: {exc}"
        if not ranked:
            return "ranker returned no candidates"
        distances = [float(item[0]) for item in ranked]
        if distances != sorted(distances):
            return "ranker output was not sorted ascending"
        if any(not math.isfinite(distance) for distance in distances):
            return "ranker returned a non-finite distance"
        return None

    def _resolve_metric_name(self, metric: str) -> str | None:
        candidate = _normalize_metric_name(metric)
        compact = candidate.replace("_", "")
        for known in self.planning_semantics.metrics:
            known_compact = known.replace("_", "")
            if candidate == known or compact == known_compact:
                return known
        return None

    def metric_query_summary(self, command: ApprovedCommand) -> str:
        raw_metric = str(command.payload.get("metric") or "")
        metric = self._resolve_metric_name(raw_metric)
        if metric is None:
            proposed = _metric_dependency_handle(_normalize_metric_name(raw_metric))
            self.last_request_plan = RequestPlan(
                request_id=f"metric_query:{abs(hash((command.utterance, raw_metric))) % 1_000_000}",
                original_utterance=command.utterance,
                objective_type="query",
                objective_summary=f"Rank doors by undefined metric {raw_metric}",
                steps=[
                    RequestPlanStep(
                        step_id="rank_scene_doors",
                        layer="grounding",
                        operation="rank",
                        required_handle=proposed,
                        inputs={"object_type": "door"},
                        outputs=["active_claims.ranked_scene_doors"],
                        constraints={"metric": raw_metric, "reference": "agent"},
                    )
                ],
                expected_response="answer_query",
            )
            self.last_readiness_graph = evaluate_request_plan(
                self.last_request_plan,
                registry=self.capability_registry,
                active_claims=self.active_claims,
                claims_valid=self._claims_valid_for_current_environment(),
                environment_identity=self.current_environment_identity,
            )
            return (
                "CUSTOM METRIC MISSING\n"
                f"metric={raw_metric}\n"
                f"handle={proposed}\n"
                "I do not have that metric defined yet. Define it first, then approve it."
            )

        handle = _metric_dependency_handle(metric)
        intent = OperatorIntent(
            intent_type="status_query",
            status_query="ground_target",
            grounding_query_plan={
                "object_type": "door",
                "operation": "rank",
                "primitive_handle": handle,
                "metric": metric,
                "reference": "agent",
                "order": "ascending",
                "ordinal": None,
                "color": None,
                "exclude_colors": [],
                "distance_value": None,
                "comparison": None,
                "tie_policy": "display",
                "answer_fields": ["ranked_doors", "distance"],
                "required_capabilities": [handle],
                "preserved_constraints": ["rank", "door", metric],
            },
            capability_status="executable",
            required_capabilities=[handle],
            confidence=1.0,
            reason=f"Operator requested ranked doors by custom metric {metric}.",
        )
        result = self.command_from_operator_intent(intent, command.utterance)
        return self.execute_command(result)

    def _command_from_grounding_query_plan(
        self,
        utterance: str,
        intent: OperatorIntent,
    ) -> ApprovedCommand | None:
        plan = intent.grounding_query_plan
        if plan is None:
            return None

        invalid_reason = self._validate_grounding_query_plan_preserves_utterance(
            utterance,
            plan,
            intent=intent,
        )
        if invalid_reason is not None:
            if plan.get("metric") is None:
                clarification = self._maybe_start_semantic_metric_clarification(
                    utterance=utterance,
                    reason=invalid_reason,
                )
                if clarification is not None:
                    return clarification
            return ApprovedCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={
                    "message": (
                        "I could not validate the semantic query plan. "
                        f"{invalid_reason}"
                    )
                },
            )

        if self._grounding_query_plan_needs_distance_metric(plan):
            clarification = self._maybe_start_semantic_metric_clarification(
                utterance=utterance,
                reason="The semantic query plan omitted a required distance metric.",
            )
            if clarification is not None:
                return clarification

        plan_required = list(plan.get("required_capabilities") or [])
        if plan_required:
            plan_intent = replace(intent, required_capabilities=plan_required)
            cap_match = default_matcher.match(plan_intent, self.capability_registry)
            if cap_match.verdict in {"missing_skills", "synthesizable", "unsupported"}:
                return self._arbitrate_gap(utterance, plan_intent, cap_match)
        else:
            cap_match = default_matcher.match(intent, self.capability_registry)

        handle = plan.get("primitive_handle")
        if handle:
            spec = self.capability_registry.lookup(handle)
            if spec is None:
                return _approved("missing_skills", utterance, f"Missing primitive: {handle}", capability_match=cap_match)

            if spec.implementation_status != "implemented":
                plan_intent = replace(
                    intent,
                    required_capabilities=plan_required or [handle],
                )
                return self._arbitrate_gap(
                    utterance,
                    plan_intent,
                    default_matcher.match(plan_intent, self.capability_registry),
                )

        if handle == "grounding.claims.last_grounded_target":
            return self._compose_claim_reference_query_plan(utterance, intent)
        if isinstance(handle, str) and handle.startswith("claims."):
            return self._dispatch_claims_filter(utterance, intent, cap_match)

        return self._compose_grounding_query_plan(
            utterance,
            intent,
            plan,
            cap_match,
        )

    def _grounding_query_plan_needs_distance_metric(self, plan: dict[str, Any]) -> bool:
        if plan.get("metric") is not None:
            return False
        if plan.get("primitive_handle") == "grounding.claims.last_grounded_target":
            return False
        answer_fields = {str(field).lower() for field in plan.get("answer_fields", [])}
        if answer_fields.intersection(
            {
                "closest",
                "farthest",
                "first_closest",
                "second_closest",
                "third_closest",
                "fourth_closest",
                "fifth_closest",
                "first_farthest",
                "second_farthest",
                "third_farthest",
                "fourth_farthest",
                "fifth_farthest",
                "distance",
            }
        ):
            return True
        if plan.get("order") in {"ascending", "descending"}:
            return True
        if plan.get("distance_value") is not None:
            return True
        return False

    def _maybe_start_semantic_metric_clarification(
        self,
        *,
        utterance: str,
        reason: str,
    ) -> ApprovedCommand | None:
        normalized = _normalize_utterance(utterance)
        mentions_ranked_distance = any(
            term in normalized
            for term in (
                "closest",
                "nearest",
                "shortest",
                "farthest",
                "furthest",
                "least close",
                "distance",
            )
        )
        if not mentions_ranked_distance:
            return None
        if "manhattan" in normalized or "euclidean" in normalized:
            return None
        self.pending_clarification = PendingClarification(
            clarification_type="semantic_query_missing_field",
            original_utterance=utterance,
            resume_kind="semantic_query_plan",
            partial_selector={"validation_reason": reason},
            missing_field="distance_metric",
            supported_values=["manhattan"],
            **self._pending_clarification_trace(
                utterance,
                "semantic_query_missing_field",
            ),
        )
        return ApprovedCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": (
                    "CLARIFY\n"
                    "That grounding request depends on distance. Which distance metric should I use?\n"
                    "Supported: manhattan"
                )
            },
        )

    def _validate_grounding_query_plan_preserves_utterance(
        self,
        utterance: str,
        plan: dict[str, Any],
        intent: "OperatorIntent | None" = None,
    ) -> str | None:
        normalized = _normalize_utterance(utterance)
        ordinal_words = {
            "first": 1,
            "1st": 1,
            "second": 2,
            "2nd": 2,
            "third": 3,
            "3rd": 3,
            "fourth": 4,
            "4th": 4,
            "fifth": 5,
            "5th": 5,
        }
        for word, ordinal in ordinal_words.items():
            if re.search(rf"\b{re.escape(word)}\b", normalized):
                expected_answer_field = None
                if any(term in normalized for term in ("closest", "nearest", "shortest")):
                    expected_answer_field = f"{word}_closest"
                if any(term in normalized for term in ("farthest", "furthest")):
                    expected_answer_field = f"{word}_farthest"
                if plan.get("ordinal") != ordinal and expected_answer_field not in {
                    str(field).lower() for field in plan.get("answer_fields", [])
                }:
                    return f"The utterance says {word}, but the plan ordinal is {plan.get('ordinal')}."

        answer_fields = {str(f).lower() for f in plan.get("answer_fields", [])}
        operation = plan.get("operation")
        order = plan.get("order")

        # ── Direction check: objective-based (primary) vs vocabulary (fallback) ──
        # When selection_objective is set by the LLM, check direction purely from
        # the structured enum — no vocabulary lists needed. This makes the check
        # domain-agnostic: "hottest"→maximum works the same as "farthest"→maximum.
        selection_obj = getattr(intent, "selection_objective", None) if intent is not None else None
        if selection_obj is not None:
            expected_order = (
                "descending" if selection_obj.direction == "maximum" else "ascending"
            )
            if order is not None and order != expected_order:
                covers = (
                    (selection_obj.direction == "maximum" and "farthest" in answer_fields)
                    or (selection_obj.direction == "minimum" and "closest" in answer_fields)
                )
                if not covers:
                    return (
                        f"Objective says direction={selection_obj.direction!r} "
                        f"(requires order={expected_order!r}) but plan declares order={order!r}."
                    )
        else:
            # Vocabulary-based fallback (SmokeTestCompiler, legacy LLM output).
            expected_order = infer_direction_from_utterance(normalized)
            if expected_order == "descending":
                if not (order == "descending" or "farthest" in answer_fields):
                    return (
                        "The utterance asks for maximum-distance selection, but "
                        "the plan does not preserve descending/farthest semantics."
                    )
            if expected_order == "ascending":
                if not (order == "ascending" or "closest" in answer_fields or operation == "rank"):
                    return (
                        "The utterance asks for minimum-distance selection, but "
                        "the plan does not preserve ascending/closest semantics."
                    )

        plan_metric = plan.get("metric")
        metric_dependencies = {
            str(metric)
            for metric in (plan.get("metric_dependencies") or [])
        }
        if (
            "euclidean" in normalized
            and plan_metric != "euclidean"
            and "euclidean" not in metric_dependencies
        ):
            return "The utterance specifies Euclidean distance, but the plan does not."
        if (
            "manhattan" in normalized
            and plan_metric != "manhattan"
            and "manhattan" not in metric_dependencies
        ):
            return "The utterance specifies Manhattan distance, but the plan does not."

        color = self.domain_helper.color_reference_in_utterance(normalized)
        if color is None:
            color = self.domain_helper.bare_color_reference(normalized)
        if color is not None and plan.get("color") not in {None, color}:
            return f"The utterance mentions {color}, but the plan targets {plan.get('color')}."
        return None

    def _compose_claim_reference_query_plan(
        self,
        utterance: str,
        intent: OperatorIntent,
    ) -> ApprovedCommand:
        scene = self.memory.scene_model
        if scene is None:
            return _approved("ambiguous", utterance, "No scene data available for that reference.")

        if self.active_claims is None:
            return _approved("ambiguous", utterance, "No active grounded target for that reference.")

        if not self._claims_valid_for_current_environment(scene):
            self.active_claims = None
            return _approved("ambiguous", utterance, "Scene has changed since that grounding. Please re-ground.")

        entry = self.active_claims.last_grounded_target
        wants_task = intent.intent_type == "task_instruction" or self._utterance_requests_navigation(
            _normalize_utterance(utterance)
        )
        if wants_task:
            return self._task_command_for_entry(entry, utterance)
        return ApprovedCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": (
                    "GROUNDING ANSWER\n"
                    f"target={self.domain_helper.entry_label(entry)}\n"
                    "source=active_claims"
                )
            },
        )

    def _compose_grounding_query_plan(
        self,
        utterance: str,
        intent: OperatorIntent,
        plan: dict[str, Any],
        cap_match: CapabilityMatchResult,
    ) -> ApprovedCommand:
        wants_task = intent.intent_type == "task_instruction" or self._utterance_requests_navigation(
            _normalize_utterance(utterance)
        )
        claims = self._ensure_ranked_door_claims(plan.get("primitive_handle"))
        if isinstance(claims, str):
            return _approved("missing_skills", utterance, payload={"message": claims, "match": cap_match.compact()}, capability_match=cap_match)


        operation = plan.get("operation")
        answer_fields = {str(f).lower() for f in plan.get("answer_fields", [])}
        distance_value = plan.get("distance_value")
        color = plan.get("color")
        ordinal = plan.get("ordinal")
        order = plan.get("order")

        if operation in {"rank", "list"}:
            return ApprovedCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self.domain_helper.format_ranked_doors_from_entries(
                        claims.ranked_scene_doors,
                        metric=str(
                            claims.last_grounding_query.get("distance_metric")
                            or self.domain_helper.metric_from_grounding_handle(
                                str(claims.last_grounding_query.get("primitive", ""))
                            )
                        ),
                        include_navigation_hint=not wants_task,
                    )
                },
                capability_match=cap_match,
            )

        if distance_value is not None:
            return self._compose_distance_reference(
                utterance,
                claims,
                int(distance_value),
                wants_task=wants_task,
            )

        if operation == "answer" and order in {"ascending", "descending"} and not answer_fields:
            return ApprovedCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self._format_plan_extreme_answer(
                        claims,
                        include_closest=order == "ascending",
                        include_farthest=order == "descending",
                    )
                },
                capability_match=cap_match,
            )

        if ordinal == 1 and order in {"ascending", "descending"}:
            if wants_task:
                return self._compose_extreme_task(
                    utterance,
                    claims,
                    extreme="farthest" if order == "descending" else "closest",
                    cap_match=cap_match,
                )
            return ApprovedCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self._format_plan_extreme_answer(
                        claims,
                        include_closest=order == "ascending",
                        include_farthest=order == "descending",
                    )
                },
                capability_match=cap_match,
            )

        if ordinal is not None and order in {"ascending", "descending"}:
            return self._compose_ordinal_plan_reference(
                utterance,
                claims,
                ordinal=int(ordinal),
                order=order,
                wants_task=wants_task,
            )

        if answer_fields.intersection({"closest", "farthest"}):
            return ApprovedCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self._format_plan_answer_fields(claims, answer_fields)
                },
                capability_match=cap_match,
            )

        ordinal_answer_fields = {
            field
            for field in answer_fields
            if re.match(r"^(first|second|third|fourth|fifth)_(closest|farthest)$", field)
        }
        if ordinal_answer_fields:
            return ApprovedCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self._format_plan_answer_fields(claims, answer_fields)
                },
                capability_match=cap_match,
            )

        if color is not None:
            matches = [entry for entry in claims.ranked_scene_doors if entry.color == color]
            if wants_task:
                if len(matches) == 1:
                    return self._task_command_for_entry(matches[0], utterance)
                return _approved("ambiguous", utterance, f"No unique {color} door is visible. I did not execute.")

            if matches:
                self._set_last_grounded_claim(matches[0], claims)
            return ApprovedCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self.domain_helper.format_color_plan_answer(
                        color=color,
                        matches=matches,
                        answer_fields=answer_fields,
                    )
                },
                capability_match=cap_match,
            )

        return _approved("ambiguous", utterance, "I could not compose a result from the semantic query plan.")


    def _arbitrate_gap(
        self,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
    ) -> ApprovedCommand:
        # Only pass implemented handles — synthesizable ones are not alternatives
        available = [
            h for h in self.capability_registry.primitive_names()
            if (spec := self.capability_registry.lookup(h)) is not None
            and spec.implementation_status == "implemented"
        ]
        # Full synthesizable surface from the registry — passed when the compiler
        # emitted no specific required_capabilities (e.g. intent_type=unsupported).
        # Lets the arbitrator LLM match the utterance against the actual synthesizable
        # surface rather than only what the compiler declared.
        registry_synthesizable = [
            h for h in self.capability_registry.primitive_names()
            if (spec := self.capability_registry.lookup(h)) is not None
            and (spec.implementation_status == "synthesizable" or spec.safe_to_synthesize)
        ]
        scene_summary = self._build_scene_summary_for_arbitrator()
        arbitration_kwargs = {
            "utterance": utterance,
            "intent_type": intent.intent_type,
            "required_capabilities": intent.required_capabilities,
            "missing_handles": cap_match.missing,
            "synthesizable_handles": cap_match.synthesizable_handles,
            "available_handles": available,
            "scene_summary": scene_summary,
            "registry_synthesizable_handles": registry_synthesizable,
        }
        arbitrate_signature = inspect.signature(self.arbitrator.arbitrate)
        accepts_registry_synthesizable = (
            "registry_synthesizable_handles" in arbitrate_signature.parameters
            or any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in arbitrate_signature.parameters.values()
            )
        )
        if not accepts_registry_synthesizable:
            arbitration_kwargs.pop("registry_synthesizable_handles")
        decision = self.arbitrator.arbitrate(**arbitration_kwargs)
        self.last_arbitration_trace = ArbitrationTrace(
            utterance=utterance,
            intent_type=intent.intent_type,
            required_capabilities=intent.required_capabilities,
            missing_handles=cap_match.missing,
            synthesizable_handles=cap_match.synthesizable_handles,
            decision=decision,
        )
        self.log(
            f"arbitration: {decision.decision_type} "
            f"safe={decision.safe_to_execute} "
            f"reason={decision.reason[:60]}"
        )

        if decision.decision_type == "synthesize":
            # Pre-declared synthesizable handles take priority — they are exact matches
            # already validated by the registry. The arbitrator's free-form proposal is
            # only used when the registry has no synthesizable candidates.
            for handle in cap_match.synthesizable_handles:
                spec = self.capability_registry.lookup(handle)
                if spec is None or not spec.safe_to_synthesize:
                    continue
                return self._propose_synthesis(handle, utterance, intent, cap_match)
            # No pre-declared synthesizable handle — use arbitrator's dynamic proposal.
            if decision.proposed_handle and decision.proposed_description:
                return self._propose_synthesis(
                    decision.proposed_handle,
                    utterance,
                    intent,
                    cap_match,
                    proposed_description=decision.proposed_description,
                    proposed_condition=decision.proposed_condition,
                )
            return ApprovedCommand(
                kind="synthesizable",
                utterance=utterance,
                payload={
                    "message": decision.operator_message,
                    "match": cap_match.compact(),
                },
                capability_match=cap_match,
            )

        if cap_match.synthesizable_handles:
            # The registry is the source of truth. If an exact required handle is
            # marked safe_to_synthesize, do not let an unreliable arbitration
            # refusal hide that path from the operator.
            handle = cap_match.synthesizable_handles[0]
            return self._propose_synthesis(
                handle,
                utterance,
                intent,
                cap_match,
                proposed_condition=(
                    decision.proposed_condition
                    or self._condition_from_intent_or_utterance(handle, intent, utterance)
                ),
            )

        if decision.decision_type == "clarify" and decision.clarification_prompt:
            # Prefer the arbitrator's suggested_handle when it names an implemented primitive.
            # Fall back to inferring from missing handles if the LLM left it null.
            if decision.suggested_handle and self.capability_registry.lookup(
                decision.suggested_handle
            ) is not None:
                offer_handle = decision.suggested_handle
            else:
                offer_handle = self._infer_offer_action(cap_match.missing)
            self.pending_clarification = PendingClarification(
                clarification_type="arbitrator_offer",
                original_utterance=utterance,
                resume_kind=offer_handle,
                partial_selector={},
                missing_field="acceptance",
                supported_values=["yes", "ok", "sure", "please"],
                **self._pending_clarification_trace(
                    utterance,
                    "arbitrator_offer",
                ),
            )
            return _approved("clarification", utterance, decision.clarification_prompt, capability_match=cap_match)


        if decision.decision_type == "substitute" and decision.suggested_handle:
            if not decision.safe_to_execute:
                msg = decision.operator_message or (
                    f"Suggested substitute: {decision.suggested_handle}, "
                    "but marked not safe to execute."
                )
                return _approved("missing_skills", utterance, payload={"message": msg, "match": cap_match.compact()}, capability_match=cap_match)

            self.log(f"arbitration substitute approved: {decision.suggested_handle}")
            return self._execute_approved_substitute(
                decision.suggested_handle, utterance, intent, cap_match, decision
            )

        # Default: refuse
        if cap_match.verdict == "synthesizable":
            kind = "synthesizable"
        elif cap_match.verdict == "ok":
            kind = "ambiguous"
        else:
            kind = "missing_skills"
        return ApprovedCommand(
            kind=kind,
            utterance=utterance,
            payload={
                "message": decision.operator_message or cap_match.operator_message(),
                "match": cap_match.compact(),
            },
            capability_match=cap_match,
        )

    def _try_synthesize_primitive(
        self,
        handle: str,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
    ) -> ApprovedCommand | None:
        """Attempt to synthesize, validate, and register a missing grounding primitive.

        Returns an ApprovedCommand on success (re-routes to grounding) or None on failure
        so the caller falls through to the standard synthesize refusal message.
        """
        spec = self.capability_registry.lookup(handle)
        # Dynamic handle: arbitrator invented a handle not pre-declared in the registry.
        # Recover the description from the pending proposal (set by _propose_synthesis).
        pending = self.pending_synthesis_proposal
        dynamic_description = (
            pending.proposed_description
            if pending is not None and pending.handle == handle
            else None
        )
        description = (
            spec.description if spec is not None
            else dynamic_description or handle
        )
        consumes: tuple[str, ...] = tuple(spec.inputs) if spec is not None else (
            "scene.door_candidates", "agent_pose"
        )
        produces: tuple[str, ...] = tuple(spec.outputs) if spec is not None else (
            "grounded_target", "distance"
        )
        validation = None
        result = None
        validation_failures: list[str] = []
        previous_code: str | None = None
        for attempt in range(2):
            if attempt == 0:
                self.log(f"synthesis: attempting {handle} (dynamic={spec is None})")
            else:
                self.log(f"synthesis: repairing {handle} after validation failure")
            result = self.synthesizer.synthesize(
                handle=handle,
                description=description,
                consumes=consumes,
                produces=produces,
                previous_code=previous_code,
                validation_error="; ".join(validation_failures) if validation_failures else None,
            )
            if result.status != "success":
                self.log(f"synthesis: {handle} refused/failed — {result.error_message}")
                return None

            self.log(f"synthesis: validating {handle}")
            self.log(f"synthesis: generated code:\n{result.code}")
            validation = self.validator.validate(
                handle=handle,
                function_name=result.function_name,
                code=result.code,
            )
            if validation.passed:
                break
            validation_failures = list(validation.failures)
            previous_code = result.code
            self.log(
                f"synthesis: {handle} validation failed — "
                + "; ".join(validation_failures)
            )

        if validation is None or result is None:
            return None
        if not validation.passed:
            return ApprovedCommand(
                kind="synthesizable",
                utterance=utterance,
                payload={
                    "message": (
                        f"I synthesized a candidate for '{handle}' but it failed "
                        f"validation: {'; '.join(validation.failures[:2])}. "
                        "I did not register or execute it."
                    ),
                    "match": cap_match.compact(),
                },
                capability_match=cap_match,
            )

        if spec is None:
            # Brand-new primitive — create the registry entry from scratch.
            registered = self.capability_registry.register_dynamic(
                handle, description, validation.compiled_fn
            )
        else:
            registered = self.capability_registry.register_synthesized(
                handle, validation.compiled_fn
            )
        if not registered:
            self.log(f"synthesis: {handle} could not be registered (already implemented or conflict)")
            return ApprovedCommand(
                kind="synthesizable",
                utterance=utterance,
                payload={
                    "message": (
                        f"I synthesized and validated '{handle}' but could not register it — "
                        "it may already be implemented. Try using it directly."
                    ),
                    "match": cap_match.compact(),
                },
                capability_match=cap_match,
            )

        self.log(f"synthesis: {handle} registered — re-routing grounding")
        return self._execute_synthesized_grounding(handle, utterance, intent, cap_match)

    def _execute_synthesized_grounding(
        self,
        handle: str,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
    ) -> ApprovedCommand:
        """Run a freshly synthesized grounding primitive and continue with original intent."""
        fn = self.capability_registry.get_synthesized_callable(handle)
        if fn is None:
            return ApprovedCommand(
                kind="synthesizable",
                utterance=utterance,
                payload={
                    "message": f"Primitive '{handle}' was registered but callable not found.",
                    "match": cap_match.compact(),
                },
                capability_match=cap_match,
            )

        scene = self._ensure_scene_model()
        if scene is None:
            return _approved("ambiguous", utterance, "No scene data available yet. I did not execute.")


        selector = dict(intent.target_selector or {})
        doors_in_scene = [o for o in scene.objects if o.object_type == "door"]
        self.log(
            f"synthesis grounding: scene has {len(scene.objects)} objects "
            f"({len(doors_in_scene)} doors), selector={selector}"
        )
        try:
            ranked = fn(scene, selector)
        except Exception as exc:  # noqa: BLE001
            return ApprovedCommand(
                kind="synthesizable",
                utterance=utterance,
                payload={
                    "message": (
                        f"Synthesized primitive '{handle}' raised an error at runtime: {exc}. "
                        "I did not execute the task."
                    ),
                    "match": cap_match.compact(),
                },
                capability_match=cap_match,
            )

        if not ranked:
            return _approved("ambiguous", utterance, "No matching doors found. I did not execute.")


        self._write_ranked_claims(ranked, {**selector, "primitive": handle})
        distance, target_obj = ranked[0]
        target = _scene_object_to_dict(target_obj)

        if intent.intent_type == "task_instruction":
            instruction = f"go to the {target['color']} {target['type']}"
            self.log(
                f"synthesis grounding: resolved to {instruction} "
                f"(distance={distance:.2f} via {handle})"
            )
            return ApprovedCommand(command_type="task_instruction", utterance=instruction)

        # For status queries, show the full ranking across all found objects.
        short_handle = handle.split(".")[-2] if "." in handle else handle
        header = f"DOORS RANKED BY {short_handle.upper()} DISTANCE FROM AGENT (synthesized: {handle})"
        lines = [header]
        for i, (dist, obj) in enumerate(ranked):
            lines.append(
                f"  {i + 1}. {obj.color} {obj.object_type}"
                f" @({obj.x},{obj.y}) dist={dist:.2f}"
            )
        lines.append("\n(I can navigate to any specific door — tell me which color.)")
        return _approved("clarification", utterance, "\n".join(lines), capability_match=cap_match)


    # ── Claims-filter synthesis path ──────────────────────────────────────────

    def _dispatch_claims_filter(
        self,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
    ) -> ApprovedCommand:
        """Execute a registered claims-filter primitive against active_claims."""
        plan = intent.grounding_query_plan or {}
        handle = plan.get("primitive_handle") or ""
        if not handle:
            # Infer handle from metric in plan
            metric = plan.get("metric", "euclidean") or "euclidean"
            handle = f"claims.filter.threshold.{metric}"

        fn = self.capability_registry.get_synthesized_callable(handle)
        if fn is None:
            return _approved("missing_skills", utterance, f"Claims-filter primitive '{handle}' is not registered.", capability_match=cap_match)

        return self._execute_synthesized_claims_filter(handle, fn, utterance, intent, cap_match)

    def _try_synthesize_claims_filter(
        self,
        handle: str,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
    ) -> ApprovedCommand | None:
        """Synthesize, validate, and register a claims-filter primitive.

        Same proposal→validate→register pipeline as _try_synthesize_primitive but
        uses CLAIMS_FILTER_FUNCTION_SIGNATURE and ClaimsFilterFixtures.
        Returns None if synthesis is refused (falls through to error message).
        """
        spec = self.capability_registry.lookup(handle)
        description = (
            spec.description if spec is not None
            else f"Parametric distance threshold filter over ActiveClaims entries ({handle})."
        )
        self.log(f"claims-filter synthesis: synthesizing {handle}")
        result = self.synthesizer.synthesize(
            handle=handle,
            description=description,
            consumes=tuple(spec.inputs) if spec is not None else ("active_claims.ranked_scene_doors", "condition"),
            produces=tuple(spec.outputs) if spec is not None else ("filtered_entries",),
        )
        if result.status != "success":
            self.log(f"claims-filter synthesis refused/failed: {result.error_message}")
            return None

        self.log(f"claims-filter synthesis: generated code:\n{result.code}")
        vr = self.validator.validate(handle, result.function_name, result.code)
        if not vr.passed:
            self.log(f"claims-filter validation failed: {vr.failures} — attempting repair")
            repair = self.synthesizer.synthesize(
                handle=handle,
                description=description,
                consumes=tuple(spec.inputs) if spec is not None else ("active_claims.ranked_scene_doors", "condition"),
                produces=tuple(spec.outputs) if spec is not None else ("filtered_entries",),
                previous_code=result.code,
                validation_error="; ".join(vr.failures),
            )
            if repair.status != "success":
                return None
            vr = self.validator.validate(handle, repair.function_name, repair.code)
            if not vr.passed:
                self.log(f"claims-filter repair also failed: {vr.failures}")
                return ApprovedCommand(
                    kind="synthesizable",
                    utterance=utterance,
                    payload={
                        "message": (
                            f"I synthesized a claims-filter candidate for '{handle}' but it failed "
                            f"validation: {'; '.join(vr.failures)}. I did not register it."
                        ),
                        "match": cap_match.compact(),
                    },
                    capability_match=cap_match,
                )

        fn = vr.compiled_fn
        registered = self.capability_registry.register_synthesized(handle, fn)
        if not registered:
            self.capability_registry.register_dynamic(handle, description, fn)
        self.log(f"claims-filter synthesis: {handle} registered")
        return self._execute_synthesized_claims_filter(handle, fn, utterance, intent, cap_match)

    def _execute_synthesized_claims_filter(
        self,
        handle: str,
        fn: Any,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
    ) -> ApprovedCommand:
        """Call a registered claims-filter fn(entries, condition) and route the result."""
        plan = intent.grounding_query_plan or {}
        metric = plan.get("metric") or "manhattan"
        expected_ranked_handle = f"grounding.all_doors.ranked.{metric}.agent"
        if (
            self.active_claims is None
            or not self.active_claims.ranked_scene_doors
            or self.active_claims.last_grounding_query.get("primitive") != expected_ranked_handle
        ):
            claims = self._ensure_ranked_door_claims(expected_ranked_handle)
            if isinstance(claims, str):
                return ApprovedCommand(
                    kind="missing_skills",
                    utterance=utterance,
                    payload={
                        "message": (
                            "No compatible active grounding claims to filter, and I could not "
                            f"refresh them with {expected_ranked_handle}: {claims}"
                        ),
                        "match": cap_match.compact(),
                    },
                    capability_match=cap_match,
                )

        if self.active_claims is None or not self.active_claims.ranked_scene_doors:
            return ApprovedCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={
                    "message": (
                        "No active grounding claims to filter. "
                        "Run a ranked grounding query first (e.g. 'show all doors by euclidean distance')."
                    ),
                },
            )

        condition = {
            "threshold": float(plan.get("distance_value") or 0),
            "comparison": plan.get("comparison") or "above",
            "metric": metric,
        }
        entries = self.active_claims.ranked_scene_doors
        self.log(
            f"claims-filter: calling {handle} with {len(entries)} entries, condition={condition}"
        )
        try:
            filtered = fn(entries, condition)
        except Exception as exc:  # noqa: BLE001
            return ApprovedCommand(
                kind="synthesizable",
                utterance=utterance,
                payload={
                    "message": f"Claims-filter '{handle}' raised an error: {exc}.",
                    "match": cap_match.compact(),
                },
                capability_match=cap_match,
            )

        if not isinstance(filtered, list):
            return _approved("synthesizable", utterance, f"Claims-filter '{handle}' returned {type(filtered).__name__}, expected list.", capability_match=cap_match)


        metric_label = condition.get("metric") or "distance"
        comparison = condition["comparison"]
        threshold = condition["threshold"]

        if not filtered:
            known = ", ".join(
                f"{e.color}@{e.distance:.1f}" for e in entries
            )
            return ApprovedCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={
                    "message": (
                        f"No doors with {metric_label} distance {comparison} {threshold}. "
                        f"Known distances: {known}"
                    ),
                },
            )

        order = plan.get("order")
        ordinal = plan.get("ordinal")
        if order in {"ascending", "descending"} or ordinal is not None:
            ordered = sorted(
                filtered,
                key=lambda e: (e.distance, e.color or "", e.x, e.y),
                reverse=order == "descending",
            )
            target_index = max(0, int(ordinal or 1) - 1)
            if target_index >= len(ordered):
                return ApprovedCommand(
                    kind="ambiguous",
                    utterance=utterance,
                    payload={
                        "message": (
                            f"The filter matched only {len(ordered)} doors, so ordinal "
                            f"{target_index + 1} is not available."
                        )
                    },
                )
            selected = ordered[target_index]
            tied = [entry for entry in ordered if entry.distance == selected.distance]
            if len(tied) > 1 and plan.get("tie_policy") == "clarify":
                message = (
                    "CLARIFY\n"
                    "That filtered selector lands inside a distance tie. Which one should I use?\n"
                    f"Options: {_format_targets([self.domain_helper.entry_target_dict(e) for e in tied])}"
                )
                return self._candidate_clarification_for_entries(
                    utterance=utterance,
                    entries=tied,
                    resume_kind=(
                        "task_instruction"
                        if intent.intent_type in {"task_instruction", "claim_reference"}
                        else "ground_target_query"
                    ),
                    message=message,
                )
            self.active_claims.last_grounded_target = selected
            self.active_claims.last_grounded_rank = next(
                (i for i, e in enumerate(entries) if e.x == selected.x and e.y == selected.y),
                0,
            )
            if intent.intent_type == "task_instruction":
                return _approved("task_instruction", f"go to the {selected.color} {selected.object_type}")

            lines = [
                f"GROUNDED TARGET (claims filter: {handle})",
                f"target={selected.color} {selected.object_type}@({selected.x},{selected.y})",
                f"distance={selected.distance:.2f} [{metric_label}]",
                f"filter={comparison} {threshold}",
                "source=active_claims",
            ]
            return _approved("clarification", utterance, "\n".join(lines), capability_match=cap_match)


        if len(filtered) == 1:
            entry = filtered[0]
            self.active_claims.last_grounded_target = entry
            self.active_claims.last_grounded_rank = next(
                (i for i, e in enumerate(entries) if e.x == entry.x and e.y == entry.y), 0
            )
            if intent.intent_type == "task_instruction":
                instruction = f"go to the {entry.color} {entry.object_type}"
                return ApprovedCommand(command_type="task_instruction", utterance=instruction)
            lines = [
                f"GROUNDED TARGET (claims filter: {handle})",
                f"target={entry.color} {entry.object_type}@({entry.x},{entry.y})",
                f"distance={entry.distance:.2f} [{metric_label}]",
                f"filter={comparison} {threshold}",
                "source=active_claims",
            ]
            return _approved("clarification", utterance, "\n".join(lines), capability_match=cap_match)


        # Multiple matches — display ranked list and let operator choose
        header = (
            f"DOORS WITH {metric_label.upper()} DISTANCE "
            f"{comparison.upper()} {threshold} (claims filter: {handle})"
        )
        lines = [header]
        for i, entry in enumerate(filtered):
            lines.append(
                f"  {i + 1}. {entry.color} {entry.object_type}"
                f"@({entry.x},{entry.y}) dist={entry.distance:.2f}"
            )
        lines.append("\n(Tell me which one to navigate to.)")
        return self._candidate_clarification_for_entries(
            utterance=utterance,
            entries=filtered,
            resume_kind=(
                "task_instruction"
                if intent.intent_type in {"task_instruction", "claim_reference"}
                else "ground_target_query"
            ),
            message="\n".join(lines),
        )

    def _propose_synthesis(
        self,
        handle: str,
        utterance: str,
        intent: Any,
        cap_match: CapabilityMatchResult,
        *,
        proposed_description: str | None = None,
        proposed_condition: dict[str, Any] | None = None,
    ) -> ApprovedCommand:
        """Build a natural-language proposal and enter pending_synthesis_proposal state.

        proposed_description is set when the arbitrator dynamically invented the handle
        (not pre-declared in the registry). If the handle is already in the registry,
        its spec description is used instead.
        """
        spec = self.capability_registry.lookup(handle)
        description = proposed_description or (spec.description if spec else handle)
        similar = self._find_similar_implemented(handle)
        similar_str = (
            ", ".join(similar) if similar else "existing grounding patterns in the registry"
        )
        origin = "new primitive" if spec is None else "marked safe to synthesize"
        message = (
            "SYNTHESIS PROPOSAL\n"
            f"I don't have '{handle}' implemented yet — I can build it as a {origin}.\n"
            f"It would be composed using {similar_str} as building blocks.\n"
            f"What it would do: {description}\n"
            "Should I build it now? (yes / no)"
        )
        self.pending_synthesis_proposal = PendingSynthesisProposal(
            handle=handle,
            original_utterance=utterance,
            intent=intent,
            cap_match=cap_match,
            similar_handles=similar,
            proposed_description=description if spec is None else None,
            proposed_condition=(
                proposed_condition
                if proposed_condition is not None
                else self._condition_from_intent_or_utterance(handle, intent, utterance)
            ),
        )
        self.log(f"synthesis proposal pending for {handle} (dynamic={spec is None})")
        return ApprovedCommand(
            kind="synthesis_proposal",
            utterance=utterance,
            payload={
                "message": message,
                "handle": handle,
                "similar_handles": similar,
                "dynamic": spec is None,
            },
            capability_match=cap_match,
        )

    def _find_similar_implemented(self, handle: str) -> list[str]:
        """Return up to 2 implemented primitives in the same layer as handle."""
        parts = handle.split(".")
        layer = parts[0] if parts else ""
        similar = [
            name
            for name in self.capability_registry.primitive_names(layer=layer)
            if name != handle
            and (spec := self.capability_registry.lookup(name)) is not None
            and spec.implementation_status == "implemented"
        ]
        return similar[:2]

    def _condition_from_intent_or_utterance(
        self,
        handle: str,
        intent: Any,
        utterance: str,
    ) -> dict[str, Any] | None:
        """Carry claims-filter parameters through synthesis approval.

        Primary source is the typed grounding_query_plan. The utterance parser is a
        conservative fallback for arbitration-only paths where the LLM proposed a
        claims handle but no query plan existed.
        """
        if not handle.startswith("claims.filter.threshold."):
            return None
        plan = getattr(intent, "grounding_query_plan", None) or {}
        threshold = plan.get("distance_value")
        comparison = plan.get("comparison")
        metric = plan.get("metric")
        if threshold is not None and comparison is not None:
            return {
                "threshold": float(threshold),
                "comparison": comparison,
                "metric": metric or handle.rsplit(".", 1)[-1],
            }

        normalized = _normalize_utterance(utterance)
        number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", normalized)
        if not number_match:
            return None
        if any(
            term in normalized
            for term in ("above", "greater than", "more than", "over", "exceeds")
        ):
            comparison = "above"
        elif any(term in normalized for term in ("at least", "no less than")):
            comparison = "at_least"
        elif any(term in normalized for term in ("below", "less than", "under")):
            comparison = "below"
        elif any(term in normalized for term in ("at most", "within", "no more than")):
            comparison = "at_most"
        else:
            comparison = "above"
        metric = (
            "euclidean"
            if "euclidean" in normalized
            else "manhattan"
            if "manhattan" in normalized
            else handle.rsplit(".", 1)[-1]
        )
        return {
            "threshold": float(number_match.group(1)),
            "comparison": comparison,
            "metric": metric,
        }

    def _intent_with_claims_filter_plan(
        self,
        proposal: PendingSynthesisProposal,
    ) -> OperatorIntent:
        intent = proposal.intent
        if not isinstance(intent, OperatorIntent):
            return intent
        condition = proposal.proposed_condition or self._condition_from_intent_or_utterance(
            proposal.handle,
            intent,
            proposal.original_utterance,
        )
        if not condition:
            return intent
        metric = condition.get("metric") or proposal.handle.rsplit(".", 1)[-1]
        threshold = condition.get("threshold")
        comparison = condition.get("comparison") or "above"
        if threshold is None:
            return intent
        threshold_float = float(threshold)
        existing_plan = intent.grounding_query_plan or {}
        wants_task = self._utterance_requests_navigation(
            _normalize_utterance(proposal.original_utterance)
        )
        required = [proposal.handle]
        if wants_task:
            required.append("task.go_to_object.door")
        plan = {
            "object_type": "door",
            "operation": "filter",
            "primitive_handle": proposal.handle,
            "metric": metric,
            "reference": "agent",
            "order": existing_plan.get("order"),
            "ordinal": existing_plan.get("ordinal"),
            "color": None,
            "exclude_colors": [],
            "distance_value": (
                int(threshold_float) if threshold_float.is_integer() else threshold_float
            ),
            "comparison": comparison,
            "tie_policy": "clarify",
            "answer_fields": existing_plan.get("answer_fields") or ["target", "distance"],
            "required_capabilities": required,
            "preserved_constraints": [
                "door",
                str(metric),
                str(comparison),
                str(threshold_float),
            ],
        }
        return replace(
            intent,
            intent_type="task_instruction" if wants_task else "claim_reference",
            task_type="go_to_object" if wants_task else intent.task_type,
            claim_reference="threshold_filter",
            grounding_query_plan=plan,
            capability_status="executable",
            required_capabilities=required,
        )

    def handle_pending_synthesis_proposal(
        self,
        utterance: str,
        command: ApprovedCommand,
    ) -> str:
        """Resolve an operator response to a synthesis proposal via LLM classification."""
        proposal = self.pending_synthesis_proposal
        if proposal is None:
            return "No synthesis proposal is pending."

        if command.kind == "quit":
            self.pending_synthesis_proposal = None
            return "QUIT"
        if command.kind == "cancel":
            self.pending_synthesis_proposal = None
            return "CANCELLED: synthesis proposal cleared"
        if command.kind == "reset":
            self.pending_synthesis_proposal = None
            return self.reset(clear_memory=bool(command.payload.get("clear_memory")))

        # Pass the pending proposal to the LLM so it can classify yes/no/redirect
        proposal_context = {
            "handle": proposal.handle,
            "proposed_description": proposal.proposed_description,
            "similar_handles": proposal.similar_handles,
            "proposed_condition": proposal.proposed_condition,
        }
        self.log("compiling operator intent with pending synthesis proposal context")
        intent_command = self.command_from_llm_intent(
            utterance, pending_proposal=proposal_context
        )

        if intent_command.kind == "accept_proposal":
            self.pending_synthesis_proposal = None
            self.log(f"synthesis proposal approved by operator: {proposal.handle}")
            if proposal.handle.startswith("claims."):
                proposal_intent = self._intent_with_claims_filter_plan(proposal)
                result = self._try_synthesize_claims_filter(
                    proposal.handle,
                    proposal.original_utterance,
                    proposal_intent,
                    proposal.cap_match,
                )
            else:
                result = self._try_synthesize_primitive(
                    proposal.handle,
                    proposal.original_utterance,
                    proposal.intent,
                    proposal.cap_match,
                )
            if result is None:
                return (
                    f"Synthesis of '{proposal.handle}' was not available — "
                    "the synthesizer backend refused or is not configured. "
                    "Set OPENROUTER_API_KEY and use compiler=llm to enable synthesis."
                )
            if result.kind == "synthesizable":
                return result.payload.get("message", "Synthesis validation failed.")
            return self.execute_command(result)

        if intent_command.kind == "reject_proposal":
            self.pending_synthesis_proposal = None
            return f"Understood. I will not build '{proposal.handle}'."

        # New unrelated instruction — clear proposal and handle it
        self.pending_synthesis_proposal = None
        self.log("synthesis proposal cleared by new operator intent (redirect)")
        return self.execute_command(intent_command)

    def _build_scene_summary_for_arbitrator(self) -> dict[str, Any] | None:
        scene = self.memory.scene_model
        if scene is None:
            return None
        doors = scene.find(object_type="door")
        summary = {
            "agent_position": {"x": scene.agent_x, "y": scene.agent_y},
            "visible_doors": [
                {
                    "color": d.color,
                    "x": d.x,
                    "y": d.y,
                    "manhattan_distance": scene.manhattan_distance_from_agent(d),
                }
                for d in doors
            ],
        }
        if self.active_claims is not None and self._claims_valid_for_current_environment(scene):
            summary["active_claims"] = self.active_claims.compact_summary()
        return summary

    def _capability_status_for_handle(self, handle: str | None) -> str:
        if handle is None:
            return "needs_clarification"
        spec = self.capability_registry.lookup(handle)
        if spec is None:
            return "missing_skills"
        if spec.implementation_status == "implemented":
            return "executable"
        if spec.implementation_status == "synthesizable" or spec.safe_to_synthesize:
            return "synthesizable"
        if spec.implementation_status in {"planned", "missing"}:
            return "missing_skills"
        return "unsupported"

    def _requested_ranked_handle_from_verification(
        self,
        verif_result: IntentVerificationResult,
    ) -> str | None:
        for signal in verif_result.signals:
            if ".ranked." in signal.required_handle:
                return signal.required_handle
        return None

    def _promote_verified_query_intent(
        self,
        utterance: str,
        intent: OperatorIntent,
        verif_result: IntentVerificationResult,
    ) -> OperatorIntent:
        """Turn verifier-rescued unresolved text into a real query intent."""
        if intent.intent_type not in {"unsupported", "ambiguous"}:
            return intent
        if not verif_result.signals:
            return intent

        handle = self._requested_ranked_handle_from_verification(verif_result)
        if handle is None:
            return intent

        normalized = _normalize_utterance(utterance)
        signal_types = {signal.signal_type for signal in verif_result.signals}
        object_type = (
            self.planning_semantics.object_type_from_text(normalized)
            or self.domain_helper.default_object_type
        )
        metric = self.domain_helper.metric_from_grounding_handle(handle)
        is_cardinality = "cardinality" in signal_types
        ordinal_words = {
            "first": 1,
            "1st": 1,
            "second": 2,
            "2nd": 2,
            "third": 3,
            "3rd": 3,
            "fourth": 4,
            "4th": 4,
            "fifth": 5,
            "5th": 5,
        }
        ordinal_word: str | None = None
        ordinal_value: int | None = None
        ordinal_direction: str | None = None
        for source in [
            *(signal.detected_term for signal in verif_result.signals),
            normalized,
        ]:
            ordinal_match = re.search(
                r"\b(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th)\s+"
                r"(closest|nearest|shortest|farthest|furthest)\b",
                source,
            )
            if ordinal_match:
                ordinal_word = ordinal_match.group(1)
                ordinal_value = ordinal_words[ordinal_word]
                direction_term = ordinal_match.group(2)
                ordinal_direction = (
                    "farthest"
                    if direction_term in {"farthest", "furthest"}
                    else "closest"
                )
                break
        is_descending = any(
            term in normalized
            for term in ("farthest", "furthest", "most distant", "least close")
        ) or ordinal_direction == "farthest"
        if ordinal_value is not None:
            operation = "answer"
            answer_fields = [f"{ordinal_word}_{ordinal_direction}"]
            tie_policy = "clarify"
        elif is_cardinality:
            operation = "rank"
            answer_fields = ["ranked_doors", "distance"]
            tie_policy = "display"
        else:
            operation = "answer"
            answer_fields = ["target", "distance"]
            tie_policy = "clarify"
        return replace(
            intent,
            intent_type="status_query",
            status_query="ground_target",
            task_type=None,
            target=None,
            target_selector=None,
            grounding_query_plan={
                "object_type": object_type,
                "operation": operation,
                "primitive_handle": handle,
                "metric": metric,
                "reference": "agent",
                "order": "descending" if is_descending else "ascending",
                "ordinal": ordinal_value,
                "color": None,
                "exclude_colors": [],
                "distance_value": None,
                "tie_policy": tie_policy,
                "answer_fields": answer_fields,
                "required_capabilities": [handle],
                "preserved_constraints": [
                    signal.signal_type for signal in verif_result.signals
                ],
            },
            capability_status=self._capability_status_for_handle(handle),
            required_capabilities=[handle],
            reason=(
                "Verifier promoted unresolved utterance into a typed grounding "
                f"query. Original reason: {intent.reason}"
            ),
        )

    def _try_compose_grounding_result(
        self,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
        verif_result: IntentVerificationResult,
    ) -> ApprovedCommand | None:
        if not verif_result.signals:
            return None
        signal_types = {signal.signal_type for signal in verif_result.signals}
        if not signal_types.intersection(
            {"superlative", "cardinality", "ordinal", "distance_value"}
        ):
            return None

        normalized = _normalize_utterance(utterance)
        ranked_handle = self._ranked_handle_from_verification(verif_result)
        if "euclidean" in normalized and ranked_handle is None:
            # Use the synthesized euclidean handle if it has already been registered.
            euc = "grounding.all_doors.ranked.euclidean.agent"
            spec = self.capability_registry.lookup(euc)
            if spec is not None and spec.implementation_status == "implemented":
                ranked_handle = euc
            else:
                return None
        claims = self._ensure_ranked_door_claims(ranked_handle)
        if isinstance(claims, str):
            return _approved("missing_skills", utterance, payload={"message": claims, "match": cap_match.compact()}, capability_match=cap_match)

        if not claims.ranked_scene_doors:
            return _approved("ambiguous", utterance, "No visible doors are available to compose from.")


        wants_task = self._utterance_requests_navigation(normalized) or (
            intent.intent_type == "task_instruction"
        )

        distance = self._distance_value_from_utterance(normalized)
        if distance is not None:
            return self._compose_distance_reference(
                utterance,
                claims,
                distance,
                wants_task=wants_task,
            )

        if "ordinal" in signal_types:
            ordinal_result = self._compose_ordinal_reference(
                utterance,
                claims,
                normalized,
                wants_task=wants_task,
            )
            if ordinal_result is not None:
                return ordinal_result

        if "cardinality" in signal_types and "superlative" not in signal_types:
            return ApprovedCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self.domain_helper.format_ranked_doors_from_entries(
                        claims.ranked_scene_doors,
                        metric=str(
                            claims.last_grounding_query.get("distance_metric")
                            or self.domain_helper.metric_from_grounding_handle(
                                str(claims.last_grounding_query.get("primitive", ""))
                            )
                        ),
                        include_navigation_hint=not wants_task,
                    )
                },
                capability_match=cap_match,
            )

        if "farthest" in normalized or "furthest" in normalized or "least close" in normalized:
            if wants_task:
                return self._compose_extreme_task(
                    utterance,
                    claims,
                    extreme="farthest",
                    cap_match=cap_match,
                )
            return _approved("clarification", utterance, self._format_extreme_answer(claims, normalized), capability_match=cap_match)


        if "closest" in normalized or "nearest" in normalized:
            if wants_task:
                return self._compose_extreme_task(
                    utterance,
                    claims,
                    extreme="closest",
                    cap_match=cap_match,
                )
            return _approved("clarification", utterance, self._format_extreme_answer(claims, normalized), capability_match=cap_match)


        return None

    def _ranked_handle_from_verification(
        self,
        verif_result: IntentVerificationResult,
    ) -> str | None:
        for signal in verif_result.signals:
            handle = signal.required_handle
            if "all_doors.ranked" not in handle:
                continue
            spec = self.capability_registry.lookup(handle)
            if spec is not None and spec.implementation_status == "implemented":
                return handle
        return None

    def _ensure_ranked_door_claims(
        self,
        handle: str | None = None,
    ) -> StationActiveClaims | str:
        handle = handle or "grounding.all_doors.ranked.manhattan.agent"
        spec = self.capability_registry.lookup(handle)
        if spec is None:
            return f"Capability '{handle}' is not registered. I cannot compose that result."
        if spec.implementation_status != "implemented":
            return (
                f"Capability '{handle}' is registered but not implemented "
                f"(status={spec.implementation_status})."
            )
        scene = self._ensure_scene_model()
        if scene is None:
            return "No scene data available yet."
        if (
            self.active_claims is not None
            and self._claims_valid_for_current_environment(scene)
            and self.active_claims.last_grounding_query.get("primitive") == handle
        ):
            return self.active_claims
        doors = scene.find(object_type="door")
        if not doors:
            return "No doors visible in the current scene."
        fn = self.capability_registry.get_synthesized_callable(handle)
        if fn is not None:
            try:
                ranked = fn(
                    scene,
                    {"object_type": "door", "color": None, "exclude_colors": []},
                )
            except Exception as exc:  # noqa: BLE001
                return f"Synthesized primitive '{handle}' raised an error: {exc}"
        else:
            ranked = sorted(
                [(scene.manhattan_distance_from_agent(d), d) for d in doors],
                key=lambda pair: (pair[0], pair[1].color or "", pair[1].x, pair[1].y),
            )
        self._write_ranked_claims(
            ranked,
            {
                "object_type": "door",
                "relation": "all",
                "distance_metric": self.domain_helper.metric_from_grounding_handle(handle),
                "distance_reference": "agent",
                "primitive": handle,
            },
        )
        if self.active_claims is None:
            return "Unable to write active grounding claims."
        return self.active_claims

    def _command_from_active_claim_text(self, utterance: str) -> ApprovedCommand | None:
        normalized = _normalize_utterance(utterance)
        # Concept-teach prefixes embed a navigation verb ("go to") inside the phrase
        # but are not themselves navigation requests — skip them so the utterance
        # can be routed to the LLM / SmokeTestCompiler for concept_teach handling.
        if re.match(r"^(?:when|if)\s+(?:i\s+)?say\b", normalized):
            return None
        # Mission contracts ("mission: X; Y") must route to the compiler, not navigation.
        # Check raw utterance because _normalize_utterance strips colons/semicolons.
        if re.match(r"^\s*mission\s*:", utterance, re.IGNORECASE):
            return None
        # Sequential utterances ("go to X then go to Y") must not be parsed as a
        # single navigation command — structural shape takes priority over content.
        if re.search(r"\b(?:and\s+then|then|followed\s+by)\b", normalized):
            return None
        if not self._utterance_requests_navigation(normalized):
            return None
        color = self.domain_helper.color_reference_in_utterance(normalized)
        if color is None:
            return None
        claims = self._ensure_ranked_door_claims()
        if isinstance(claims, str):
            return None
        matches = [entry for entry in claims.ranked_scene_doors if entry.color == color]
        if len(matches) != 1:
            return None
        return _approved("task_instruction", f"go to the {matches[0].color} door")


    def _command_from_grounding_followup(self, utterance: str) -> ApprovedCommand | None:
        normalized = _normalize_utterance(utterance)
        if self.active_claims is None:
            return None
        if not self._claims_valid_for_current_environment():
            return None
        if "distance" not in normalized:
            return None
        if self._utterance_requests_navigation(normalized):
            return None
        if normalize_distance_ordinal(normalized) is not None:
            return None
        if infer_direction_from_utterance(normalized) is not None:
            return None

        metric = next(
            (
                candidate
                for candidate in self.planning_semantics.metrics
                if re.search(rf"\b{re.escape(candidate)}\b", normalized)
            ),
            None,
        )
        if metric is None:
            return None

        last_query = dict(self.active_claims.last_grounding_query or {})
        object_type = str(
            last_query.get("object_type")
            or self.planning_semantics.default_object_type
            or self.domain_helper.default_object_type
        )
        handle = self.planning_semantics.ranked_handle(metric, object_type=object_type)
        if handle is None:
            return None

        intent = OperatorIntent(
            intent_type="status_query",
            status_query="ground_target",
            grounding_query_plan={
                "object_type": object_type,
                "operation": "rank",
                "primitive_handle": handle,
                "metric": metric,
                "reference": last_query.get("distance_reference") or "agent",
                "order": "ascending",
                "ordinal": None,
                "color": None,
                "exclude_colors": [],
                "distance_value": None,
                "tie_policy": "display",
                "answer_fields": ["ranked_doors", "distance"],
                "required_capabilities": [handle],
                "preserved_constraints": ["followup", object_type, metric],
            },
            capability_status=self._capability_status_for_handle(handle),
            required_capabilities=[handle],
            confidence=0.85,
            reason=(
                "Metric-only follow-up resolved against the last ranked "
                "grounding context."
            ),
        )
        self.log(
            "grounding follow-up resolved: "
            f"object_type={object_type} metric={metric} handle={handle}"
        )
        return self.command_from_operator_intent(intent, utterance)

    def _utterance_requests_navigation(self, normalized: str) -> bool:
        return bool(
            re.search(r"\b(go to|go the|reach|find|get to|head to|navigate to)\b", normalized)
        )

    def _distance_value_from_utterance(self, normalized: str) -> int | None:
        match = re.search(
            r"\b(?:distance\s+(?:of\s+)?|with\s+(?:a\s+)?distance\s+(?:of\s+)?)(\d+)\b",
            normalized,
        )
        if not match:
            return None
        return int(match.group(1))

    def _task_command_for_entry(self, entry: GroundedDoorEntry, utterance: str) -> ApprovedCommand:
        return _approved("task_instruction", self.domain_helper.task_utterance_for_entry(entry))


    def _candidate_clarification_for_entries(
        self,
        *,
        utterance: str,
        entries: list[GroundedDoorEntry],
        resume_kind: str,
        message: str,
    ) -> ApprovedCommand:
        self.pending_clarification = PendingClarification(
            clarification_type="target_selector_candidate_choice",
            original_utterance=utterance,
            resume_kind=resume_kind,
            partial_selector={},
            missing_field="candidate",
            supported_values=sorted(
                str(entry.color) for entry in entries if entry.color is not None
            ),
            candidates=[self.domain_helper.entry_target_dict(entry) for entry in entries],
            **self._pending_clarification_trace(
                utterance,
                "target_selector_candidate_choice",
            ),
        )
        return _approved("clarification", utterance, message)


    def _compose_distance_reference(
        self,
        utterance: str,
        claims: StationActiveClaims,
        distance: int,
        *,
        wants_task: bool,
    ) -> ApprovedCommand:
        matches = [entry for entry in claims.ranked_scene_doors if entry.distance == distance]
        if not matches:
            return ApprovedCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={
                    "message": (
                        f"No visible door has Manhattan distance {distance} from the agent."
                    )
                },
            )
        if len(matches) > 1:
            message = (
                "CLARIFY\n"
                f"Multiple doors have distance {distance}. Which one should I use?\n"
                f"Options: {_format_targets([self.domain_helper.entry_target_dict(e) for e in matches])}"
            )
            if wants_task:
                return self._candidate_clarification_for_entries(
                    utterance=utterance,
                    entries=matches,
                    resume_kind="task_instruction",
                    message=message,
                )
            return ApprovedCommand(command_type="clarification", utterance=utterance, payload={"message": message})
        entry = matches[0]
        if wants_task:
            return self._task_command_for_entry(entry, utterance)
        return ApprovedCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": (
                    "GROUNDING ANSWER\n"
                    f"distance={distance}\n"
                    f"target={self.domain_helper.entry_label(entry)}"
                )
            },
        )

    def _compose_ordinal_reference(
        self,
        utterance: str,
        claims: StationActiveClaims,
        normalized: str,
        *,
        wants_task: bool,
    ) -> ApprovedCommand | None:
        match = re.search(
            r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th)\s+(closest|nearest|farthest|furthest)\b",
            normalized,
        )
        if not match:
            return None
        ordinal = match.group(1)
        direction = match.group(2)
        ordinal_to_index = {
            "second": 1,
            "2nd": 1,
            "third": 2,
            "3rd": 2,
            "fourth": 3,
            "4th": 3,
            "fifth": 4,
            "5th": 4,
        }
        rank = ordinal_to_index[ordinal]
        ranked = list(claims.ranked_scene_doors)
        if direction in {"farthest", "furthest"}:
            ranked = list(reversed(ranked))
        if rank >= len(ranked):
            return _approved("ambiguous", utterance, "There are not enough visible doors for that ordinal request.")

        entry = ranked[rank]
        tied = [item for item in ranked if item.distance == entry.distance]
        if len(tied) > 1:
            message = (
                "CLARIFY\n"
                f"That ordinal falls inside a distance tie at distance {entry.distance}. "
                "Which one should I use?\n"
                f"Options: {_format_targets([self.domain_helper.entry_target_dict(e) for e in tied])}"
            )
            if wants_task:
                return self._candidate_clarification_for_entries(
                    utterance=utterance,
                    entries=tied,
                    resume_kind="task_instruction",
                    message=message,
                )
            return ApprovedCommand(command_type="clarification", utterance=utterance, payload={"message": message})
        self._set_last_grounded_claim(entry, claims)
        if wants_task:
            return self._task_command_for_entry(entry, utterance)
        return ApprovedCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": (
                    "GROUNDING ANSWER\n"
                    f"{ordinal} {direction}={self.domain_helper.entry_label(entry)}"
                )
            },
        )

    def _compose_ordinal_plan_reference(
        self,
        utterance: str,
        claims: StationActiveClaims,
        *,
        ordinal: int,
        order: str,
        wants_task: bool,
    ) -> ApprovedCommand:
        ranked = list(claims.ranked_scene_doors)
        if order == "descending":
            ranked = list(reversed(ranked))
        rank = ordinal - 1
        if rank >= len(ranked):
            return _approved("ambiguous", utterance, "There are not enough visible doors for that ordinal request.")

        entry = ranked[rank]
        tied = [item for item in ranked if item.distance == entry.distance]
        if len(tied) > 1:
            message = (
                "CLARIFY\n"
                f"That ordinal falls inside a distance tie at distance {entry.distance}. "
                "Which one should I use?\n"
                f"Options: {_format_targets([self.domain_helper.entry_target_dict(e) for e in tied])}"
            )
            if wants_task:
                return self._candidate_clarification_for_entries(
                    utterance=utterance,
                    entries=tied,
                    resume_kind="task_instruction",
                    message=message,
                )
            return _approved("clarification", utterance, message)

        self._set_last_grounded_claim(entry, claims)
        if wants_task:
            return self._task_command_for_entry(entry, utterance)
        label = "farthest" if order == "descending" else "closest"
        ordinal_labels = {
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
        }
        ordinal_label = ordinal_labels.get(ordinal, str(ordinal))
        return ApprovedCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": (
                    "GROUNDING ANSWER\n"
                    f"ordinal={ordinal}\n"
                    f"order={order}\n"
                    f"{ordinal_label} {label}={self.domain_helper.entry_label(entry)}"
                )
            },
        )

    def _compose_extreme_task(
        self,
        utterance: str,
        claims: StationActiveClaims,
        *,
        extreme: str,
        cap_match: CapabilityMatchResult,
    ) -> ApprovedCommand:
        entries = claims.ranked_scene_doors
        distance = entries[0].distance if extreme == "closest" else entries[-1].distance
        tied = [entry for entry in entries if entry.distance == distance]
        if len(tied) > 1:
            message = (
                "CLARIFY\n"
                f"That matched multiple {extreme} doors. Which one should I use?\n"
                f"Options: {_format_targets([self.domain_helper.entry_target_dict(e) for e in tied])}"
            )
            return self._candidate_clarification_for_entries(
                utterance=utterance,
                entries=tied,
                resume_kind="task_instruction",
                message=message,
            )
        return self._task_command_for_entry(tied[0], utterance)

    def _set_last_grounded_claim(
        self,
        entry: GroundedDoorEntry,
        claims: StationActiveClaims,
    ) -> None:
        for i, candidate in enumerate(claims.ranked_scene_doors):
            if candidate.x == entry.x and candidate.y == entry.y:
                claims.last_grounded_target = candidate
                claims.last_grounded_rank = i
                return

    def _format_extreme_answer(self, claims: StationActiveClaims, normalized: str) -> str:
        entries = claims.ranked_scene_doors
        min_distance = entries[0].distance
        max_distance = entries[-1].distance
        closest = [entry for entry in entries if entry.distance == min_distance]
        farthest = [entry for entry in entries if entry.distance == max_distance]
        lines = ["GROUNDING ANSWER"]
        include_closest = "closest" in normalized or "nearest" in normalized
        include_farthest = (
            "farthest" in normalized or "furthest" in normalized or "least close" in normalized
        )
        if include_closest:
            lines.append(
                "closest="
                + ", ".join(self.domain_helper.entry_label(entry) for entry in closest)
            )
        if include_farthest:
            lines.append(
                "farthest="
                + ", ".join(self.domain_helper.entry_label(entry) for entry in farthest)
            )
        if len(lines) == 1:
            lines.append(self.domain_helper.entry_label(closest[0]))
        if include_closest and not include_farthest and len(closest) == 1:
            self._set_last_grounded_claim(closest[0], claims)
        if include_farthest and not include_closest and len(farthest) == 1:
            self._set_last_grounded_claim(farthest[0], claims)
        if len(farthest) > 1 and include_farthest:
            lines.append("tie=" + ", ".join(self.domain_helper.entry_label(entry) for entry in farthest))
        return "\n".join(lines)

    def _format_plan_extreme_answer(
        self,
        claims: StationActiveClaims,
        *,
        include_closest: bool,
        include_farthest: bool,
    ) -> str:
        entries = claims.ranked_scene_doors
        min_distance = entries[0].distance
        max_distance = entries[-1].distance
        closest = [entry for entry in entries if entry.distance == min_distance]
        farthest = [entry for entry in entries if entry.distance == max_distance]
        lines = ["GROUNDING ANSWER"]
        if include_closest:
            lines.append(
                "closest="
                + ", ".join(self.domain_helper.entry_label(entry) for entry in closest)
            )
        if include_farthest:
            lines.append(
                "farthest="
                + ", ".join(self.domain_helper.entry_label(entry) for entry in farthest)
            )
            if len(farthest) > 1:
                lines.append(
                    "tie="
                    + ", ".join(self.domain_helper.entry_label(entry) for entry in farthest)
                )
        if include_closest and not include_farthest and len(closest) == 1:
            self._set_last_grounded_claim(closest[0], claims)
        if include_farthest and not include_closest and len(farthest) == 1:
            self._set_last_grounded_claim(farthest[0], claims)
        return "\n".join(lines)

    def _format_plan_answer_fields(
        self,
        claims: StationActiveClaims,
        answer_fields: set[str],
    ) -> str:
        entries = list(claims.ranked_scene_doors)
        descending = list(reversed(entries))
        ordinal_index = {
            "first": 0,
            "second": 1,
            "third": 2,
            "fourth": 3,
            "fifth": 4,
        }
        lines = ["GROUNDING ANSWER"]

        def append_extreme(label: str, ranked: list[GroundedDoorEntry]) -> None:
            if not ranked:
                return
            distance = ranked[0].distance
            tied = [entry for entry in ranked if entry.distance == distance]
            lines.append(
                f"{label}="
                + ", ".join(self.domain_helper.entry_label(entry) for entry in tied)
            )
            if len(tied) == 1:
                self._set_last_grounded_claim(tied[0], claims)
            elif label in {"closest", "farthest"}:
                lines.append(
                    "tie="
                    + ", ".join(self.domain_helper.entry_label(entry) for entry in tied)
                )

        def append_ordinal(label: str, ranked: list[GroundedDoorEntry], ordinal: int) -> None:
            if ordinal >= len(ranked):
                lines.append(f"{label}=none")
                return
            entry = ranked[ordinal]
            tied = [candidate for candidate in ranked if candidate.distance == entry.distance]
            if len(tied) > 1:
                lines.append(
                    f"{label}=tie at distance {entry.distance}: "
                    + ", ".join(self.domain_helper.entry_label(candidate) for candidate in tied)
                )
                return
            self._set_last_grounded_claim(entry, claims)
            lines.append(f"{label}={self.domain_helper.entry_label(entry)}")

        if "closest" in answer_fields:
            append_extreme("closest", entries)
        if "farthest" in answer_fields:
            append_extreme("farthest", descending)

        for word, index in ordinal_index.items():
            closest_field = f"{word}_closest"
            farthest_field = f"{word}_farthest"
            if closest_field in answer_fields:
                append_ordinal(closest_field.replace("_", " "), entries, index)
            if farthest_field in answer_fields:
                append_ordinal(farthest_field.replace("_", " "), descending, index)

        if len(lines) == 1:
            lines.append("No composed answer fields were available.")
        return "\n".join(lines)

    def _question_override_command(self, utterance: str) -> ApprovedCommand | None:
        normalized = _normalize_utterance(utterance)
        if not _looks_like_question(normalized):
            return None
        if "delivery target" in normalized or "target" in normalized:
            return _approved("status_query", utterance, payload={"query": "delivery_target"})

        return None

    def grounded_target_summary(self, payload: dict[str, Any]) -> str:
        readiness_command = self.command_from_selector_readiness(
            payload.get("target_selector"),
            "ground target",
        )
        if readiness_command is not None:
            return readiness_command.payload.get("message", "I cannot safely execute that capability yet.")
        grounded = self.ground_target_selector(payload.get("target_selector"))
        if not grounded["ok"]:
            clarification = self.maybe_start_selector_clarification(
                utterance="ground target",
                resume_kind="ground_target_query",
                grounded=grounded,
            )
            if clarification is not None:
                return clarification.payload["message"]
            return grounded["message"]
        target = grounded["target"]
        lines = [
            "GROUNDED TARGET",
            f"target={target.get('color')} {target.get('type')}@({target.get('x')},{target.get('y')})",
        ]
        if grounded.get("distance") is not None:
            lines.append(f"distance={grounded['distance']}")
        return "\n".join(lines)

    def claim_reference_summary(self, ref_type: str) -> str:
        grounded = self._resolve_claim_reference(ref_type)
        if not grounded["ok"]:
            if grounded.get("status") == "ambiguous":
                clarification = self.maybe_start_selector_clarification(
                    utterance="claim reference",
                    resume_kind="ground_target_query",
                    grounded=grounded,
                )
                if clarification is not None:
                    return clarification.payload["message"]
            return grounded["message"]
        target = grounded["target"]
        lines = [
            "GROUNDED TARGET",
            f"target={target.get('color')} {target.get('type')}@({target.get('x')},{target.get('y')})",
        ]
        if grounded.get("distance") is not None:
            lines.append(f"distance={grounded['distance']}")
        lines.append("source=active_claims")
        return "\n".join(lines)

    def ground_target_selector(self, selector: dict[str, Any] | None) -> dict[str, Any]:
        if isinstance(selector, dict):
            if (
                selector.get("object_type") == "door"
                and selector.get("relation") == "closest"
                and selector.get("distance_metric") is None
            ):
                return {
                    "ok": False,
                    "status": "missing_required_clarifiable",
                    "missing_field": "distance_metric",
                    "supported_values": ["manhattan"],
                    "selector": dict(selector),
                    "message": self.clarification_prompt("distance_metric", ["manhattan"]),
                }
        try:
            selector = TargetSelector.from_dict(selector).__dict__
        except SchemaValidationError as exc:
            return {
                "ok": False,
                "status": "invalid_unsupported",
                "message": f"Invalid target selector: {exc}. I did not execute.",
            }
        if not isinstance(selector, dict):
            return {
                "ok": False,
                "status": "invalid_unsupported",
                "message": "No target selector was provided. I did not execute.",
            }
        if selector.get("object_type") != "door":
            return {
                "ok": False,
                "status": "invalid_unsupported",
                "message": "Only door selectors are supported right now.",
            }
        if selector.get("relation") == "closest":
            metric = selector.get("distance_metric")
            if metric != "manhattan":
                # Check whether a synthesized callable exists for this metric
                if metric is not None:
                    synth_handle = f"grounding.closest_door.{metric}.agent"
                    fn = self.capability_registry.get_synthesized_callable(synth_handle)
                    if fn is not None:
                        return self._ground_with_synthesized_callable(fn, selector)
                return {
                    "ok": False,
                    "status": "invalid_unsupported",
                    "message": (
                        f'I need distance_metric=manhattan to ground "closest" '
                        f"(metric={metric!r} has no registered primitive). "
                        "I did not execute."
                    ),
                }
            if selector.get("distance_reference") != "agent":
                return {
                    "ok": False,
                    "status": "invalid_unsupported",
                    "message": (
                        'I need distance_reference=agent to ground "closest". '
                        "Clarification is not enabled yet."
                    ),
                }

        scene = self._ensure_scene_model()
        if scene is None:
            return {
                "ok": False,
                "status": "invalid_unsupported",
                "message": "No scene data is available yet. I did not execute.",
            }

        candidates = scene.find(
            object_type="door",
            color=selector.get("color"),
            exclude_colors=selector.get("exclude_colors") or [],
        )

        if not candidates:
            return {
                "ok": False,
                "status": "no_match",
                "message": "No matching door found. I did not execute.",
            }

        relation = selector.get("relation")
        if relation == "closest":
            ranked = sorted(
                [(scene.manhattan_distance_from_agent(obj), obj) for obj in candidates],
                key=lambda pair: (pair[0], pair[1].color or "", pair[1].x, pair[1].y),
            )
            distance, target_obj = ranked[0]
            if len(ranked) > 1 and ranked[1][0] == distance:
                tied = [_scene_object_to_dict(obj) for d, obj in ranked if d == distance]
                return {
                    "ok": False,
                    "status": "ambiguous",
                    "candidates": tied,
                    "message": (
                        "That selector matched multiple closest doors: "
                        f"{_format_targets(tied)}. I did not execute."
                    ),
                }
            self._write_ranked_claims(ranked, selector)
            return {"ok": True, "target": _scene_object_to_dict(target_obj), "distance": distance}

        if relation in {None, "unique"}:
            if len(candidates) != 1:
                dicts = [_scene_object_to_dict(obj) for obj in candidates]
                return {
                    "ok": False,
                    "status": "ambiguous",
                    "candidates": dicts,
                    "message": (
                        "That selector matched multiple doors: "
                        f"{_format_targets(dicts)}. I did not execute."
                    ),
                }
            target_obj = candidates[0]
            distance = scene.manhattan_distance_from_agent(target_obj)
            return {"ok": True, "target": _scene_object_to_dict(target_obj), "distance": distance}

        return {
            "ok": False,
            "status": "invalid_unsupported",
            "message": "Unsupported target selector relation. I did not execute.",
        }

    def _ground_with_synthesized_callable(
        self,
        fn: Any,
        selector: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a synthesized grounding callable and return a grounded dict result."""
        scene = self._ensure_scene_model()
        if scene is None:
            return {
                "ok": False,
                "status": "invalid_unsupported",
                "message": "No scene data available yet. I did not execute.",
            }
        try:
            ranked = fn(scene, selector)
        except Exception as exc:  # noqa: BLE001
            return {
                "ok": False,
                "status": "invalid_unsupported",
                "message": f"Synthesized primitive raised an error: {exc}",
            }
        if not ranked:
            return {
                "ok": False,
                "status": "no_match",
                "message": "No matching door found. I did not execute.",
            }
        self._write_ranked_claims(ranked, selector)
        distance, target_obj = ranked[0]
        return {"ok": True, "target": _scene_object_to_dict(target_obj), "distance": distance}

    def command_from_selector_readiness(
        self,
        selector: dict[str, Any] | None,
        utterance: str,
    ) -> ApprovedCommand | None:
        if (
            isinstance(selector, dict)
            and selector.get("object_type") == "door"
            and selector.get("relation") == "closest"
            and selector.get("distance_metric") is None
        ):
            return None
        try:
            readiness = self.capability_registry.readiness_for_selector(selector)
        except SchemaValidationError as exc:
            return _approved("ambiguous", utterance, f"Invalid target selector: {exc}. I did not execute.")

        status = readiness["status"]
        if status == "executable":
            return None
        if status == "synthesizable_missing_primitive":
            return ApprovedCommand(
                kind="unsupported",
                utterance=utterance,
                payload={
                    "message": (
                        "I understand that grounding request, but the required primitive is "
                        f"not implemented yet: {readiness['primitive']}.\n"
                        "It is marked safe to synthesize in Phase 7.7, but synthesis is not "
                        "enabled in Phase 7.55."
                    )
                },
            )
        if status == "missing_primitive":
            return ApprovedCommand(
                kind="unsupported",
                utterance=utterance,
                payload={
                    "message": (
                        "I understand that grounding request, but a required primitive is "
                        f"missing: {readiness['primitive']}. I did not execute."
                    )
                },
            )
        if status == "unsupported":
            return ApprovedCommand(
                kind="unsupported",
                utterance=utterance,
                payload={
                    "message": (
                        "I understand that request, but the capability is unsupported: "
                        f"{readiness['reason']}"
                    )
                },
            )
        return None

    def maybe_start_selector_clarification(
        self,
        *,
        utterance: str,
        resume_kind: str,
        grounded: dict[str, Any],
    ) -> ApprovedCommand | None:
        if grounded.get("status") == "missing_required_clarifiable":
            if grounded.get("missing_field") != "distance_metric":
                return None
            self.pending_clarification = PendingClarification(
                clarification_type="target_selector_missing_field",
                original_utterance=utterance,
                resume_kind=resume_kind,
                partial_selector=dict(grounded["selector"]),
                missing_field="distance_metric",
                supported_values=list(grounded.get("supported_values", ["manhattan"])),
                **self._pending_clarification_trace(
                    utterance,
                    "target_selector_missing_field",
                ),
            )
            return _approved("clarification", utterance, grounded["message"])


        if grounded.get("status") != "ambiguous":
            return None
        candidates = grounded.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        colors = sorted(
            {
                str(candidate["color"])
                for candidate in candidates
                if candidate.get("color") in self.domain_helper.supported_colors
            }
        )
        if not colors:
            return None
        self.pending_clarification = PendingClarification(
            clarification_type="target_selector_candidate_choice",
            original_utterance=utterance,
            resume_kind=resume_kind,
            partial_selector={},
            missing_field="candidate",
            supported_values=colors,
            candidates=[dict(candidate) for candidate in candidates],
            **self._pending_clarification_trace(
                utterance,
                "target_selector_candidate_choice",
            ),
        )
        return ApprovedCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": self.clarification_prompt(
                    "candidate",
                    colors,
                    candidates=candidates,
                )
            },
        )

    def clarification_prompt(
        self,
        missing_field: str,
        supported_values: list[str],
        *,
        candidates: list[dict[str, Any]] | None = None,
    ) -> str:
        repair_note = self._repair_non_execution_note()
        if missing_field == "distance_metric":
            message = (
                "CLARIFY\n"
                'To ground "closest", which distance metric should I use?\n'
                f"Supported: {', '.join(supported_values)}"
            )
            return f"{message}\n{repair_note}" if repair_note else message
        if missing_field == "candidate":
            message = (
                "CLARIFY\n"
                "That matched multiple doors. Which one should I use?\n"
                f"Options: {_format_targets(candidates or [])}"
            )
            return f"{message}\n{repair_note}" if repair_note else message
        message = (
            "CLARIFY\n"
            f"I need {missing_field} before I can ground this selector.\n"
            f"Supported: {', '.join(supported_values)}"
        )
        return f"{message}\n{repair_note}" if repair_note else message

    def _repair_non_execution_note(self) -> str | None:
        if not any(event.success for event in self.last_repair_events):
            return None
        return (
            "Repair note: stale or invalid execution state was cleared, "
            "but I did not execute the request yet."
        )

    def execute_command(self, command: ApprovedCommand) -> str:
        return self.turn_orchestrator.execute_command(self, command)

    def _intent_for_clarification_resume(
        self,
        pending: PendingClarification,
        selector: dict[str, Any],
    ) -> OperatorIntent:
        if pending.resume_kind == "ground_target_query":
            return OperatorIntent(
                intent_type="status_query",
                status_query="ground_target",
                target_selector=selector,
                capability_status="executable",
                required_capabilities=[],
                confidence=1.0,
                reason="Clarification answer resumed a grounding query.",
            )
        if pending.resume_kind == "knowledge_update":
            return OperatorIntent(
                intent_type="knowledge_update",
                knowledge_update={"delivery_target": None},
                target_selector=selector,
                capability_status="executable",
                required_capabilities=[],
                confidence=1.0,
                reason="Clarification answer resumed a durable knowledge update.",
            )
        return OperatorIntent(
            intent_type="task_instruction",
            task_type="go_to_object",
            target_selector=selector,
            capability_status="executable",
            required_capabilities=["task.go_to_object.door"],
            confidence=1.0,
            reason="Clarification answer resumed a task instruction.",
        )

    def _clarification_blocked_message(self, graph: ReadinessGraph) -> str:
        blocking = graph.blocking_step_id or "unknown"
        return (
            "CLARIFICATION BLOCKED\n"
            f"graph_status={graph.graph_status}\n"
            f"blocking_step={blocking}\n"
            f"{graph.explanation}"
        )

    def resume_pending_clarification(self, distance_metric: str) -> str:
        pending = self.pending_clarification
        if pending is None:
            return self.status_summary(query="help")
        if pending.clarification_type != "target_selector_missing_field":
            return self.clarification_prompt(
                pending.missing_field,
                pending.supported_values,
                candidates=pending.candidates,
            )
        selector = dict(pending.partial_selector)
        selector["distance_metric"] = distance_metric
        selector["distance_reference"] = "agent"
        intent = self._intent_for_clarification_resume(pending, selector)
        self.last_operator_intent = intent
        claims_valid = self._claims_valid_for_current_environment()
        active_summary = (
            self.active_claims.compact_summary()
            if self.active_claims is not None and claims_valid
            else None
        )
        plan = build_request_plan(
            pending.original_utterance,
            intent,
            active_claims_summary=active_summary,
            environment_identity=self.current_environment_identity,
            planning_semantics=self.planning_semantics,
        )
        graph = evaluate_request_plan(
            plan,
            registry=self.capability_registry,
            active_claims=self.active_claims,
            claims_valid=claims_valid,
            environment_identity=self.current_environment_identity,
        )
        self.last_request_plan = plan
        self.last_readiness_graph = graph
        self._record_request_state(
            request_plan=plan,
            readiness_graph=graph,
            reason="clarification_resumed",
        )
        self.pending_clarification = None
        if graph.graph_status != "executable":
            return self._clarification_blocked_message(graph)

        grounded = self.ground_target_selector(selector)
        if not grounded["ok"]:
            return grounded["message"]
        target = grounded["target"]
        if pending.resume_kind == "ground_target_query":
            lines = [
                "GROUNDED TARGET",
                f"target={target.get('color')} {target.get('type')}@({target.get('x')},{target.get('y')})",
            ]
            if grounded.get("distance") is not None:
                lines.append(f"distance={grounded['distance']}")
            return "\n".join(lines)
        if pending.resume_kind == "knowledge_update":
            return self._apply_knowledge_update_from_payload(
                pending.original_utterance,
                {
                    "target_color": target["color"],
                    "target_type": target["type"],
                    "delivery_target": {
                        "color": target["color"],
                        "object_type": target["type"],
                    },
                },
                source="clarification",
            )
        instruction = f"go to the {target['color']} {target['type']}"
        result = self._run_task_from_instruction(
            pending.original_utterance,
            instruction,
            source="clarification",
            record_plan=True,
        )
        return self.result_summary(result)

    def resume_candidate_clarification(self, color: str) -> str | None:
        pending = self.pending_clarification
        if pending is None or pending.clarification_type != "target_selector_candidate_choice":
            return None
        matches = [
            candidate
            for candidate in pending.candidates
            if candidate.get("color") == color
        ]
        if len(matches) != 1:
            return self.clarification_prompt(
                pending.missing_field,
                pending.supported_values,
                candidates=pending.candidates,
            )
        target = matches[0]
        selector = {
            "object_type": target.get("type") or "door",
            "color": target.get("color"),
        }
        intent = self._intent_for_clarification_resume(pending, selector)
        self.last_operator_intent = intent
        claims_valid = self._claims_valid_for_current_environment()
        active_summary = (
            self.active_claims.compact_summary()
            if self.active_claims is not None and claims_valid
            else None
        )
        plan = build_request_plan(
            pending.original_utterance,
            intent,
            active_claims_summary=active_summary,
            environment_identity=self.current_environment_identity,
            planning_semantics=self.planning_semantics,
        )
        graph = evaluate_request_plan(
            plan,
            registry=self.capability_registry,
            active_claims=self.active_claims,
            claims_valid=claims_valid,
            environment_identity=self.current_environment_identity,
        )
        self.last_request_plan = plan
        self.last_readiness_graph = graph
        self._record_request_state(
            request_plan=plan,
            readiness_graph=graph,
            reason="candidate_clarification_resumed",
        )
        self.pending_clarification = None
        if graph.graph_status != "executable":
            return self._clarification_blocked_message(graph)
        if pending.resume_kind == "ground_target_query":
            return (
                "GROUNDED TARGET\n"
                f"target={target.get('color')} {target.get('type')}@({target.get('x')},{target.get('y')})"
            )
        if pending.resume_kind == "knowledge_update":
            return self._apply_knowledge_update_from_payload(
                pending.original_utterance,
                {
                    "target_color": target["color"],
                    "target_type": target["type"],
                    "delivery_target": {
                        "color": target["color"],
                        "object_type": target["type"],
                    },
                },
                source="clarification",
            )
        result = self._run_task_from_instruction(
            pending.original_utterance,
            f"go to the {target['color']} {target['type']}",
            source="clarification",
            record_plan=True,
        )
        return self.result_summary(result)

    def task_selector_summary(self, command: ApprovedCommand) -> str:
        grounded = self.ground_target_selector(command.payload.get("target_selector"))
        if not grounded["ok"]:
            clarification = self.maybe_start_selector_clarification(
                utterance=command.utterance,
                resume_kind="task_instruction",
                grounded=grounded,
            )
            if clarification is not None:
                return clarification.payload["message"]
            return grounded["message"]
        target = grounded["target"]
        result = self._run_task_from_instruction(
            command.utterance,
            f"go to the {target['color']} {target['type']}",
            source="selector",
            record_plan=True,
        )
        return self.result_summary(result)

    def _is_acceptance(self, normalized: str) -> bool:
        return normalized in {
            "yes", "ok", "okay", "sure", "please", "go ahead", "do it",
            "that works", "yes please", "yep", "yeah", "that would help",
            "yes that works", "sounds good", "correct", "go for it",
        }

    def _infer_offer_action(self, missing_handles: list[str]) -> str:
        """Return the registered primitive handle that best serves as an offer action."""
        return "grounding.all_doors.ranked.manhattan.agent"

    def resume_arbitration_offer(self, handle: str) -> str:
        """Execute the grounding primitive named by handle and return a display string.

        Dispatches through the capability registry — no inline logic for unregistered handles.
        """
        spec = self.capability_registry.lookup(handle)
        if spec is None:
            return f"Capability '{handle}' is not registered. I cannot fulfil that offer."
        if spec.implementation_status != "implemented":
            return (
                f"Capability '{handle}' is registered but not yet implemented "
                f"(status={spec.implementation_status}). I cannot fulfil that offer."
            )
        return self._execute_grounding_display(handle)

    def _execute_grounding_display(self, handle: str) -> str:
        """Run the named grounding primitive against the current SceneModel and return text.

        Writes results to active_claims so follow-up turns resolve against this grounding.
        """
        scene = self._ensure_scene_model()
        if scene is None:
            return "No scene data available yet."
        if "all_doors.ranked" in handle:
            doors = scene.find(object_type="door")
            if not doors:
                return "No doors visible in the current scene."
            fn = self.capability_registry.get_synthesized_callable(handle)
            if fn is not None:
                try:
                    ranked = fn(
                        scene,
                        {"object_type": "door", "color": None, "exclude_colors": []},
                    )
                except Exception as exc:  # noqa: BLE001
                    return f"Synthesized primitive '{handle}' raised an error: {exc}"
            else:
                ranked = sorted(
                    [(scene.manhattan_distance_from_agent(d), d) for d in doors],
                    key=lambda pair: (pair[0], pair[1].color or ""),
                )
            metric = self.domain_helper.metric_from_grounding_handle(handle)
            self._write_ranked_claims(
                ranked,
                {
                    "object_type": "door",
                    "relation": "all",
                    "distance_metric": metric,
                    "distance_reference": "agent",
                    "primitive": handle,
                },
            )
            return self.domain_helper.format_ranked_doors_from_entries(
                [
                    {
                        "color": d.color,
                        "object_type": d.object_type,
                        "x": d.x,
                        "y": d.y,
                        "distance": dist,
                    }
                    for dist, d in ranked
                ],
                metric=metric,
            )
        return f"No display handler implemented for grounding primitive: {handle}"

    def _execute_approved_substitute(
        self,
        handle: str,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
        decision: Any,
    ) -> ApprovedCommand:
        """Execute a registry-validated substitute primitive in place of the missing one.

        Only called when decision.safe_to_execute=True. The handle must be registered
        and implemented — anything else is a hard refuse, not a silent fallback.
        """
        spec = self.capability_registry.lookup(handle)
        if spec is None or spec.implementation_status != "implemented":
            return ApprovedCommand(
                kind="missing_skills",
                utterance=utterance,
                payload={
                    "message": (
                        f"Approved substitute '{handle}' is not implemented in the registry. "
                        "I did not execute."
                    ),
                    "match": cap_match.compact(),
                },
                capability_match=cap_match,
            )

        # Selector-grounding substitute (e.g., euclidean → manhattan for closest door).
        # Re-ground using the substitute metric then continue with the original intent path.
        if handle.startswith("grounding.closest_door."):
            parts = handle.split(".")  # grounding, closest_door, <metric>, <reference>
            metric = parts[2] if len(parts) > 2 else "manhattan"
            reference = parts[3] if len(parts) > 3 else "agent"
            base_selector: dict[str, Any] = dict(intent.target_selector or {})
            base_selector.update({
                "object_type": "door",
                "relation": "closest",
                "distance_metric": metric,
                "distance_reference": reference,
            })
            grounded = self.ground_target_selector(base_selector)
            if not grounded["ok"]:
                return _approved("ambiguous", utterance, grounded["message"])

            target = grounded["target"]
            if intent.intent_type == "task_instruction":
                return _approved("task_instruction", f"go to the {target['color']} {target['type']}")

            lines = [
                "GROUNDED TARGET (substitute)",
                f"substitute={handle}",
                f"target={target.get('color')} {target.get('type')}"
                f"@({target.get('x')},{target.get('y')})",
            ]
            if grounded.get("distance") is not None:
                lines.append(f"distance={grounded['distance']}")
            return _approved("clarification", utterance, "\n".join(lines), capability_match=cap_match)


        # Display-grounding substitute (e.g., all_doors.ranked for a ranked query).
        if "all_doors.ranked" in handle:
            result_text = self._execute_grounding_display(handle)
            return _approved("clarification", utterance, result_text, capability_match=cap_match)


        # Unknown primitive type — refuse clearly rather than silently doing nothing.
        msg = decision.operator_message or (
            f"Substitute '{handle}' was approved but no execution path exists for "
            f"primitive type '{spec.layer}'. I did not execute."
        )
        return _approved("missing_skills", utterance, payload={"message": msg, "match": cap_match.compact()}, capability_match=cap_match)


    def _ensure_scene_model(self) -> SceneModel | None:
        """Return the current SceneModel, building it via an idle sense pass if needed.

        Uses the existing Sense path (parse_grid_objects) against the live adapter.
        Never resets the env — if no adapter exists, creates a temporary one only
        for the initial observation, then closes it.
        """
        if self.memory.scene_model is not None:
            scene = self.memory.scene_model
            identity = self._build_environment_identity(scene)
            current = self.current_environment_identity
            if current is None or current.fingerprint() == identity.fingerprint():
                self.current_environment_identity = identity
                return scene
            self.log("environment_identity_changed")
            self.active_claims = None
            self.representation.clear_scene_model()
            self.last_environment_invalidation_reason = "environment_identity_changed"
        self.substrate.sense_idle_scene(self.sense, seed=self.seed)
        scene = self.memory.scene_model
        if scene is not None:
            self.current_environment_identity = self._build_environment_identity(scene)
        return scene

    def _task_family_for_env(self) -> str | None:
        if "GoToDoor" in self.env_id:
            return "go_to_object"
        if "MiniGrid" in self.env_id:
            return "minigrid"
        return None

    def _scene_state_fingerprint(self, scene: SceneModel) -> str:
        return (
            f"agent=({scene.agent_x},{scene.agent_y},{scene.agent_dir});"
            f"step={scene.step_count}"
        )

    def _build_environment_identity(self, scene: SceneModel) -> EnvironmentIdentity:
        objects = sorted(
            (
                {
                    "type": obj.object_type,
                    "color": obj.color,
                    "x": obj.x,
                    "y": obj.y,
                    "state": obj.state,
                }
                for obj in scene.objects
            ),
            key=lambda item: (
                str(item["type"]),
                str(item["color"]),
                int(item["x"]),
                int(item["y"]),
                str(item["state"]),
            ),
        )
        return EnvironmentIdentity(
            env_id=self.env_id or scene.env_id,
            seed=self.seed if self.seed is not None else scene.seed,
            grid_width=scene.grid_width,
            grid_height=scene.grid_height,
            mission=self._last_scene_mission(),
            task_family=self._task_family_for_env(),
            scene_fingerprint=self._scene_state_fingerprint(scene),
            summary={
                "object_count": len(objects),
                "objects": objects,
            },
        )

    def _update_current_environment_identity(self, scene: SceneModel) -> bool:
        identity = self._build_environment_identity(scene)
        current = self.current_environment_identity
        changed = current is not None and current.fingerprint() != identity.fingerprint()
        if changed:
            self.log("environment_identity_changed")
            self.active_claims = None
            self.representation.clear_scene_model()
            self.last_environment_invalidation_reason = "environment_identity_changed"
        self.current_environment_identity = identity
        return changed

    def _last_scene_mission(self) -> str | None:
        sample = self.memory.episodic_memory.get("last_world_sample")
        if not isinstance(sample, dict):
            return None
        mission = sample.get("mission")
        return mission if isinstance(mission, str) else None

    def _claims_valid_for_current_environment(self, scene: SceneModel | None = None) -> bool:
        if self.active_claims is None:
            return False
        scene = scene or self.memory.scene_model
        if scene is None:
            return False
        identity = self._build_environment_identity(scene)
        current = self.current_environment_identity
        if current is not None and current.fingerprint() != identity.fingerprint():
            self.log("environment_identity_changed")
            self.active_claims = None
            self.representation.clear_scene_model()
            self.current_environment_identity = identity
            self.last_environment_invalidation_reason = "environment_identity_changed"
            return False
        self.current_environment_identity = identity
        return self.active_claims.is_valid_for(scene, environment_identity=identity)

    def _write_ranked_claims(
        self,
        ranked_pairs: list[tuple[float, SceneObject]],
        selector: dict[str, Any],
    ) -> None:
        scene = self.memory.scene_model
        if scene is None or not ranked_pairs:
            return
        metric = selector.get("distance_metric") or selector.get("metric")
        provenance = selector.get("primitive")
        entries = [
            GroundedDoorEntry(
                color=obj.color, x=obj.x, y=obj.y, distance=dist,
                metric=metric, provenance=provenance,
            )
            for dist, obj in ranked_pairs
        ]
        identity = self.current_environment_identity or self._build_environment_identity(scene)
        self.current_environment_identity = identity
        self.last_environment_invalidation_reason = None
        self.active_claims = StationActiveClaims(
            scene_fingerprint=(scene.agent_x, scene.agent_y, scene.step_count),
            ranked_scene_doors=entries,
            last_grounded_target=entries[0],
            last_grounded_rank=0,
            last_grounding_query=dict(selector),
            environment_fingerprint=identity.fingerprint(),
            environment_identity=identity,
        )

    def _resolve_claim_reference(self, ref_type: str) -> dict[str, Any]:
        scene = self.memory.scene_model
        if self.last_environment_invalidation_reason == "environment_identity_changed":
            return {
                "ok": False,
                "message": (
                    "Scene has changed since the last grounding "
                    "(environment_identity_changed). Please re-ground."
                ),
            }
        if self.active_claims is None:
            return {
                "ok": False,
                "message": self._missing_claim_reference_prerequisite(ref_type),
            }
        if scene is None:
            return {"ok": False, "message": "No scene data available."}
        if not self._claims_valid_for_current_environment(scene):
            self.active_claims = None
            return {
                "ok": False,
                "message": "Scene has changed since the last grounding. Please re-ground.",
            }

        if ref_type == "next_closest":
            entry, rank = self.active_claims.next_ranked()
            if entry is None:
                return {"ok": False, "message": "No further doors in the ranked list."}
            self.active_claims.last_grounded_target = entry
            self.active_claims.last_grounded_rank = rank
            return {"ok": True, "target": entry.as_dict(), "distance": entry.distance}

        if ref_type == "other_door":
            others = self.active_claims.other_doors()
            if not others:
                return {"ok": False, "message": "No other doors available."}
            if len(others) > 1:
                dicts = [e.as_dict() for e in others]
                return {
                    "ok": False,
                    "status": "ambiguous",
                    "candidates": dicts,
                    "message": (
                        "Multiple other doors: "
                        f"{_format_targets(dicts)}. Which one did you mean?"
                    ),
                }
            entry = others[0]
            rank = next(
                i for i, d in enumerate(self.active_claims.ranked_scene_doors)
                if d.x == entry.x and d.y == entry.y
            )
            self.active_claims.last_grounded_target = entry
            self.active_claims.last_grounded_rank = rank
            return {"ok": True, "target": entry.as_dict(), "distance": entry.distance}

        return {"ok": False, "message": f"Unknown claim reference: {ref_type}"}

    def _missing_claim_reference_prerequisite(self, ref_type: str) -> str:
        if ref_type == "next_closest":
            return self._grounding_prerequisite_message(
                relation="ranked",
                object_type="door",
                metric="manhattan",
                reference="agent",
            )
        if ref_type == "other_door":
            return self._grounding_prerequisite_message(
                relation="selected",
                object_type="door",
                metric="manhattan",
                reference="agent",
            )
        return "No active grounding claims. Ground the relevant target first."

    def _grounding_prerequisite_message(
        self,
        *,
        relation: str,
        object_type: str,
        metric: str | None,
        reference: str | None,
    ) -> str:
        metric_part = f" by {metric} distance" if metric is not None else ""
        reference_part = " from you" if reference == "agent" else ""
        if relation == "ranked":
            needed = f"a ranked {object_type} grounding"
            query = f"which {object_type} is closest{metric_part}{reference_part}?"
        else:
            needed = f"a selected {object_type} grounding"
            query = f"which {object_type} should I use{metric_part}{reference_part}?"
        return f"No active grounding claims for that reference. First ask: {query} ({needed})."

    def compose_known_task(self, instruction: str) -> TaskRequest:
        parsed = _parse_go_to_object_utterance(instruction)
        if parsed is None:
            raise ValueError(f"Unsupported known task instruction: {instruction}")
        return TaskRequest(
            instruction=f"go to the {parsed['color']} {parsed['object_type']}",
            task_type="go_to_object",
            params=canonical_task_params(
                color=parsed["color"],
                object_type=parsed["object_type"],
            ),
            source="operator_station_known_family",
        )

    def compose_known_procedure(self, task: TaskRequest) -> ProcedureRecipe:
        recipe = self.memory.understanding.get(task.task_type, {}).get("default_recipe")
        if not recipe:
            raise ValueError(f"No default recipe for task type: {task.task_type}")
        return ProcedureRecipe(
            task_type=task.task_type,
            steps=list(recipe),
            source="operator_station_known_family",
            compiler_backend=self.compiler.active_backend,
            validated=True,
            rationale="Known operator-station go_to_object family.",
        )

    def open_preview(self) -> None:
        if self.render_mode != "human":
            self.log("render preview skipped because render mode is not human")
            return
        self.log("opening idle render preview")
        self.substrate.open_preview(seed=self.seed)

    def _pump_render_window(self) -> None:
        self.substrate.pump_render_window()

    def close_preview(self) -> None:
        if not self.substrate.has_preview_window():
            return
        self.log("closing idle render preview")
        self.substrate.close_preview()

    def close_task_window(self) -> None:
        if not self.substrate.has_task_window():
            return
        self.log("closing previous task render window")
        self.substrate.close_task_window()

    def _record_rejected_raw_execution_attempt(self, value: Any) -> None:
        if not isinstance(value, str):
            return
        intent = OperatorIntent(
            intent_type="unsupported",
            canonical_instruction=value,
            capability_status="unsupported",
            required_capabilities=["task.execution_ticket.required"],
            confidence=1.0,
            reason="Raw task execution requires an ExecutionTicket.",
        )
        self._record_request_plan(value, intent)

    def run_task(self, ticket: ExecutionTicket) -> dict[str, Any]:
        if not isinstance(ticket, ExecutionTicket):
            self._record_rejected_raw_execution_attempt(ticket)
            raise TypeError("run_task requires an ExecutionTicket")
        return self._run_task_with_ticket(ticket)

    def _run_task_with_ticket(self, ticket: ExecutionTicket) -> dict[str, Any]:
        self.last_execution_ticket = ticket
        instruction = ticket.instruction
        self.active_claims = None
        self.log(f"starting task: {instruction}")
        self.log("compiling task/procedure and warming JIT templates before motion")
        task_override = None
        procedure_override = None
        if _parse_go_to_object_utterance(instruction) is not None:
            task_override = self.compose_known_task(instruction)
            procedure_override = self.compose_known_procedure(task_override)
            self.log(
                "composed known task locally: "
                f"type={task_override.task_type} params={task_override.params}"
            )
            self.log(f"composed procedure steps={procedure_override.steps}")
        self.last_result = self.substrate.run_task_episode(
            instruction=instruction,
            compiler_name=self.compiler_name,
            compiler=self.compiler,
            seed=self.seed,
            max_loops=self.max_loops,
            memory=self.memory,
            plan_cache=self.plan_cache,
            progress_callback=self._progress_callback,
            task_override=task_override,
            procedure_override=procedure_override,
        )
        if self.last_result["final_state"]["task_complete"]:
            self.store_successful_task_memory(self.last_result)
        self.log(
            "rendered cached runtime finished: "
            f"task_complete={self.last_result['final_state']['task_complete']} "
            f"runtime_llm_calls={self.last_result['runtime_llm_calls_during_render']} "
            f"cache_misses={self.last_result['cache_miss_during_render']}"
        )
        return self.last_result

    def _progress_callback(self, event: str, payload: dict[str, Any]) -> None:
        self._pump_render_window()
        if event == "task_compile_started":
            self.log(f"LLM task compile starting: {payload['instruction']}")
        elif event == "procedure_compile_started":
            self.log(
                "LLM procedure compile starting: "
                f"type={payload['task_type']} params={payload['params']}"
            )
        elif event == "task_compiled":
            task = payload["task"]
            self.log(
                "compiled task: "
                f"type={task['task_type']} params={task['params']}"
            )
        elif event == "procedure_ready":
            procedure = payload["procedure"]
            self.log(
                "procedure ready: "
                f"cache={payload['cache_status']} steps={procedure['steps']}"
            )
        elif event == "readiness_checked":
            readiness = payload["readiness"]
            self.log(
                "readiness: "
                f"status={readiness['status']} "
                f"missing_actions={readiness['missing_actions']} "
                f"missing_evidence={readiness['missing_evidence']}"
            )
        elif event == "prewarm_started":
            self.log(f"JIT prewarm starting for steps={payload['procedure']}")
        elif event == "prewarm_template":
            self.log(
                "warming template: "
                f"{payload['template_type']}:{payload['label']}"
            )
        elif event == "prewarm_finished":
            self.log(f"JIT prewarm finished: cache_entries={payload['cache_entries']}")
        elif event == "runtime_started":
            self.log("rendered cached runtime starting")

    def resolve_task_instruction(self, utterance: str) -> str | None:
        normalized = _normalize_utterance(utterance)
        if "delivery target" in normalized:
            return self._instruction_from_delivery_target()

        if normalized in {"go there again", "go to the same door", "go to same door"}:
            return self._instruction_from_last_target()

        if normalized in {"repeat the last task", "repeat the previous task"}:
            last_instruction = self.memory.episodic_memory.get("last_successful_instruction")
            if isinstance(last_instruction, str) and last_instruction:
                return last_instruction

            last_task = self.memory.episodic_memory.get("last_task")
            if isinstance(last_task, dict):
                instruction = last_task.get("instruction")
                if isinstance(instruction, str) and instruction:
                    return instruction
            return None

        return _canonicalize_task_instruction(utterance)

    def _instruction_from_delivery_target(self) -> str | None:
        delivery_target = self.memory.knowledge.get("delivery_target")
        if not isinstance(delivery_target, dict):
            return None

        color = delivery_target.get("color")
        object_type = delivery_target.get("object_type")
        if not color or not object_type:
            return None
        return f"go to the {color} {object_type}"

    def _instruction_from_last_target(self) -> str | None:
        last_target = self.memory.episodic_memory.get("last_target")
        if not isinstance(last_target, dict):
            return None

        color = last_target.get("color")
        object_type = last_target.get("object_type")
        if not color or not object_type:
            return None
        return f"go to the {color} {object_type}"

    def missing_reference_summary(self, utterance: str) -> str:
        normalized = _normalize_utterance(utterance)
        if "delivery target" in normalized:
            return (
                "I do not have a delivery target yet.\n"
                "Tell me something like: the red door is the delivery target"
            )
        if normalized in {"repeat the last task", "repeat the previous task"}:
            return "I do not have a previous successful task yet."
        if normalized in {"go there again", "go to the same door", "go to same door"}:
            return "I do not have a previous target yet."
        return "I could not resolve that instruction yet."

    def store_successful_task_memory(self, result: dict[str, Any]) -> None:
        task = result.get("task")
        if not isinstance(task, dict):
            return

        params = task.get("params")
        if not isinstance(params, dict):
            return

        color = params.get("color")
        object_type = params.get("object_type")
        instruction = task.get("instruction")
        if not color or not object_type or not isinstance(instruction, str):
            return

        last_target = {
            "color": color,
            "object_type": object_type,
        }
        self.memory.update_episodic_memory("last_target", last_target)
        self.memory.update_episodic_memory(
            "last_task",
            {
                "instruction": instruction,
                "task_type": task.get("task_type"),
                "params": dict(params),
            },
        )
        self.memory.update_episodic_memory("last_successful_instruction", instruction)

    def close(self) -> None:
        self.close_preview()
        self.close_task_window()

    def teach_concept(self, name: str, utterance: str) -> str:
        """Teach a named concept and pre-compile its plan into the reuse cache."""
        plan = None
        try:
            intent = self.compiler.compile_operator_intent(
                utterance,
                memory=self.memory,
                scene_summary=None,
                capability_manifest=self.capability_registry.compact_summary(),
                active_claims_summary=None,
            )
            from .request_planner import build_request_plan
            from .readiness_graph import evaluate_request_plan

            candidate = build_request_plan(
                utterance,
                intent,
                active_claims_summary=None,
                environment_identity=self.current_environment_identity,
                planning_semantics=self.planning_semantics,
            )
            graph = evaluate_request_plan(
                candidate,
                registry=self.capability_registry,
                active_claims=None,
                claims_valid=False,
                environment_identity=self.current_environment_identity,
            )
            if graph.graph_status == "executable":
                plan = candidate
                self.request_plan_reuse_cache.store(plan)
        except Exception as exc:
            self.log(f"concept teach planning failed: {exc}")
            return (
                "CONCEPT NOT STORED\n"
                f"name={name!r}\n"
                f"reason=planning_failed:{type(exc).__name__}"
            )

        concept = self.knowledge_base.teach(name, utterance, plan=plan)
        plan_status = "plan pre-compiled and cached" if plan is not None else "no plan (will compile on first recall)"
        self.log(f"concept taught: name={concept.name!r} utterance={concept.utterance!r} type={concept.concept_type} {plan_status}")
        result = (
            f"CONCEPT STORED\n"
            f"name={concept.name!r}\n"
            f"type={concept.concept_type}\n"
            f"utterance={concept.utterance!r}\n"
            f"plan_cached={plan is not None}"
        )
        if concept.concept_type == "procedure":
            result += f"\nsteps={concept.steps}"
        return result

    def forget_concept(self, name: str) -> str:
        if self.knowledge_base.forget(name):
            self.log(f"concept forgotten: {name!r}")
            return f"CONCEPT FORGOTTEN: {name!r}"
        return f"CONCEPT NOT FOUND: {name!r} (nothing forgotten)"

    def _try_natural_sequence(self, utterance: str) -> ApprovedCommand | None:
        """Detect natural-language sequential patterns like 'do X then Y' or 'X followed by Y'.

        Single-word tokens → procedure_execute (named KB concepts).
        Multi-word tokens → sequence_execute (raw task utterances).
        Returns None if no sequential pattern is detected.
        """
        normalized = _normalize_utterance(utterance)
        parts = re.split(r"\b(?:and\s+then|then|followed\s+by)\b", normalized)
        if len(parts) < 2:
            return None
        _pre = re.compile(r"^\s*(?:(?:do|execute|run|perform|also)\s+)?(?:a\s+|the\s+|an\s+)?(?:first\s+)?")
        _suf = re.compile(r"\s+(?:first|next|also|too)\s*$")
        cleaned = [_suf.sub("", _pre.sub("", p)).strip() for p in parts]
        cleaned = [p for p in cleaned if p]
        if len(cleaned) < 2:
            return None
        seq = self.knowledge_base._is_sequence(",".join(cleaned))
        if seq is not None:
            return ApprovedCommand(command_type="procedure_execute", utterance=utterance, payload={"steps": seq})
        # Multi-word parts that aren't named concepts → raw utterance sequence
        if all(cleaned):
            return ApprovedCommand(command_type="sequence_execute", utterance=utterance, payload={"steps": cleaned})
        return None

    def _command_from_concept(self, utterance: str) -> ApprovedCommand | None:
        """Resolve a bare concept name (or comma-separated sequence) to an ApprovedCommand."""
        normalized = _normalize_utterance(utterance.strip())

        # Check if utterance is an anonymous comma-separated sequence of known concepts
        seq = self.knowledge_base._is_sequence(utterance.strip())
        if seq is not None:
            self.log(f"anonymous procedure detected: {seq}")
            return _approved("procedure_execute", utterance, payload={"steps": seq})


        # Check for natural-language "X then Y" / "X followed by Y" sequences
        nat_cmd = self._try_natural_sequence(utterance)
        if nat_cmd is not None:
            self.log(f"natural-language sequence detected: kind={nat_cmd.kind} steps={nat_cmd.payload.get('steps')}")
            return nat_cmd

        concept = self.knowledge_base.recall(normalized) or self.knowledge_base.recall(utterance.strip())
        if concept is None:
            # Try stripping execution-verb prefixes: "execute bingo", "run scout", "do alpha"
            m = re.match(r"^(?:execute|run|do)\s+(.+)$", normalized)
            if m:
                concept = self.knowledge_base.recall(m.group(1).strip())
        if concept is None:
            return None

        self.log(f"concept {concept.name!r} resolved to: {concept.utterance!r} (type={concept.concept_type})")

        # Procedure-type concept — unpack its steps
        if concept.concept_type == "procedure":
            return _approved("procedure_execute", utterance, payload={"steps": list(concept.steps)})


        # Atomic concept — expand and dispatch
        expanded = concept.utterance
        expanded_command = classify_utterance(expanded)
        if expanded_command.kind == "unresolved":
            expanded_command = self.command_from_llm_intent(expanded)
        return expanded_command

    def _build_procedure_request_plan(
        self,
        steps: list[str],
        original_utterance: str,
    ) -> RequestPlan | None:
        """Build a multi-step RequestPlan from a list of atomic concept names."""
        from .schemas import RequestPlan, RequestPlanStep
        import uuid

        plan_steps: list[RequestPlanStep] = []
        prev_step_id: str | None = None
        for idx, concept_name in enumerate(steps):
            concept = self.knowledge_base.recall(concept_name)
            if concept is None:
                self.log(f"procedure build failed: concept '{concept_name}' not found")
                return None
            if concept.concept_type == "procedure":
                self.log(f"procedure build failed: nested procedure '{concept_name}' not allowed")
                return None
            try:
                task = self.compose_known_task(concept.utterance)
            except ValueError:
                self.log(f"procedure build failed: cannot resolve handle for '{concept.utterance}'")
                return None
            step_id = f"step_{idx}"
            plan_steps.append(
                RequestPlanStep(
                    step_id=step_id,
                    layer="task",
                    operation="execute",
                    required_handle="task.go_to_object.door",
                    implementation_status="implemented",
                    constraints={"utterance": concept.utterance},
                    depends_on=[prev_step_id] if prev_step_id is not None else [],
                )
            )
            prev_step_id = step_id

        return RequestPlan(
            request_id=str(uuid.uuid4()),
            original_utterance=original_utterance,
            objective_type="task",
            objective_summary=f"procedure: {' → '.join(steps)}",
            steps=plan_steps,
            expected_response="execute_task",
        )

    def _run_procedure(self, steps: list[str], original_utterance: str) -> str:
        """Execute a procedure: build multi-step RequestPlan, gate via ReadinessGraph, run steps."""
        plan = self._build_procedure_request_plan(steps, original_utterance)
        if plan is None:
            return (
                "PROCEDURE ERROR\n"
                "Could not build execution plan — one or more steps could not be resolved.\n"
                "Ensure all concept names are defined as atomic concepts before teaching a procedure."
            )

        claims_valid = self._claims_valid_for_current_environment()
        readiness_graph = evaluate_request_plan(
            plan,
            registry=self.capability_registry,
            active_claims=self.active_claims,
            claims_valid=claims_valid,
            environment_identity=self.current_environment_identity,
        )
        self.last_request_plan = plan
        self.last_readiness_graph = readiness_graph

        if readiness_graph.graph_status != "executable":
            blocking = readiness_graph.blocking_step_id or "unknown"
            return (
                f"PROCEDURE BLOCKED\n"
                f"graph_status={readiness_graph.graph_status}\n"
                f"blocking_step={blocking}\n"
                f"{readiness_graph.explanation}"
            )

        results: list[str] = []
        for step_idx, step_id_name in enumerate(steps):
            concept = self.knowledge_base.recall(step_id_name)
            if concept is None:
                return f"PROCEDURE ERROR\nConcept '{step_id_name}' disappeared during execution."
            self.log(f"procedure step {step_idx + 1}/{len(steps)}: {concept.name!r} → {concept.utterance!r}")
            result = self._run_task_from_instruction(
                concept.utterance,
                concept.utterance,
                source="procedure_step",
                record_plan=False,
            )
            results.append(self.result_summary(result))

        step_labels = " → ".join(steps)
        return f"PROCEDURE COMPLETE ({step_labels})\n" + "\n---\n".join(results)

    def _build_sequence_request_plan(
        self,
        utterance_steps: list[str],
        original_utterance: str,
    ) -> RequestPlan | None:
        """Build a multi-step RequestPlan from a list of raw task utterances."""
        from .schemas import RequestPlan, RequestPlanStep
        import uuid

        plan_steps: list[RequestPlanStep] = []
        prev_step_id: str | None = None
        for idx, step_utterance in enumerate(utterance_steps):
            try:
                task = self.compose_known_task(step_utterance)
            except ValueError:
                self.log(f"sequence build failed: cannot resolve handle for '{step_utterance}'")
                return None
            step_id = f"step_{idx}"
            plan_steps.append(
                RequestPlanStep(
                    step_id=step_id,
                    layer="task",
                    operation="execute",
                    required_handle="task.go_to_object.door",
                    implementation_status="implemented",
                    constraints={"utterance": step_utterance},
                    depends_on=[prev_step_id] if prev_step_id is not None else [],
                )
            )
            prev_step_id = step_id

        return RequestPlan(
            request_id=str(uuid.uuid4()),
            original_utterance=original_utterance,
            objective_type="task",
            objective_summary=f"sequence: {' → '.join(utterance_steps)}",
            steps=plan_steps,
            expected_response="execute_task",
        )

    def _run_sequence(self, utterance_steps: list[str], original_utterance: str) -> str:
        """Execute a sequence of raw task utterances: build RequestPlan, gate, run each step."""
        plan = self._build_sequence_request_plan(utterance_steps, original_utterance)
        if plan is None:
            return (
                "SEQUENCE ERROR\n"
                "Could not build execution plan — one or more steps could not be compiled.\n"
                "Ensure each step is a valid task instruction (e.g. 'go to the red door')."
            )

        claims_valid = self._claims_valid_for_current_environment()
        readiness_graph = evaluate_request_plan(
            plan,
            registry=self.capability_registry,
            active_claims=self.active_claims,
            claims_valid=claims_valid,
            environment_identity=self.current_environment_identity,
        )
        self.last_request_plan = plan
        self.last_readiness_graph = readiness_graph

        if readiness_graph.graph_status != "executable":
            blocking = readiness_graph.blocking_step_id or "unknown"
            return (
                f"SEQUENCE BLOCKED\n"
                f"graph_status={readiness_graph.graph_status}\n"
                f"blocking_step={blocking}\n"
                f"{readiness_graph.explanation}"
            )

        results: list[str] = []
        for step_idx, step_utterance in enumerate(utterance_steps):
            self.log(f"sequence step {step_idx + 1}/{len(utterance_steps)}: {step_utterance!r}")
            result = self._run_task_from_instruction(
                step_utterance,
                step_utterance,
                source="sequence_step",
                record_plan=False,
            )
            results.append(self.result_summary(result))

        step_labels = " → ".join(utterance_steps)
        return f"PROCEDURE COMPLETE ({step_labels})\n" + "\n---\n".join(results)

    def _run_motor_command(self, action_name: str, count: int, original_utterance: str) -> str:
        """Authorize and execute an explicit low-level motor primitive."""
        if not self.substrate.is_action_known(action_name):
            return (
                f"ERROR: Unknown motor action '{action_name}'.\n"
                f"Known actions: {self.substrate.known_action_names()}"
            )
        ticket = self._raw_motor_ticket_for_command(
            original_utterance,
            action_name,
            count,
            source="operator",
        )
        return self._run_motor_with_ticket(ticket)

    def _run_motor_with_ticket(self, ticket: RawMotorTicket) -> str:
        if not isinstance(ticket, RawMotorTicket):
            raise TypeError("_run_motor_with_ticket requires a RawMotorTicket")
        self.last_raw_motor_ticket = ticket
        action_name = ticket.action_name
        count = ticket.repeat_count
        if not self.substrate.is_action_known(action_name):
            return (
                f"ERROR: Unknown motor action '{action_name}'.\n"
                f"Known actions: {self.substrate.known_action_names()}"
            )
        actions = [action_name] * count
        self.log(f"motor_execute: {action_name} × {count}")
        result = self.substrate.run_motor_actions(seed=self.seed, actions=actions)
        self.last_result = result
        label = action_name.replace("_", " ")
        return f"MOTOR COMPLETE ({label} × {count}): {result.get('steps_taken', count)} steps executed."

    def _run_motor_sequence_command(
        self, sequence: list[dict[str, Any]], original_utterance: str
    ) -> str:
        """Execute an ordered list of motor actions on the persistent session adapter."""
        parts: list[str] = []
        for step in sequence:
            action_name = step["action"]
            count = step["count"]
            self.log(f"motor_sequence step: {action_name} × {count}")
            result_text = self._run_motor_command(action_name, count, original_utterance)
            parts.append(result_text)
        return "MOTOR SEQUENCE\n" + "\n".join(parts)

    def _run_mission(self, steps: list[str], original_utterance: str) -> str:
        """Execute a mission contract — ordered tasks with abort-on-failure semantics."""
        results: list[str] = []
        for step_idx, step_utterance in enumerate(steps):
            self.log(f"mission step {step_idx + 1}/{len(steps)}: {step_utterance!r}")
            ticket = self._execution_ticket_for_instruction(
                step_utterance,
                step_utterance,
                source="mission_step",
                record_plan=False,
            )
            result = self._run_task_with_ticket(ticket)
            summary = self.result_summary(result)
            results.append(summary)
            if not result.get("final_state", {}).get("task_complete", False):
                return (
                    f"MISSION ABORTED (step {step_idx + 1} failed: {step_utterance!r})\n"
                    + "\n---\n".join(results)
                )
        step_labels = " → ".join(steps)
        self.last_result = result  # type: ignore[possibly-undefined]
        return f"MISSION COMPLETE ({step_labels})\n" + "\n---\n".join(results)

    def concepts_summary(self) -> str:
        concepts = self.knowledge_base.all_concepts()
        if not concepts:
            return "CONCEPTS\n(none — teach a concept with: remember <name> means <utterance>)"
        lines = ["CONCEPTS"]
        for c in concepts:
            plan_tag = " [plan cached]" if c.plan is not None else ""
            type_tag = f" [{c.concept_type}]" if c.concept_type != "atomic" else ""
            lines.append(f"  {c.name!r} → {c.utterance!r}{type_tag}{plan_tag}  (recalled {c.recall_count}x)")
        return "\n".join(lines)

    def apply_knowledge_update(self, ticket: MemoryWriteTicket) -> str:
        if not isinstance(ticket, MemoryWriteTicket):
            raise TypeError("apply_knowledge_update requires a MemoryWriteTicket")
        self.last_memory_write_ticket = ticket
        self.log("updating durable target knowledge")
        if any(write.scope != "knowledge" for write in ticket.writes):
            unsupported = sorted({write.scope for write in ticket.writes if write.scope != "knowledge"})
            raise ValueError(f"Unsupported memory write scope: {', '.join(unsupported)}")
        self.representation.apply_memory_write_ticket(ticket)
        return (
            "KNOWLEDGE UPDATED\n"
            f"delivery_target={self.memory.knowledge.get('delivery_target')}"
        )

    def reset(self, *, clear_memory: bool = False) -> str:
        self.log("resetting station state")
        self.pending_clarification = None
        self.pending_synthesis_proposal = None
        self.active_claims = None
        self.last_environment_invalidation_reason = None
        self.last_request_plan = None
        self.last_readiness_graph = None
        self.last_execution_ticket = None
        self.last_memory_write_ticket = None
        self.last_raw_motor_ticket = None
        self.last_operator_intent = None
        self.last_cortical_envelope = None
        self.last_approved_command = None
        self.last_command_result = None
        self.memory.reset_episode()
        self.last_result = None
        if clear_memory:
            self.representation.clear_operator_knowledge()
            return "RESET: episodic state and durable knowledge cleared"
        return "RESET: episodic state cleared; durable knowledge kept"

    def status_summary(self, *, query: str = "status") -> str:
        if query == "last_run":
            return self.last_run_summary()
        if query == "scene":
            return self.scene_summary()
        if query == "last_target":
            return self.last_target_summary()
        if query == "delivery_target":
            return self.delivery_target_summary()
        if query == "help":
            return self.capability_registry.help_text()
        if query == "concepts":
            return self.concepts_summary()
        knowledge = self.memory.knowledge
        return (
            "STATUS\n"
            f"env_id={self.env_id}\n"
            f"seed={self.seed}\n"
            f"compiler={self.compiler.active_backend}\n"
            f"delivery_target={knowledge.get('delivery_target')}\n"
            f"last_instruction={knowledge.get('last_instruction')}\n"
            f"last_task_complete={self._last_task_complete()}"
        )

    def scene_summary(self) -> str:
        scene = self._ensure_scene_model()
        if scene is None:
            return f"SCENE\nenv_id={self.env_id}\nseed={self.seed}\nstatus=no scene data"
        doors = scene.find(object_type="door")
        door_strs = [f"{d.color} door@({d.x},{d.y})" for d in doors]
        return (
            "SCENE\n"
            f"env_id={self.env_id}\n"
            f"seed={self.seed}\n"
            f"agent=({scene.agent_x},{scene.agent_y}) dir={scene.agent_dir}\n"
            f"source={scene.source}\n"
            f"doors={', '.join(door_strs) if door_strs else 'none'}\n"
            f"object_count={len(scene.objects)}"
        )

    def last_run_summary(self) -> str:
        if self.last_result is None:
            return "LAST RUN\nnone"
        return "LAST RUN\n" + self.result_summary(self.last_result)

    def last_target_summary(self) -> str:
        last_target = self.memory.episodic_memory.get("last_target")
        if not isinstance(last_target, dict):
            return "LAST TARGET: none"

        instruction = self.memory.episodic_memory.get("last_successful_instruction")
        return (
            "LAST TARGET\n"
            f"color={last_target.get('color')}\n"
            f"object_type={last_target.get('object_type')}\n"
            f"instruction={instruction}"
        )

    def delivery_target_summary(self) -> str:
        delivery_target = self.memory.knowledge.get("delivery_target")
        if not isinstance(delivery_target, dict):
            return "DELIVERY TARGET: none"
        return (
            "DELIVERY TARGET\n"
            f"color={delivery_target.get('color')}\n"
            f"object_type={delivery_target.get('object_type')}"
        )

    def cache_summary(self) -> str:
        summary = self.plan_cache.summary(include_entries=False)
        return (
            "CACHE\n"
            f"enabled={self.plan_cache.enabled}\n"
            f"entries={len(self.plan_cache.entries)}\n"
            f"hits={summary['hits']}\n"
            f"misses={summary['misses']}\n"
            f"llm_calls_saved={summary['llm_calls_saved']}"
        )

    def result_summary(self, result: dict[str, Any]) -> str:
        final_record = _final_record(result.get("loop_records", []))
        lines = [
            "RUN COMPLETE" if result["final_state"]["task_complete"] else "RUN FAILED",
            f"task_complete={result['final_state']['task_complete']}",
            f"runtime_llm_calls_during_render={result['runtime_llm_calls_during_render']}",
            f"cache_miss_during_render={result['cache_miss_during_render']}",
            f"final_skill_plan={final_record.get('skill_plan') if final_record else None}",
            f"final_report_status={_final_report_status(final_record)}",
        ]
        last_report = result["final_state"].get("last_report")
        if isinstance(last_report, dict) and last_report.get("reason"):
            lines.append(f"reason={last_report['reason']}")
            progress = last_report.get("progress", {})
            available_targets = progress.get("available_targets")
            if available_targets is not None:
                lines.append(f"available_targets={_format_targets(available_targets)}")
        return "\n".join(lines)

    def _last_task_complete(self) -> bool | None:
        if self.last_result is None:
            return None
        return bool(self.last_result["final_state"]["task_complete"])


def _final_record(loop_records: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    records = [record for record in loop_records if record.get("skill_plan") is not None]
    return records[-1] if records else None


def _final_report_status(final_record: dict[str, Any] | None) -> str | None:
    if final_record is None or final_record.get("report") is None:
        return None
    return final_record["report"].get("status")


def _format_targets(targets: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{target.get('color')} {target.get('type')}@({target.get('x')},{target.get('y')})"
        for target in targets
    )


def run_repl(session: OperatorStationSession) -> int:
    print(session.startup())
    try:
        while True:
            try:
                utterance = input("READY> ")
            except EOFError:
                print()
                return 0
            response = session.handle_utterance(utterance)
            if response == "QUIT":
                print("BYE")
                return 0
            print(response)
    finally:
        session.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the JEENOM natural-language operator station.")
    parser.add_argument("--env-id", default="MiniGrid-GoToDoor-8x8-v0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compiler", choices=["smoke_test", "llm"], default="llm")
    parser.add_argument("--render-mode", choices=["none", "human", "rgb_array"], default="human")
    parser.add_argument("--max-loops", type=int, default=128)
    parser.add_argument("--memory-root", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    session = OperatorStationSession(
        env_id=args.env_id,
        seed=args.seed,
        compiler_name=args.compiler,
        memory_root=args.memory_root,
        render_mode=args.render_mode,
        max_loops=args.max_loops,
        use_cache=not args.no_cache,
        verbose=not args.quiet,
    )
    return run_repl(session)
