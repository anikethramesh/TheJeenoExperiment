from __future__ import annotations

import argparse
import re
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from .capability_arbitrator import ArbitratorBackend, build_arbitrator
from .capability_matcher import CapabilityMatchResult, CapabilityMatcher, default_matcher
from .capability_registry import CapabilityRegistry
from .primitive_synthesizer import SynthesizerBackend, build_synthesizer
from .primitive_validator import PrimitiveValidator, default_validator
from .intent_verifier import IntentVerificationResult, IntentVerifier, default_verifier
from .cortex import Cortex
from .llm_compiler import CompilerBackend, SmokeTestCompiler, build_compiler, canonical_task_params
from .memory import OperationalMemory
from .minigrid_envs import ensure_custom_minigrid_envs_registered
from .minigrid_adapter import MiniGridAdapter
from .plan_cache import PlanCache
from .primitive_library import TASK_PRIMITIVES
from .schemas import (
    ArbitrationTrace,
    GroundedDoorEntry,
    OperatorIntent,
    ProcedureRecipe,
    SceneModel,
    SceneObject,
    SchemaValidationError,
    StationActiveClaims,
    TargetSelector,
    TaskRequest,
)
from . import run_demo
from .sense import MiniGridSense
from .spine import MiniGridSpine


SUPPORTED_COLORS = ("red", "green", "blue", "yellow", "purple", "grey", "gray")


@dataclass
class OperatorCommand:
    kind: str
    utterance: str
    payload: dict[str, Any] = field(default_factory=dict)
    capability_match: Any = None  # CapabilityMatchResult | None


@dataclass
class PendingClarification:
    clarification_type: str
    original_utterance: str
    resume_kind: str
    partial_selector: dict[str, Any]
    missing_field: str
    supported_values: list[str]
    candidates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PendingSynthesisProposal:
    handle: str
    original_utterance: str
    intent: Any
    cap_match: Any
    similar_handles: list[str]
    proposed_description: str | None = None
    proposed_condition: dict[str, Any] | None = None


def _normalize_color(color: str) -> str:
    return "grey" if color == "gray" else color


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


def classify_utterance(utterance: str) -> OperatorCommand:
    text = utterance.strip()
    normalized = _normalize_utterance(text)

    if normalized in {"quit", "exit", "bye"}:
        return OperatorCommand(kind="quit", utterance=text)

    if normalized in {"cancel", "never mind", "nevermind"}:
        return OperatorCommand(kind="cancel", utterance=text)

    if normalized in {"reset", "reset episode"}:
        return OperatorCommand(kind="reset", utterance=text, payload={"clear_memory": False})
    if normalized in {"clear memory", "forget memory", "forget everything", "reset memory"}:
        return OperatorCommand(kind="reset", utterance=text, payload={"clear_memory": True})

    if normalized in {"show cache", "cache", "cache status", "what is cached?"}:
        return OperatorCommand(kind="cache_query", utterance=text)

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
        return OperatorCommand(kind="status_query", utterance=text, payload={"query": "scene"})

    if normalized in {
        "what was the last target?",
        "what was the last target",
        "what was the previous target?",
        "what was the previous target",
        "previous target",
        "last target",
    }:
        return OperatorCommand(kind="status_query", utterance=text, payload={"query": "last_target"})

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
        return OperatorCommand(kind="status_query", utterance=text, payload={"query": query})

    if normalized in {
        "go there again",
        "go to the same door",
        "go to same door",
        "repeat the last task",
        "repeat the previous task",
    }:
        return OperatorCommand(kind="task_instruction", utterance=text)

    if "delivery target" in normalized and re.search(
        r"^(go to|reach|find|get to|head to|navigate to) (the )?delivery target$",
        normalized,
    ):
        return OperatorCommand(kind="task_instruction", utterance=text)

    target_fact = _parse_target_fact(normalized)
    if target_fact is not None:
        return OperatorCommand(kind="knowledge_update", utterance=text, payload=target_fact)

    if _parse_exact_go_to_object_utterance(normalized) is not None:
        return OperatorCommand(kind="task_instruction", utterance=text)

    return OperatorCommand(kind="unresolved", utterance=text)


def _parse_target_fact(normalized: str) -> dict[str, str] | None:
    color_pattern = "|".join(SUPPORTED_COLORS)
    patterns = [
        rf"^(?:please )?(?:your |the |my |our )?delivery target is (?:the )?(?P<color>{color_pattern}) (?P<object_type>door)$",
        rf"^(?:please )?(?:the )?(?P<color>{color_pattern}) (?P<object_type>door) is (?:your |the |my |our )?delivery target$",
        rf"^(?:please )?target is (?:the )?(?P<color>{color_pattern}) (?P<object_type>door)$",
        rf"^(?:please )?remember (?:that )?(?:the )?(?P<color>{color_pattern}) (?P<object_type>door)$",
        rf"^(?:please )?set (?:the )?delivery target to (?:the )?(?P<color>{color_pattern}) (?P<object_type>door)$",
        rf"^(?:please )?use (?:the )?(?P<color>{color_pattern}) (?P<object_type>door) as (?:your |the |my |our )?delivery target$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if match:
            return {
                "target_color": _normalize_color(match.group("color")),
                "target_type": match.group("object_type"),
                "delivery_target": {
                    "color": _normalize_color(match.group("color")),
                    "object_type": match.group("object_type"),
                },
            }
    return None


def _canonicalize_task_instruction(utterance: str) -> str:
    match = _parse_go_to_object_utterance(utterance)
    if not match:
        return utterance

    verb = match["verb"]
    if verb in {"go the", "head to", "navigate to"}:
        verb = "go to"
    return f"{verb} the {match['color']} {match['object_type']}"


def _parse_go_to_object_utterance(utterance: str) -> dict[str, str] | None:
    normalized = _normalize_utterance(utterance)
    color_pattern = "|".join(SUPPORTED_COLORS)
    match = re.search(
        rf"\b(?P<verb>go to|go the|reach|find|get to|head to|navigate to)\s+"
        rf"(?:the )?(?P<color>{color_pattern}) (?P<object_type>door)\b",
        normalized,
    )
    if not match:
        return None
    return {
        "verb": match.group("verb"),
        "color": _normalize_color(match.group("color")),
        "object_type": match.group("object_type"),
    }


def _parse_exact_go_to_object_utterance(utterance: str) -> dict[str, str] | None:
    normalized = _normalize_utterance(utterance)
    color_pattern = "|".join(SUPPORTED_COLORS)
    match = re.match(
        rf"^(?P<verb>go to|reach|find|get to|head to|navigate to)\s+"
        rf"(?:the )?(?P<color>{color_pattern}) (?P<object_type>door)$",
        normalized,
    )
    if not match:
        return None
    return {
        "verb": match.group("verb"),
        "color": _normalize_color(match.group("color")),
        "object_type": match.group("object_type"),
    }


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
    ) -> None:
        ensure_custom_minigrid_envs_registered()
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
        self.capability_registry = CapabilityRegistry.minigrid_default()
        self.cortex = Cortex(self.memory, self.compiler, plan_cache=self.plan_cache)
        self.sense = MiniGridSense(self.memory, self.compiler, plan_cache=self.plan_cache)
        self.spine = MiniGridSpine(self.memory, None, self.compiler, plan_cache=self.plan_cache)
        self.prewarm_compiler = SmokeTestCompiler()
        self.prewarm_cortex = Cortex(
            self.memory,
            self.prewarm_compiler,
            plan_cache=self.plan_cache,
        )
        self.prewarm_sense = MiniGridSense(
            self.memory,
            self.prewarm_compiler,
            plan_cache=self.plan_cache,
        )
        self.prewarm_spine = MiniGridSpine(
            self.memory,
            None,
            self.prewarm_compiler,
            plan_cache=self.plan_cache,
        )
        self.startup_prewarm_summary: dict[str, Any] | None = None
        self.preview_adapter: MiniGridAdapter | None = None
        self.task_adapter: MiniGridAdapter | None = None
        self.pending_clarification: PendingClarification | None = None
        self.pending_synthesis_proposal: PendingSynthesisProposal | None = None
        self.active_claims: StationActiveClaims | None = None
        self.arbitrator: ArbitratorBackend = build_arbitrator(compiler_name)
        self.last_arbitration_trace: ArbitrationTrace | None = None
        self.synthesizer: SynthesizerBackend = build_synthesizer(compiler_name)
        self.validator: PrimitiveValidator = default_validator

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
        summary = run_demo.prewarm_jit_cache(
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
            if color in SUPPORTED_COLORS and object_type == "door":
                return f"go to the {_normalize_color(color)} door"

        target_color = self.memory.knowledge.get("target_color")
        target_type = self.memory.knowledge.get("target_type")
        if target_color in SUPPORTED_COLORS and target_type == "door":
            return f"go to the {_normalize_color(target_color)} door"

        return "go to the red door"

    def handle_utterance(self, utterance: str) -> str:
        command = classify_utterance(utterance)
        if self.pending_synthesis_proposal is not None:
            return self.handle_pending_synthesis_proposal(utterance, command)
        if self.pending_clarification is not None:
            pending_response = self.handle_pending_clarification(utterance, command)
            if pending_response is not None:
                return pending_response
        if command.kind == "unresolved":
            command = self._command_from_active_claim_text(utterance) or command
            if command.kind == "unresolved":
                self.log("deterministic fast path unresolved; compiling operator intent")
                command = self.command_from_llm_intent(utterance)
        self.log(f"classified utterance as {command.kind}")
        if command.kind == "quit":
            self.pending_clarification = None
            return "QUIT"
        if command.kind == "cancel":
            self.pending_clarification = None
            return "CANCELLED: pending clarification cleared"
        if command.kind == "reset":
            return self.reset(clear_memory=bool(command.payload.get("clear_memory")))
        if command.kind == "clarification":
            return command.payload["message"]
        if command.kind == "cache_query":
            return self.cache_summary()
        if command.kind == "status_query":
            return self.status_summary(query=command.payload.get("query", "status"))
        if command.kind == "ground_target_query":
            return self.grounded_target_summary(command.payload)
        if command.kind == "task_selector":
            return self.task_selector_summary(command)
        if command.kind == "knowledge_update":
            return self.apply_knowledge_update(command.payload)
        if command.kind == "claim_reference":
            return self.claim_reference_summary(command.payload.get("ref_type", ""))
        if command.kind == "synthesis_proposal":
            return command.payload["message"]
        if command.kind in {"accept_proposal", "reject_proposal"}:
            return self.status_summary(query="help")
        if command.kind == "missing_skills":
            return command.payload.get("message", "I do not have the required capabilities.")
        if command.kind == "synthesizable":
            return command.payload.get("message", "That capability is not yet implemented.")
        if command.kind == "task_instruction":
            instruction = self.resolve_task_instruction(command.utterance)
            if instruction is None:
                return self.missing_reference_summary(command.utterance)
            if instruction != command.utterance:
                self.log(f"resolved task instruction to: {instruction}")
            result = self.run_task(instruction)
            return self.result_summary(result)
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
        return self.status_summary(query="help")

    def command_from_llm_intent(
        self,
        utterance: str,
        *,
        pending_proposal: dict[str, Any] | None = None,
    ) -> OperatorCommand:
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
    ) -> OperatorCommand:
        if intent.grounding_query_plan is not None:
            plan_command = self._command_from_grounding_query_plan(utterance, intent)
            if plan_command is not None:
                return plan_command

        # ── Proactive Intent Signal Verification (Phase 7.595) ───────────────
        # Extracts semantic signals from the utterance regardless of LLM output.
        # Injects required_capabilities the LLM failed to declare.
        # Catches intent inversion (farthest→closest) and silent degradation.
        intent, verif_result = default_verifier.enrich(utterance, intent)
        if verif_result.injected_handles:
            self.log(f"intent verifier injected: {verif_result.summary()}")

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
            return OperatorCommand(kind=intent.intent_type, utterance=utterance)

        if intent.intent_type == "claim_reference":
            if intent.claim_reference == "threshold_filter":
                # Route through capability matching → arbitration → synthesis pipeline.
                # The plan carries threshold, comparison, and metric for execution.
                if cap_match.verdict in {"synthesizable", "missing_skills"}:
                    return self._arbitrate_gap(utterance, intent, cap_match)
                # Already registered — execute directly.
                return self._dispatch_claims_filter(utterance, intent, cap_match)
            return OperatorCommand(
                kind="claim_reference",
                utterance=utterance,
                payload={"ref_type": intent.claim_reference or ""},
            )

        if intent.intent_type in {"unsupported", "ambiguous"}:
            # Route through the arbitrator — it has access to the full capability manifest
            # and SceneModel API surface, so it can recognise synthesisable spatial
            # computations (e.g. threshold filtering) that the LLM compiler missed.
            return self._arbitrate_gap(utterance, intent, cap_match)

        if intent.intent_type == "quit":
            return OperatorCommand(kind="quit", utterance=utterance)

        if intent.intent_type == "reset":
            return OperatorCommand(
                kind="reset",
                utterance=utterance,
                payload={"clear_memory": bool(intent.clear_memory)},
            )

        if intent.intent_type == "cache_query":
            return OperatorCommand(kind="cache_query", utterance=utterance)

        if intent.intent_type == "status_query":
            if intent.status_query == "ground_target" and intent.target_selector is not None:
                return OperatorCommand(
                    kind="ground_target_query",
                    utterance=utterance,
                    payload={"target_selector": intent.target_selector},
                )
            return OperatorCommand(
                kind="status_query",
                utterance=utterance,
                payload={"query": intent.status_query or "status"},
            )

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
                    return OperatorCommand(
                        kind="ambiguous",
                        utterance=utterance,
                        payload={"message": grounded["message"]},
                    )
                target = grounded["target"]
                return OperatorCommand(
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
                return OperatorCommand(
                    kind="knowledge_update",
                    utterance=utterance,
                    payload={
                        "target_color": None,
                        "target_type": None,
                        "delivery_target": None,
                    },
                )
            if not isinstance(delivery_target, dict):
                return OperatorCommand(kind="unsupported", utterance=utterance)
            return OperatorCommand(
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
                    return OperatorCommand(
                        kind="ambiguous",
                        utterance=utterance,
                        payload={"message": grounded["message"]},
                    )
                target = grounded["target"]
                instruction = f"go to the {target['color']} {target['type']}"
            elif isinstance(intent.target, dict):
                if "closest" in _normalize_utterance(utterance):
                    return OperatorCommand(
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
                    return OperatorCommand(kind="unsupported", utterance=utterance)
                instruction = intent.canonical_instruction or f"go to the {color} {object_type}"
            else:
                return OperatorCommand(kind="unsupported", utterance=utterance)
            return OperatorCommand(kind="task_instruction", utterance=instruction)

        return OperatorCommand(kind="unsupported", utterance=utterance)

    def _command_from_grounding_query_plan(
        self,
        utterance: str,
        intent: OperatorIntent,
    ) -> OperatorCommand | None:
        plan = intent.grounding_query_plan
        if plan is None:
            return None

        invalid_reason = self._validate_grounding_query_plan_preserves_utterance(
            utterance,
            plan,
        )
        if invalid_reason is not None:
            if plan.get("metric") is None:
                clarification = self._maybe_start_semantic_metric_clarification(
                    utterance=utterance,
                    reason=invalid_reason,
                )
                if clarification is not None:
                    return clarification
            return OperatorCommand(
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
                return OperatorCommand(
                    kind="missing_skills",
                    utterance=utterance,
                    payload={"message": f"Missing primitive: {handle}"},
                    capability_match=cap_match,
                )
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
    ) -> OperatorCommand | None:
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
        )
        return OperatorCommand(
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
    ) -> str | None:
        normalized = _normalize_utterance(utterance)
        constraints = {str(c).lower() for c in plan.get("preserved_constraints", [])}
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

        has_farthest = any(
            term in normalized
            for term in ("farthest", "furthest", "most distant", "least close")
        )
        has_closest = any(
            term in normalized
            for term in ("closest", "nearest", "shortest")
        )
        answer_fields = {str(f).lower() for f in plan.get("answer_fields", [])}
        operation = plan.get("operation")
        order = plan.get("order")

        if has_farthest and "farthest" not in constraints:
            if not (order == "descending" or "farthest" in answer_fields):
                return "The utterance asks for farthest, but the plan does not preserve descending/farthest semantics."
        if has_closest and "closest" not in constraints:
            if not (order == "ascending" or "closest" in answer_fields or operation == "rank"):
                return "The utterance asks for closest, but the plan does not preserve ascending/closest semantics."
        if "euclidean" in normalized and plan.get("metric") != "euclidean":
            return "The utterance specifies Euclidean distance, but the plan does not."
        if "manhattan" in normalized and plan.get("metric") != "manhattan":
            return "The utterance specifies Manhattan distance, but the plan does not."

        color = self._color_reference_in_utterance(normalized)
        if color is None:
            bare_color_pattern = "|".join(SUPPORTED_COLORS)
            match = re.search(rf"\b(?P<color>{bare_color_pattern})\b", normalized)
            if match:
                color = _normalize_color(match.group("color"))
        if color is not None and plan.get("color") not in {None, color}:
            return f"The utterance mentions {color}, but the plan targets {plan.get('color')}."
        return None

    def _compose_claim_reference_query_plan(
        self,
        utterance: str,
        intent: OperatorIntent,
    ) -> OperatorCommand:
        scene = self.memory.scene_model
        if scene is None:
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={"message": "No scene data available for that reference."},
            )
        if self.active_claims is None:
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={"message": "No active grounded target for that reference."},
            )
        if not self.active_claims.is_valid_for(scene):
            self.active_claims = None
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={"message": "Scene has changed since that grounding. Please re-ground."},
            )
        entry = self.active_claims.last_grounded_target
        wants_task = intent.intent_type == "task_instruction" or self._utterance_requests_navigation(
            _normalize_utterance(utterance)
        )
        if wants_task:
            return self._task_command_for_entry(entry, utterance)
        return OperatorCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": (
                    "GROUNDING ANSWER\n"
                    f"target={self._entry_label(entry)}\n"
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
    ) -> OperatorCommand:
        wants_task = intent.intent_type == "task_instruction" or self._utterance_requests_navigation(
            _normalize_utterance(utterance)
        )
        claims = self._ensure_ranked_door_claims(plan.get("primitive_handle"))
        if isinstance(claims, str):
            return OperatorCommand(
                kind="missing_skills",
                utterance=utterance,
                payload={"message": claims, "match": cap_match.compact()},
                capability_match=cap_match,
            )

        operation = plan.get("operation")
        answer_fields = {str(f).lower() for f in plan.get("answer_fields", [])}
        distance_value = plan.get("distance_value")
        color = plan.get("color")
        ordinal = plan.get("ordinal")
        order = plan.get("order")

        if operation in {"rank", "list"}:
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self._format_ranked_doors_from_claims(
                        claims,
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
            return OperatorCommand(
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
            return OperatorCommand(
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
            return OperatorCommand(
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
            return OperatorCommand(
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
                return OperatorCommand(
                    kind="ambiguous",
                    utterance=utterance,
                    payload={"message": f"No unique {color} door is visible. I did not execute."},
                )
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self._format_color_plan_answer(
                        color,
                        matches,
                        answer_fields,
                    )
                },
                capability_match=cap_match,
            )

        return OperatorCommand(
            kind="ambiguous",
            utterance=utterance,
            payload={"message": "I could not compose a result from the semantic query plan."},
        )

    def _arbitrate_gap(
        self,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
    ) -> OperatorCommand:
        # Only pass implemented handles — synthesizable ones are not alternatives
        available = [
            h for h in self.capability_registry.primitive_names()
            if (spec := self.capability_registry.lookup(h)) is not None
            and spec.implementation_status == "implemented"
        ]
        scene_summary = self._build_scene_summary_for_arbitrator()
        decision = self.arbitrator.arbitrate(
            utterance=utterance,
            intent_type=intent.intent_type,
            required_capabilities=intent.required_capabilities,
            missing_handles=cap_match.missing,
            synthesizable_handles=cap_match.synthesizable_handles,
            available_handles=available,
            scene_summary=scene_summary,
        )
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
            return OperatorCommand(
                kind="synthesizable",
                utterance=utterance,
                payload={
                    "message": decision.operator_message,
                    "match": cap_match.compact(),
                },
                capability_match=cap_match,
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
            )
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={"message": decision.clarification_prompt},
                capability_match=cap_match,
            )

        if decision.decision_type == "substitute" and decision.suggested_handle:
            if not decision.safe_to_execute:
                msg = decision.operator_message or (
                    f"Suggested substitute: {decision.suggested_handle}, "
                    "but marked not safe to execute."
                )
                return OperatorCommand(
                    kind="missing_skills",
                    utterance=utterance,
                    payload={"message": msg, "match": cap_match.compact()},
                    capability_match=cap_match,
                )
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
        return OperatorCommand(
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
    ) -> OperatorCommand | None:
        """Attempt to synthesize, validate, and register a missing grounding primitive.

        Returns an OperatorCommand on success (re-routes to grounding) or None on failure
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
            return OperatorCommand(
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
            return OperatorCommand(
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
    ) -> OperatorCommand:
        """Run a freshly synthesized grounding primitive and continue with original intent."""
        fn = self.capability_registry.get_synthesized_callable(handle)
        if fn is None:
            return OperatorCommand(
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
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={"message": "No scene data available yet. I did not execute."},
            )

        selector = dict(intent.target_selector or {})
        doors_in_scene = [o for o in scene.objects if o.object_type == "door"]
        self.log(
            f"synthesis grounding: scene has {len(scene.objects)} objects "
            f"({len(doors_in_scene)} doors), selector={selector}"
        )
        try:
            ranked = fn(scene, selector)
        except Exception as exc:  # noqa: BLE001
            return OperatorCommand(
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
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={"message": "No matching doors found. I did not execute."},
            )

        self._write_ranked_claims(ranked, {**selector, "primitive": handle})
        distance, target_obj = ranked[0]
        target = _scene_object_to_dict(target_obj)

        if intent.intent_type == "task_instruction":
            instruction = f"go to the {target['color']} {target['type']}"
            self.log(
                f"synthesis grounding: resolved to {instruction} "
                f"(distance={distance:.2f} via {handle})"
            )
            return OperatorCommand(kind="task_instruction", utterance=instruction)

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
        return OperatorCommand(
            kind="clarification",
            utterance=utterance,
            payload={"message": "\n".join(lines)},
            capability_match=cap_match,
        )

    # ── Claims-filter synthesis path ──────────────────────────────────────────

    def _dispatch_claims_filter(
        self,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
    ) -> OperatorCommand:
        """Execute a registered claims-filter primitive against active_claims."""
        plan = intent.grounding_query_plan or {}
        handle = plan.get("primitive_handle") or ""
        if not handle:
            # Infer handle from metric in plan
            metric = plan.get("metric", "euclidean") or "euclidean"
            handle = f"claims.filter.threshold.{metric}"

        fn = self.capability_registry.get_synthesized_callable(handle)
        if fn is None:
            return OperatorCommand(
                kind="missing_skills",
                utterance=utterance,
                payload={"message": f"Claims-filter primitive '{handle}' is not registered."},
                capability_match=cap_match,
            )
        return self._execute_synthesized_claims_filter(handle, fn, utterance, intent, cap_match)

    def _try_synthesize_claims_filter(
        self,
        handle: str,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
    ) -> OperatorCommand | None:
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
                return OperatorCommand(
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
    ) -> OperatorCommand:
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
                return OperatorCommand(
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
            return OperatorCommand(
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
            return OperatorCommand(
                kind="synthesizable",
                utterance=utterance,
                payload={
                    "message": f"Claims-filter '{handle}' raised an error: {exc}.",
                    "match": cap_match.compact(),
                },
                capability_match=cap_match,
            )

        if not isinstance(filtered, list):
            return OperatorCommand(
                kind="synthesizable",
                utterance=utterance,
                payload={"message": f"Claims-filter '{handle}' returned {type(filtered).__name__}, expected list."},
                capability_match=cap_match,
            )

        metric_label = condition.get("metric") or "distance"
        comparison = condition["comparison"]
        threshold = condition["threshold"]

        if not filtered:
            known = ", ".join(
                f"{e.color}@{e.distance:.1f}" for e in entries
            )
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={
                    "message": (
                        f"No doors with {metric_label} distance {comparison} {threshold}. "
                        f"Known distances: {known}"
                    ),
                },
            )

        if len(filtered) == 1:
            entry = filtered[0]
            self.active_claims.last_grounded_target = entry
            self.active_claims.last_grounded_rank = next(
                (i for i, e in enumerate(entries) if e.x == entry.x and e.y == entry.y), 0
            )
            if intent.intent_type == "task_instruction":
                instruction = f"go to the {entry.color} {entry.object_type}"
                return OperatorCommand(kind="task_instruction", utterance=instruction)
            lines = [
                f"GROUNDED TARGET (claims filter: {handle})",
                f"target={entry.color} {entry.object_type}@({entry.x},{entry.y})",
                f"distance={entry.distance:.2f} [{metric_label}]",
                f"filter={comparison} {threshold}",
                "source=active_claims",
            ]
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={"message": "\n".join(lines)},
                capability_match=cap_match,
            )

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
    ) -> OperatorCommand:
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
        return OperatorCommand(
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
            "order": None,
            "ordinal": None,
            "color": None,
            "exclude_colors": [],
            "distance_value": (
                int(threshold_float) if threshold_float.is_integer() else threshold_float
            ),
            "comparison": comparison,
            "tie_policy": "clarify",
            "answer_fields": ["target", "distance"],
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
        command: OperatorCommand,
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
        if self.active_claims is not None and self.active_claims.is_valid_for(scene):
            summary["active_claims"] = self.active_claims.compact_summary()
        return summary

    def _try_compose_grounding_result(
        self,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
        verif_result: IntentVerificationResult,
    ) -> OperatorCommand | None:
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
            return OperatorCommand(
                kind="missing_skills",
                utterance=utterance,
                payload={"message": claims, "match": cap_match.compact()},
                capability_match=cap_match,
            )
        if not claims.ranked_scene_doors:
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={"message": "No visible doors are available to compose from."},
            )

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
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={
                    "message": self._format_ranked_doors_from_claims(
                        claims,
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
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={"message": self._format_extreme_answer(claims, normalized)},
                capability_match=cap_match,
            )

        if "closest" in normalized or "nearest" in normalized:
            if wants_task:
                return self._compose_extreme_task(
                    utterance,
                    claims,
                    extreme="closest",
                    cap_match=cap_match,
                )
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={"message": self._format_extreme_answer(claims, normalized)},
                capability_match=cap_match,
            )

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
            and self.active_claims.is_valid_for(scene)
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
                "distance_metric": self._metric_from_grounding_handle(handle),
                "distance_reference": "agent",
                "primitive": handle,
            },
        )
        if self.active_claims is None:
            return "Unable to write active grounding claims."
        return self.active_claims

    def _command_from_active_claim_text(self, utterance: str) -> OperatorCommand | None:
        normalized = _normalize_utterance(utterance)
        if not self._utterance_requests_navigation(normalized):
            return None
        color = self._color_reference_in_utterance(normalized)
        if color is None:
            return None
        claims = self._ensure_ranked_door_claims()
        if isinstance(claims, str):
            return None
        matches = [entry for entry in claims.ranked_scene_doors if entry.color == color]
        if len(matches) != 1:
            return None
        return OperatorCommand(
            kind="task_instruction",
            utterance=f"go to the {matches[0].color} door",
        )

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

    def _color_reference_in_utterance(self, normalized: str) -> str | None:
        color_pattern = "|".join(SUPPORTED_COLORS)
        match = re.search(rf"\b(?P<color>{color_pattern})\s+(?:one|door)\b", normalized)
        if match:
            return _normalize_color(match.group("color"))
        return None

    def _entry_target_dict(self, entry: GroundedDoorEntry) -> dict[str, Any]:
        return {"type": "door", "color": entry.color, "x": entry.x, "y": entry.y}

    def _entry_label(self, entry: GroundedDoorEntry) -> str:
        return f"{entry.color} door@({entry.x},{entry.y}) distance={entry.distance}"

    def _task_command_for_entry(self, entry: GroundedDoorEntry, utterance: str) -> OperatorCommand:
        return OperatorCommand(
            kind="task_instruction",
            utterance=f"go to the {entry.color} door",
        )

    def _candidate_clarification_for_entries(
        self,
        *,
        utterance: str,
        entries: list[GroundedDoorEntry],
        resume_kind: str,
        message: str,
    ) -> OperatorCommand:
        self.pending_clarification = PendingClarification(
            clarification_type="target_selector_candidate_choice",
            original_utterance=utterance,
            resume_kind=resume_kind,
            partial_selector={},
            missing_field="candidate",
            supported_values=sorted(
                str(entry.color) for entry in entries if entry.color is not None
            ),
            candidates=[self._entry_target_dict(entry) for entry in entries],
        )
        return OperatorCommand(
            kind="clarification",
            utterance=utterance,
            payload={"message": message},
        )

    def _compose_distance_reference(
        self,
        utterance: str,
        claims: StationActiveClaims,
        distance: int,
        *,
        wants_task: bool,
    ) -> OperatorCommand:
        matches = [entry for entry in claims.ranked_scene_doors if entry.distance == distance]
        if not matches:
            return OperatorCommand(
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
                f"Options: {_format_targets([self._entry_target_dict(e) for e in matches])}"
            )
            if wants_task:
                return self._candidate_clarification_for_entries(
                    utterance=utterance,
                    entries=matches,
                    resume_kind="task_instruction",
                    message=message,
                )
            return OperatorCommand(kind="clarification", utterance=utterance, payload={"message": message})
        entry = matches[0]
        if wants_task:
            return self._task_command_for_entry(entry, utterance)
        return OperatorCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": (
                    "GROUNDING ANSWER\n"
                    f"distance={distance}\n"
                    f"target={self._entry_label(entry)}"
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
    ) -> OperatorCommand | None:
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
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={"message": "There are not enough visible doors for that ordinal request."},
            )
        entry = ranked[rank]
        tied = [item for item in ranked if item.distance == entry.distance]
        if len(tied) > 1:
            message = (
                "CLARIFY\n"
                f"That ordinal falls inside a distance tie at distance {entry.distance}. "
                "Which one should I use?\n"
                f"Options: {_format_targets([self._entry_target_dict(e) for e in tied])}"
            )
            if wants_task:
                return self._candidate_clarification_for_entries(
                    utterance=utterance,
                    entries=tied,
                    resume_kind="task_instruction",
                    message=message,
                )
            return OperatorCommand(kind="clarification", utterance=utterance, payload={"message": message})
        self._set_last_grounded_claim(entry, claims)
        if wants_task:
            return self._task_command_for_entry(entry, utterance)
        return OperatorCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": (
                    "GROUNDING ANSWER\n"
                    f"{ordinal} {direction}={self._entry_label(entry)}"
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
    ) -> OperatorCommand:
        ranked = list(claims.ranked_scene_doors)
        if order == "descending":
            ranked = list(reversed(ranked))
        rank = ordinal - 1
        if rank >= len(ranked):
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={"message": "There are not enough visible doors for that ordinal request."},
            )
        entry = ranked[rank]
        tied = [item for item in ranked if item.distance == entry.distance]
        if len(tied) > 1:
            message = (
                "CLARIFY\n"
                f"That ordinal falls inside a distance tie at distance {entry.distance}. "
                "Which one should I use?\n"
                f"Options: {_format_targets([self._entry_target_dict(e) for e in tied])}"
            )
            if wants_task:
                return self._candidate_clarification_for_entries(
                    utterance=utterance,
                    entries=tied,
                    resume_kind="task_instruction",
                    message=message,
                )
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={"message": message},
            )
        self._set_last_grounded_claim(entry, claims)
        if wants_task:
            return self._task_command_for_entry(entry, utterance)
        label = "farthest" if order == "descending" else "closest"
        return OperatorCommand(
            kind="clarification",
            utterance=utterance,
            payload={
                "message": (
                    "GROUNDING ANSWER\n"
                    f"ordinal={ordinal}\n"
                    f"order={order}\n"
                    f"{ordinal} {label}={self._entry_label(entry)}"
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
    ) -> OperatorCommand:
        entries = claims.ranked_scene_doors
        distance = entries[0].distance if extreme == "closest" else entries[-1].distance
        tied = [entry for entry in entries if entry.distance == distance]
        if len(tied) > 1:
            message = (
                "CLARIFY\n"
                f"That matched multiple {extreme} doors. Which one should I use?\n"
                f"Options: {_format_targets([self._entry_target_dict(e) for e in tied])}"
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
                + ", ".join(self._entry_label(entry) for entry in closest)
            )
        if include_farthest:
            lines.append(
                "farthest="
                + ", ".join(self._entry_label(entry) for entry in farthest)
            )
        if len(lines) == 1:
            lines.append(self._entry_label(closest[0]))
        if include_closest and not include_farthest and len(closest) == 1:
            self._set_last_grounded_claim(closest[0], claims)
        if include_farthest and not include_closest and len(farthest) == 1:
            self._set_last_grounded_claim(farthest[0], claims)
        if len(farthest) > 1 and include_farthest:
            lines.append("tie=" + ", ".join(self._entry_label(entry) for entry in farthest))
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
                + ", ".join(self._entry_label(entry) for entry in closest)
            )
        if include_farthest:
            lines.append(
                "farthest="
                + ", ".join(self._entry_label(entry) for entry in farthest)
            )
            if len(farthest) > 1:
                lines.append(
                    "tie="
                    + ", ".join(self._entry_label(entry) for entry in farthest)
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
                + ", ".join(self._entry_label(entry) for entry in tied)
            )
            if len(tied) == 1:
                self._set_last_grounded_claim(tied[0], claims)
            elif label in {"closest", "farthest"}:
                lines.append(
                    "tie="
                    + ", ".join(self._entry_label(entry) for entry in tied)
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
                    + ", ".join(self._entry_label(candidate) for candidate in tied)
                )
                return
            self._set_last_grounded_claim(entry, claims)
            lines.append(f"{label}={self._entry_label(entry)}")

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

    def _format_color_plan_answer(
        self,
        color: str,
        matches: list[GroundedDoorEntry],
        answer_fields: set[str],
    ) -> str:
        if not matches:
            if "exists" in answer_fields:
                return f"GROUNDING ANSWER\nexists=false\ncolor={color}\nobject_type=door"
            return f"No matching {color} door found."
        if self.active_claims is not None:
            self._set_last_grounded_claim(matches[0], self.active_claims)
        if "exists" in answer_fields and "distance" not in answer_fields:
            return f"GROUNDING ANSWER\nexists=true\ntarget={self._entry_label(matches[0])}"
        if "distance" in answer_fields:
            lines = ["GROUNDING ANSWER"]
            for entry in matches:
                lines.append(f"target={self._entry_label(entry)}")
            return "\n".join(lines)
        return "GROUNDING ANSWER\n" + "\n".join(
            f"target={self._entry_label(entry)}" for entry in matches
        )

    def _format_ranked_doors_from_claims(
        self,
        claims: StationActiveClaims,
        *,
        include_navigation_hint: bool = True,
    ) -> str:
        metric = str(
            claims.last_grounding_query.get("distance_metric")
            or self._metric_from_grounding_handle(
                str(claims.last_grounding_query.get("primitive", ""))
            )
        )
        lines = [f"DOORS RANKED BY {metric.upper()} DISTANCE FROM AGENT"]
        for i, entry in enumerate(claims.ranked_scene_doors):
            lines.append(f"  {i + 1}. {self._entry_label(entry)}")
        if include_navigation_hint:
            lines.append("\n(I can navigate to any specific door — tell me which color.)")
        return "\n".join(lines)

    def _question_override_command(self, utterance: str) -> OperatorCommand | None:
        normalized = _normalize_utterance(utterance)
        if not _looks_like_question(normalized):
            return None
        if "delivery target" in normalized or "target" in normalized:
            return OperatorCommand(
                kind="status_query",
                utterance=utterance,
                payload={"query": "delivery_target"},
            )
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
    ) -> OperatorCommand | None:
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
            return OperatorCommand(
                kind="ambiguous",
                utterance=utterance,
                payload={"message": f"Invalid target selector: {exc}. I did not execute."},
            )
        status = readiness["status"]
        if status == "executable":
            return None
        if status == "synthesizable_missing_primitive":
            return OperatorCommand(
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
            return OperatorCommand(
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
            return OperatorCommand(
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
    ) -> OperatorCommand | None:
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
            )
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={"message": grounded["message"]},
            )

        if grounded.get("status") != "ambiguous":
            return None
        candidates = grounded.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None
        colors = sorted(
            {
                str(candidate["color"])
                for candidate in candidates
                if candidate.get("color") in SUPPORTED_COLORS
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
        )
        return OperatorCommand(
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
        if missing_field == "distance_metric":
            return (
                "CLARIFY\n"
                'To ground "closest", which distance metric should I use?\n'
                f"Supported: {', '.join(supported_values)}"
            )
        if missing_field == "candidate":
            return (
                "CLARIFY\n"
                "That matched multiple doors. Which one should I use?\n"
                f"Options: {_format_targets(candidates or [])}"
            )
        return (
            "CLARIFY\n"
            f"I need {missing_field} before I can ground this selector.\n"
            f"Supported: {', '.join(supported_values)}"
        )

    def handle_pending_clarification(
        self,
        utterance: str,
        command: OperatorCommand,
    ) -> str | None:
        normalized = _normalize_utterance(utterance)
        if command.kind == "quit":
            self.pending_clarification = None
            return "QUIT"
        if command.kind == "cancel":
            self.pending_clarification = None
            return "CANCELLED: pending clarification cleared"
        if command.kind == "reset":
            return self.reset(clear_memory=bool(command.payload.get("clear_memory")))
        if command.kind == "cache_query":
            return self.cache_summary()
        if command.kind == "status_query":
            return self.status_summary(query=command.payload.get("query", "status"))
        pending = self.pending_clarification
        if pending is not None and pending.clarification_type == "arbitrator_offer":
            if self._is_acceptance(normalized):
                self.pending_clarification = None
                return self.resume_arbitration_offer(pending.resume_kind)
        if pending is not None and pending.clarification_type == "semantic_query_missing_field":
            if self._is_manhattan_answer(normalized):
                self.pending_clarification = None
                clarified = f"{pending.original_utterance} using manhattan distance"
                self.log("resuming semantic query plan with distance_metric=manhattan")
                resumed = self.command_from_llm_intent(clarified)
                if resumed.kind == "clarification":
                    return resumed.payload["message"]
                return self.execute_command(resumed)
            if self._is_euclidean_answer(normalized):
                self.pending_clarification = None
                return "I cannot use Euclidean distance yet. Supported: manhattan."
        if self._is_manhattan_answer(normalized):
            return self.resume_pending_clarification("manhattan")
        if self._is_euclidean_answer(normalized):
            self.pending_clarification = None
            return "I cannot use Euclidean distance yet. Supported: manhattan."
        color_answer = self._color_answer(normalized)
        if color_answer is not None:
            candidate_response = self.resume_candidate_clarification(color_answer)
            if candidate_response is not None:
                return candidate_response

        if command.kind == "unresolved":
            self.log("deterministic fast path unresolved; compiling operator intent")
            command = self.command_from_llm_intent(utterance)
        if command.kind in {
            "task_instruction",
            "task_selector",
            "knowledge_update",
            "ground_target_query",
            "unsupported",
            "ambiguous",
        }:
            self.pending_clarification = None
            self.log("new operator intent cancelled pending clarification")
            return self.execute_command(command)
        # A synthesis proposal supersedes the old clarification.
        if command.kind == "synthesis_proposal":
            self.pending_clarification = None
            return command.payload["message"]
        # A new arbitration fired a clarification — _arbitrate_gap already replaced
        # self.pending_clarification with the new pending. Return the new message directly.
        if command.kind == "clarification":
            return command.payload["message"]
        # Arbitration refused or synthesized — no new pending was set, so clear the old one.
        if command.kind in {"missing_skills", "synthesizable"}:
            self.pending_clarification = None
            return command.payload.get("message", "I cannot fulfil that request.")

        pending = self.pending_clarification
        if pending is None:
            return None
        return self.clarification_prompt(
            pending.missing_field,
            pending.supported_values,
            candidates=pending.candidates,
        )

    def execute_command(self, command: OperatorCommand) -> str:
        self.log(f"classified utterance as {command.kind}")
        if command.kind == "clarification":
            return command.payload.get("message", "")
        if command.kind == "synthesis_proposal":
            return command.payload["message"]
        if command.kind == "missing_skills":
            return command.payload.get("message", "I do not have the required capabilities.")
        if command.kind == "synthesizable":
            return command.payload.get("message", "That capability is not yet implemented.")
        if command.kind == "knowledge_update":
            return self.apply_knowledge_update(command.payload)
        if command.kind == "task_instruction":
            instruction = self.resolve_task_instruction(command.utterance)
            if instruction is None:
                return self.missing_reference_summary(command.utterance)
            if instruction != command.utterance:
                self.log(f"resolved task instruction to: {instruction}")
            result = self.run_task(instruction)
            return self.result_summary(result)
        if command.kind == "ground_target_query":
            return self.grounded_target_summary(command.payload)
        if command.kind == "claim_reference":
            return self.claim_reference_summary(command.payload.get("ref_type", ""))
        if command.kind == "task_selector":
            return self.task_selector_summary(command)
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
        return self.status_summary(query="help")

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
        self.pending_clarification = None

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
            return self.apply_knowledge_update(
                {
                    "target_color": target["color"],
                    "target_type": target["type"],
                    "delivery_target": {
                        "color": target["color"],
                        "object_type": target["type"],
                    },
                }
            )
        instruction = f"go to the {target['color']} {target['type']}"
        result = self.run_task(instruction)
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
        self.pending_clarification = None
        if pending.resume_kind == "ground_target_query":
            return (
                "GROUNDED TARGET\n"
                f"target={target.get('color')} {target.get('type')}@({target.get('x')},{target.get('y')})"
            )
        if pending.resume_kind == "knowledge_update":
            return self.apply_knowledge_update(
                {
                    "target_color": target["color"],
                    "target_type": target["type"],
                    "delivery_target": {
                        "color": target["color"],
                        "object_type": target["type"],
                    },
                }
            )
        result = self.run_task(f"go to the {target['color']} {target['type']}")
        return self.result_summary(result)

    def task_selector_summary(self, command: OperatorCommand) -> str:
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
        result = self.run_task(f"go to the {target['color']} {target['type']}")
        return self.result_summary(result)

    def _is_manhattan_answer(self, normalized: str) -> bool:
        return normalized in {
            "manhattan",
            "use manhattan",
            "manhattan distance",
            "by manhattan distance",
            "use manhattan distance",
        }

    def _is_euclidean_answer(self, normalized: str) -> bool:
        return normalized in {
            "euclidean",
            "use euclidean",
            "euclidean distance",
            "by euclidean distance",
            "use euclidean distance",
        }

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
            metric = self._metric_from_grounding_handle(handle)
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
            lines = [f"DOORS RANKED BY {metric.upper()} DISTANCE FROM AGENT"]
            for i, (dist, d) in enumerate(ranked):
                lines.append(f"  {i + 1}. {d.color} door@({d.x},{d.y}) distance={dist}")
            lines.append("\n(I can navigate to any specific door — tell me which color.)")
            return "\n".join(lines)
        return f"No display handler implemented for grounding primitive: {handle}"

    def _metric_from_grounding_handle(self, handle: str) -> str:
        if ".euclidean." in handle:
            return "euclidean"
        if ".manhattan." in handle:
            return "manhattan"
        return "manhattan"

    def _execute_approved_substitute(
        self,
        handle: str,
        utterance: str,
        intent: OperatorIntent,
        cap_match: CapabilityMatchResult,
        decision: Any,
    ) -> OperatorCommand:
        """Execute a registry-validated substitute primitive in place of the missing one.

        Only called when decision.safe_to_execute=True. The handle must be registered
        and implemented — anything else is a hard refuse, not a silent fallback.
        """
        spec = self.capability_registry.lookup(handle)
        if spec is None or spec.implementation_status != "implemented":
            return OperatorCommand(
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
                return OperatorCommand(
                    kind="ambiguous",
                    utterance=utterance,
                    payload={"message": grounded["message"]},
                )
            target = grounded["target"]
            if intent.intent_type == "task_instruction":
                return OperatorCommand(
                    kind="task_instruction",
                    utterance=f"go to the {target['color']} {target['type']}",
                )
            lines = [
                "GROUNDED TARGET (substitute)",
                f"substitute={handle}",
                f"target={target.get('color')} {target.get('type')}"
                f"@({target.get('x')},{target.get('y')})",
            ]
            if grounded.get("distance") is not None:
                lines.append(f"distance={grounded['distance']}")
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={"message": "\n".join(lines)},
                capability_match=cap_match,
            )

        # Display-grounding substitute (e.g., all_doors.ranked for a ranked query).
        if "all_doors.ranked" in handle:
            result_text = self._execute_grounding_display(handle)
            return OperatorCommand(
                kind="clarification",
                utterance=utterance,
                payload={"message": result_text},
                capability_match=cap_match,
            )

        # Unknown primitive type — refuse clearly rather than silently doing nothing.
        msg = decision.operator_message or (
            f"Substitute '{handle}' was approved but no execution path exists for "
            f"primitive type '{spec.layer}'. I did not execute."
        )
        return OperatorCommand(
            kind="missing_skills",
            utterance=utterance,
            payload={"message": msg, "match": cap_match.compact()},
            capability_match=cap_match,
        )

    def _color_answer(self, normalized: str) -> str | None:
        color_pattern = "|".join(SUPPORTED_COLORS)
        match = re.match(
            rf"^(?:the )?(?P<color>{color_pattern})(?: one| door)?$",
            normalized,
        )
        if not match:
            return None
        return _normalize_color(match.group("color"))

    def _ensure_scene_model(self) -> SceneModel | None:
        """Return the current SceneModel, building it via an idle sense pass if needed.

        Uses the existing Sense path (parse_grid_objects) against the live adapter.
        Never resets the env — if no adapter exists, creates a temporary one only
        for the initial observation, then closes it.
        """
        if self.memory.scene_model is not None:
            return self.memory.scene_model
        adapter = self.preview_adapter or self.task_adapter
        close_after = False
        if adapter is None:
            env = run_demo.build_env(self.env_id, "none")
            adapter = MiniGridAdapter(env)
            adapter.reset(seed=self.seed)
            close_after = True
        try:
            observation = adapter.observe()
            self.sense.sense_idle_scene(
                observation, env_id=self.env_id, seed=self.seed
            )
        finally:
            if close_after:
                adapter.close()
        return self.memory.scene_model

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
        self.active_claims = StationActiveClaims(
            scene_fingerprint=(scene.agent_x, scene.agent_y, scene.step_count),
            ranked_scene_doors=entries,
            last_grounded_target=entries[0],
            last_grounded_rank=0,
            last_grounding_query=dict(selector),
        )

    def _resolve_claim_reference(self, ref_type: str) -> dict[str, Any]:
        scene = self.memory.scene_model
        if scene is None:
            return {"ok": False, "message": "No scene data available."}
        if self.active_claims is None:
            return {
                "ok": False,
                "message": (
                    "No active grounding claims. "
                    "Try 'which door is closest by Manhattan distance?' first."
                ),
            }
        if not self.active_claims.is_valid_for(scene):
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
        self.close_preview()
        env = run_demo.build_env(self.env_id, self.render_mode)
        self.preview_adapter = MiniGridAdapter(env)
        self.preview_adapter.reset(seed=self.seed)
        try:
            env.render()
        except Exception:  # noqa: BLE001
            pass

    def _pump_render_window(self) -> None:
        adapter = self.preview_adapter or self.task_adapter
        if adapter is None:
            return
        try:
            adapter.env.render()
        except Exception:  # noqa: BLE001
            pass

    def close_preview(self) -> None:
        if self.preview_adapter is None:
            return
        self.log("closing idle render preview")
        self.preview_adapter.close()
        self.preview_adapter = None

    def close_task_window(self) -> None:
        if self.task_adapter is None:
            return
        self.log("closing previous task render window")
        self.task_adapter.close()
        self.task_adapter = None

    def run_task(self, instruction: str) -> dict[str, Any]:
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
        render_adapter = self.preview_adapter
        self.preview_adapter = None
        if render_adapter is None:
            self.close_task_window()
        elif self.task_adapter is not None and self.task_adapter is not render_adapter:
            self.close_task_window()
        self.last_result = run_demo.run_episode(
            instruction=instruction,
            compiler_name=self.compiler_name,
            compiler=self.compiler,
            env_id=self.env_id,
            seed=self.seed,
            max_loops=self.max_loops,
            render_mode=self.render_mode,
            memory=self.memory,
            plan_cache=self.plan_cache,
            use_cache=self.plan_cache.enabled,
            prewarm=True,
            keep_render_open=self.render_mode == "human",
            render_adapter=render_adapter,
            progress_callback=self._progress_callback,
            task_override=task_override,
            procedure_override=procedure_override,
        )
        self.task_adapter = self.last_result.pop("_render_adapter", None)
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

    def apply_knowledge_update(self, payload: dict[str, Any]) -> str:
        self.log("updating durable target knowledge")
        for key in ("target_color", "target_type", "delivery_target"):
            if key in payload:
                self.memory.update_knowledge(key, payload[key])
        return (
            "KNOWLEDGE UPDATED\n"
            f"delivery_target={self.memory.knowledge.get('delivery_target')}"
        )

    def reset(self, *, clear_memory: bool = False) -> str:
        self.log("resetting station state")
        self.pending_clarification = None
        self.pending_synthesis_proposal = None
        self.active_claims = None
        self.memory.reset_episode()
        self.last_result = None
        if clear_memory:
            self.memory.update_knowledge("target_color", None, persist=False)
            self.memory.update_knowledge("target_type", None, persist=False)
            self.memory.update_knowledge("delivery_target", None, persist=False)
            self.memory.update_knowledge("last_task_type", None, persist=False)
            self.memory.update_knowledge("last_instruction", None)
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
