from __future__ import annotations

import argparse
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .capability_arbitrator import ArbitratorBackend, build_arbitrator
from .capability_matcher import CapabilityMatchResult, CapabilityMatcher, default_matcher
from .capability_registry import CapabilityRegistry
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
        self.active_claims: StationActiveClaims | None = None
        self.arbitrator: ArbitratorBackend = build_arbitrator(compiler_name)
        self.last_arbitration_trace: ArbitrationTrace | None = None

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
        if self.pending_clarification is not None:
            pending_response = self.handle_pending_clarification(utterance, command)
            if pending_response is not None:
                return pending_response
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

    def command_from_llm_intent(self, utterance: str) -> OperatorCommand:
        intent = self.compiler.compile_operator_intent(
            utterance,
            memory=self.memory,
            scene_summary=None,
            capability_manifest=self.capability_registry.compact_summary(),
            active_claims_summary=(
                self.active_claims.compact_summary() if self.active_claims is not None else None
            ),
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

        if intent.intent_type == "claim_reference":
            return OperatorCommand(
                kind="claim_reference",
                utterance=utterance,
                payload={"ref_type": intent.claim_reference or ""},
            )

        if intent.intent_type in {"unsupported", "ambiguous"}:
            return OperatorCommand(
                kind=intent.intent_type,
                utterance=utterance,
                payload={
                    "message": (
                        "I cannot safely execute that capability yet."
                        if intent.intent_type == "unsupported"
                        else "I could not safely resolve that instruction yet."
                    ),
                    "reason": intent.reason,
                },
            )

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
            # Only allow substitution when the arbitrator explicitly flags it safe.
            if decision.safe_to_execute:
                self.log(
                    f"arbitration substitute approved: {decision.suggested_handle}"
                )
            # Regardless of safe_to_execute, report the suggestion but refuse execution
            # unless safe_to_execute is explicitly true (and we trust the arbitrator).
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
            # safe_to_execute=True — pass through with a note (rare; arbitrator must be sure)
            msg = decision.operator_message or (
                f"Using substitute capability: {decision.suggested_handle}"
            )
            return OperatorCommand(
                kind="missing_skills",
                utterance=utterance,
                payload={"message": msg, "match": cap_match.compact()},
                capability_match=cap_match,
            )

        # Default: refuse
        kind = "synthesizable" if cap_match.verdict == "synthesizable" else "missing_skills"
        return OperatorCommand(
            kind=kind,
            utterance=utterance,
            payload={
                "message": decision.operator_message or cap_match.operator_message(),
                "match": cap_match.compact(),
            },
            capability_match=cap_match,
        )

    def _build_scene_summary_for_arbitrator(self) -> dict[str, Any] | None:
        scene = self.memory.scene_model
        if scene is None:
            return None
        doors = scene.find(object_type="door")
        return {
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
            if selector.get("distance_metric") != "manhattan":
                return {
                    "ok": False,
                    "status": "invalid_unsupported",
                    "message": (
                        'I need distance_metric=manhattan to ground "closest". '
                        "Clarification is not enabled yet."
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
        """Run the named grounding primitive against the current SceneModel and return text."""
        scene = self._ensure_scene_model()
        if scene is None:
            return "No scene data available yet."
        if "all_doors.ranked" in handle:
            doors = scene.find(object_type="door")
            if not doors:
                return "No doors visible in the current scene."
            ranked = sorted(
                [(scene.manhattan_distance_from_agent(d), d) for d in doors],
                key=lambda pair: (pair[0], pair[1].color or ""),
            )
            lines = ["DOORS RANKED BY MANHATTAN DISTANCE FROM AGENT"]
            for i, (dist, d) in enumerate(ranked):
                lines.append(f"  {i + 1}. {d.color} door@({d.x},{d.y}) distance={dist}")
            lines.append("\n(I can navigate to any specific door — tell me which color.)")
            return "\n".join(lines)
        return f"No display handler implemented for grounding primitive: {handle}"

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
        ranked_pairs: list[tuple[int, SceneObject]],
        selector: dict[str, Any],
    ) -> None:
        scene = self.memory.scene_model
        if scene is None or not ranked_pairs:
            return
        entries = [
            GroundedDoorEntry(color=obj.color, x=obj.x, y=obj.y, distance=dist)
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
