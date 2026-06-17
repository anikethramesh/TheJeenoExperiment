"""Phase 7.595 — Proactive Intent Signal Verification.

IntentVerifier sits between the LLM compiler output and the CapabilityMatcher.
It extracts semantic signals from the utterance text directly — regardless of
what the LLM declared — and injects the correct required_capabilities so the
CapabilityMatcher can fire.

Pure, substrate-independent. No env, no sense, no memory, no LLM calls.
No minigrid, no gymnasium, no substrate imports.

Blueprint Rule 9: intent inversion (farthest → closest) is a hard stop.
Silent degradation (ranked listing → single closest) is a hard stop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .planning_semantics import PlanningSemantics, default_planning_semantics
from .semantic_normalizer import infer_direction_from_utterance, normalize_distance_ordinal

if TYPE_CHECKING:
    from .schemas import OperatorIntent, SelectionObjective, SteeringDirective

SIGNAL_SUPERLATIVE = "superlative"   # farthest, furthest, most distant
SIGNAL_CARDINALITY = "cardinality"   # all objects, sort/rank/list by distance
SIGNAL_ORDINAL = "ordinal"           # second closest, third nearest
SIGNAL_DISTANCE_VALUE = "distance_value"  # object with distance N

_SUPERLATIVE_TERMS: frozenset[str] = frozenset([
    "farthest", "furthest", "most distant", "most far",
    "farthest away", "furthest away", "longest way", "maximum distance",
    "max distance", "highest", "largest", "greatest", "maximum",
    "least close", "least nearest",
])

_ORDINAL_PATTERN: re.Pattern[str] = re.compile(
    r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th)\s+(closest|nearest|farthest|furthest)\b"
)

_DISTANCE_VALUE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:distance\s+(?:of\s+)?|with\s+(?:a\s+)?distance\s+(?:of\s+)?)(?P<distance>\d+)\b"
)


def _detect_metric(normalized: str, semantics: PlanningSemantics) -> str | None:
    for metric in semantics.metrics:
        if re.search(rf"\b{re.escape(metric)}\b", normalized):
            return metric
    return semantics.default_metric


def _cardinality_triggers(semantics: PlanningSemantics) -> list[tuple[re.Pattern[str], str]]:
    object_terms: list[str] = []
    for object_type in semantics.object_types:
        object_terms.extend([object_type, semantics.pluralize(object_type)])
    object_terms.append("them")
    object_pattern = "|".join(re.escape(term) for term in object_terms if term)
    if not object_pattern:
        object_pattern = "objects?"
    return [
        (
            re.compile(
                rf"\b(sort|rank|ranked|ranking|order|list)\b.{{0,40}}\b({object_pattern}|distance)\b"
            ),
            "sort/rank/list",
        ),
        (
            re.compile(rf"\b(all|every|each)\b.{{0,20}}\b({object_pattern})\b"),
            "all/every/each objects",
        ),
        (re.compile(r"\b(all of them|each of them|every one of them)\b"), "all of them"),
        (re.compile(r"\bdistances?\s+of\s+(all|the|every|each)\b"), "distance of all"),
        (
            re.compile(
                rf"\ball\b.{{0,30}}\bdistance\b|\bdistance\b.{{0,30}}\ball\b.{{0,20}}\b({object_pattern})"
            ),
            "distance+all+object",
        ),
        (
            re.compile(rf"\b(descending|ascending)\b.{{0,30}}\b({object_pattern}|distance)\b"),
            "descending/ascending order",
        ),
    ]


@dataclass
class IntentSignal:
    signal_type: str
    detected_term: str
    required_handle: str


@dataclass
class IntentVerificationResult:
    signals: list[IntentSignal] = field(default_factory=list)
    injected_handles: list[str] = field(default_factory=list)
    inversion_detected: bool = False
    inversion_reason: str = ""

    @property
    def has_signals(self) -> bool:
        return bool(self.signals)

    def summary(self) -> str:
        if not self.signals:
            return "no signals detected"
        parts = [f"{s.signal_type}:{s.detected_term!r}→{s.required_handle}" for s in self.signals]
        return "; ".join(parts)


class IntentVerifier:
    """Proactive, deterministic semantic signal extractor."""

    def __init__(self, planning_semantics: PlanningSemantics | None = None) -> None:
        self.planning_semantics = planning_semantics or default_planning_semantics()

    def verify(self, utterance: str, intent: "OperatorIntent") -> IntentVerificationResult:
        signals, inversion_detected, inversion_reason = self._analyze(utterance, intent)
        existing = set(intent.required_capabilities or [])
        return IntentVerificationResult(
            signals=signals,
            injected_handles=[
                signal.required_handle
                for signal in signals
                if signal.required_handle not in existing
            ],
            inversion_detected=inversion_detected,
            inversion_reason=inversion_reason,
        )

    def enrich(
        self,
        utterance: str,
        intent: "OperatorIntent",
        steering_directive: "SteeringDirective | None" = None,
    ) -> tuple["OperatorIntent", IntentVerificationResult]:
        """Return (enriched_intent, result). Primary API for the station.

        A parsed SteeringDirective (Phase 13A) is attached here — the gate — and only
        when coherent with the intent (steering shapes *action* intents). An incoherent
        directive (e.g. steering a control/query turn) is dropped, never silently
        applied, so a misparse cannot reshape a plan it has no business touching.
        """
        import dataclasses

        signals, inversion_detected, inversion_reason = self._analyze(utterance, intent)
        attach_steering = (
            steering_directive is not None
            and intent.steering_directive is None
            and intent.knowledge_type == "action"
        )

        if not signals and not inversion_detected and not attach_steering:
            return intent, IntentVerificationResult()

        existing = set(intent.required_capabilities or [])
        to_inject = [
            signal.required_handle
            for signal in signals
            if signal.required_handle not in existing
        ]
        replace_kwargs: dict = {}
        if to_inject:
            replace_kwargs["required_capabilities"] = list(existing) + to_inject
        if attach_steering:
            replace_kwargs["steering_directive"] = steering_directive
        enriched = (
            dataclasses.replace(intent, **replace_kwargs) if replace_kwargs else intent
        )
        return enriched, IntentVerificationResult(
            signals=signals,
            injected_handles=to_inject,
            inversion_detected=inversion_detected,
            inversion_reason=inversion_reason,
        )

    def _analyze(
        self,
        utterance: str,
        intent: "OperatorIntent",
    ) -> tuple[list[IntentSignal], bool, str]:
        normalized = " ".join(utterance.lower().strip().split())
        semantics = self.planning_semantics
        metric = _detect_metric(normalized, semantics)
        object_type = semantics.object_type_from_text(normalized)
        signals: list[IntentSignal] = []

        ordinal_semantics = normalize_distance_ordinal(normalized)
        if ordinal_semantics is not None and semantics.metric_supported(ordinal_semantics.metric):
            handle = semantics.ranked_handle(ordinal_semantics.metric, object_type=object_type)
            if handle is not None:
                signals.append(IntentSignal(
                    signal_type=SIGNAL_ORDINAL,
                    detected_term=(
                        f"{ordinal_semantics.ordinal_word} "
                        f"{ordinal_semantics.direction_term}"
                    ),
                    required_handle=handle,
                ))
        else:
            m = _ORDINAL_PATTERN.search(normalized)
            if m and metric is not None:
                handle = semantics.ranked_handle(metric, object_type=object_type)
                if handle is not None:
                    signals.append(IntentSignal(
                        signal_type=SIGNAL_ORDINAL,
                        detected_term=f"{m.group(1)} {m.group(2)}",
                        required_handle=handle,
                    ))

        for term in _SUPERLATIVE_TERMS:
            if not signals and term in normalized and metric is not None:
                handle = semantics.ranked_handle(metric, object_type=object_type)
                if handle is not None:
                    signals.append(IntentSignal(
                        signal_type=SIGNAL_SUPERLATIVE,
                        detected_term=term,
                        required_handle=handle,
                    ))
                break

        if not signals and metric is not None:
            for pattern, label in _cardinality_triggers(semantics):
                if pattern.search(normalized):
                    handle = semantics.ranked_handle(metric, object_type=object_type)
                    if handle is not None:
                        signals.append(IntentSignal(
                            signal_type=SIGNAL_CARDINALITY,
                            detected_term=label,
                            required_handle=handle,
                        ))
                    break

        if not signals and object_type is not None and metric is not None:
            m = _DISTANCE_VALUE_PATTERN.search(normalized)
            if m:
                handle = semantics.ranked_handle(metric, object_type=object_type)
                if handle is not None:
                    signals.append(IntentSignal(
                        signal_type=SIGNAL_DISTANCE_VALUE,
                        detected_term=f"distance {m.group('distance')}",
                        required_handle=handle,
                    ))

        inversion_detected, inversion_reason = self._detect_inversion(normalized, intent)
        selection_obj = getattr(intent, "selection_objective", None)
        if selection_obj is not None and not signals:
            objective_metric = selection_obj.metric or semantics.default_metric
            objective_object_type = getattr(selection_obj, "object_type", None) or object_type
            handle = semantics.ranked_handle(
                objective_metric,
                object_type=objective_object_type,
            )
            existing_caps = set(intent.required_capabilities or [])
            if handle is not None and handle not in existing_caps:
                signals.append(IntentSignal(
                    signal_type=SIGNAL_SUPERLATIVE,
                    detected_term=f"objective:{selection_obj.direction}",
                    required_handle=handle,
                ))

        return signals, inversion_detected, inversion_reason

    def _detect_inversion(
        self,
        normalized: str,
        intent: "OperatorIntent",
    ) -> tuple[bool, str]:
        grounding_plan = getattr(intent, "grounding_query_plan", None)
        if grounding_plan is None:
            return False, ""
        plan_order = grounding_plan.get("order") if isinstance(grounding_plan, dict) else None
        if plan_order is None:
            return False, ""

        selection_obj = getattr(intent, "selection_objective", None)
        if selection_obj is not None:
            expected_order = (
                "descending" if selection_obj.direction == "maximum" else "ascending"
            )
            if plan_order == expected_order:
                return False, ""
            answer_fields = {
                str(field).lower()
                for field in (grounding_plan.get("answer_fields") or [])
            }
            covers = (
                (selection_obj.direction == "maximum" and "farthest" in answer_fields)
                or (selection_obj.direction == "minimum" and "closest" in answer_fields)
            )
            if covers:
                return False, ""
            return True, (
                f"Objective says direction={selection_obj.direction!r} "
                f"(requires order={expected_order!r}) "
                f"but plan declares order={plan_order!r}."
            )

        expected_order = infer_direction_from_utterance(normalized)
        if expected_order is None or plan_order == expected_order:
            return False, ""
        answer_fields = {
            str(field).lower()
            for field in (grounding_plan.get("answer_fields") or [])
        }
        covers = (
            (expected_order == "descending" and "farthest" in answer_fields)
            or (expected_order == "ascending" and "closest" in answer_fields)
        )
        if covers:
            return False, ""
        return True, (
            f"Utterance implies order={expected_order!r} "
            f"but plan declares order={plan_order!r}."
        )


default_verifier = IntentVerifier()
