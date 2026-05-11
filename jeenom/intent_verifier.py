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

if TYPE_CHECKING:
    from .schemas import OperatorIntent

SIGNAL_SUPERLATIVE = "superlative"   # farthest, furthest, most distant
SIGNAL_CARDINALITY = "cardinality"   # all doors, sort/rank/list by distance
SIGNAL_ORDINAL = "ordinal"           # second closest, third nearest
SIGNAL_DISTANCE_VALUE = "distance_value"  # door with distance N

_SUPERLATIVE_TERMS: frozenset[str] = frozenset([
    "farthest", "furthest", "most distant", "most far",
    "farthest away", "furthest away", "longest way", "maximum distance",
    "max distance", "least close", "least nearest",
])

_CARDINALITY_TRIGGERS: list[tuple[re.Pattern[str], str]] = [
    # "sort/rank/order/list" near "doors" or "by distance"
    (re.compile(r"\b(sort|rank|ranked|ranking|order|list)\b.{0,40}\b(door|doors|them|distance)\b"), "sort/rank/list"),
    # "all/every/each" + "door(s)"
    (re.compile(r"\b(all|every|each)\b.{0,20}\b(door|doors)\b"), "all/every/each doors"),
    # "all/every" + them/of them in a door context
    (re.compile(r"\b(all of them|all the doors|each of them|every one of them)\b"), "all of them"),
    # "distance(s) of" multiple objects
    (re.compile(r"\bdistances?\s+of\s+(all|the|every|each)\b"), "distance of all"),
    # "distance" + "all" + "doors" (any order within window)
    (re.compile(r"\ball\b.{0,30}\bdistance\b|\bdistance\b.{0,30}\ball\b.{0,20}\bdoor"), "distance+all+door"),
    # descending/ascending without "closest" already handled by ranked check
    (re.compile(r"\b(descending|ascending)\b.{0,30}\b(door|doors|distance)\b"), "descending/ascending order"),
]

_ORDINAL_PATTERN: re.Pattern[str] = re.compile(
    r"\b(second|third|fourth|fifth|2nd|3rd|4th|5th)\s+(closest|nearest|farthest|furthest)\b"
)

_DISTANCE_VALUE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(?:distance\s+(?:of\s+)?|with\s+(?:a\s+)?distance\s+(?:of\s+)?)(?P<distance>\d+)\b"
)


def _detect_metric(normalized: str) -> str:
    if "euclidean" in normalized:
        return "euclidean"
    return "manhattan"


@dataclass
class IntentSignal:
    signal_type: str
    detected_term: str
    required_handle: str


@dataclass
class IntentVerificationResult:
    signals: list[IntentSignal] = field(default_factory=list)
    injected_handles: list[str] = field(default_factory=list)

    @property
    def has_signals(self) -> bool:
        return bool(self.signals)

    def summary(self) -> str:
        if not self.signals:
            return "no signals detected"
        parts = [f"{s.signal_type}:{s.detected_term!r}→{s.required_handle}" for s in self.signals]
        return "; ".join(parts)


class IntentVerifier:
    """Proactive, deterministic semantic signal extractor.

    Reads the utterance directly. Does not trust the LLM's required_capabilities
    field. Injects missing handles so CapabilityMatcher can fire correctly.
    """

    def verify(self, utterance: str, intent: "OperatorIntent") -> IntentVerificationResult:
        normalized = " ".join(utterance.lower().strip().split())
        metric = _detect_metric(normalized)
        signals: list[IntentSignal] = []

        # ── Ordinal signals ────────────────────────────────────────────────
        # Check ordinal before superlative so "second farthest" is not
        # collapsed into plain "farthest".
        m = _ORDINAL_PATTERN.search(normalized)
        if m:
            ordinal = m.group(1)
            direction = m.group(2)
            handle = f"grounding.all_doors.ranked.{metric}.agent"
            signals.append(IntentSignal(
                signal_type=SIGNAL_ORDINAL,
                detected_term=f"{ordinal} {direction}",
                required_handle=handle,
            ))

        # ── Superlative signals ────────────────────────────────────────────
        for term in _SUPERLATIVE_TERMS:
            if not signals and term in normalized:
                signals.append(IntentSignal(
                    signal_type=SIGNAL_SUPERLATIVE,
                    detected_term=term,
                    required_handle=f"grounding.all_doors.ranked.{metric}.agent",
                ))
                break  # one superlative signal is enough

        # ── Cardinality signals ────────────────────────────────────────────
        if not signals:  # superlative takes precedence
            for pattern, label in _CARDINALITY_TRIGGERS:
                if pattern.search(normalized):
                    signals.append(IntentSignal(
                        signal_type=SIGNAL_CARDINALITY,
                        detected_term=label,
                    required_handle=f"grounding.all_doors.ranked.{metric}.agent",
                    ))
                    break

        # ── Distance-value reference signals ──────────────────────────────
        if not signals and "door" in normalized:
            m = _DISTANCE_VALUE_PATTERN.search(normalized)
            if m:
                signals.append(IntentSignal(
                    signal_type=SIGNAL_DISTANCE_VALUE,
                    detected_term=f"distance {m.group('distance')}",
                    required_handle=f"grounding.all_doors.ranked.{metric}.agent",
                ))

        return self._inject(intent, signals)

    def _inject(
        self,
        intent: "OperatorIntent",
        signals: list[IntentSignal],
    ) -> IntentVerificationResult:
        if not signals:
            return IntentVerificationResult()

        import dataclasses
        existing = set(intent.required_capabilities or [])
        to_inject = [
            s.required_handle for s in signals
            if s.required_handle not in existing
        ]

        if to_inject:
            enriched = dataclasses.replace(
                intent,
                required_capabilities=list(existing) + to_inject,
            )
            # Mutate the intent in-place via the caller — we return the signals
            # and injected list; the caller applies the enriched intent.
        else:
            enriched = intent  # noqa: F841 — caller uses signals list

        return IntentVerificationResult(
            signals=signals,
            injected_handles=to_inject,
        )

    def enrich(self, utterance: str, intent: "OperatorIntent") -> tuple["OperatorIntent", IntentVerificationResult]:
        """Return (enriched_intent, result). Primary API for the station."""
        import dataclasses
        normalized = " ".join(utterance.lower().strip().split())
        metric = _detect_metric(normalized)
        signals: list[IntentSignal] = []

        m = _ORDINAL_PATTERN.search(normalized)
        if m:
            ordinal = m.group(1)
            direction = m.group(2)
            handle = f"grounding.all_doors.ranked.{metric}.agent"
            signals.append(IntentSignal(
                signal_type=SIGNAL_ORDINAL,
                detected_term=f"{ordinal} {direction}",
                required_handle=handle,
            ))

        for term in _SUPERLATIVE_TERMS:
            if not signals and term in normalized:
                signals.append(IntentSignal(
                    signal_type=SIGNAL_SUPERLATIVE,
                    detected_term=term,
                    required_handle=f"grounding.all_doors.ranked.{metric}.agent",
                ))
                break

        if not signals:
            for pattern, label in _CARDINALITY_TRIGGERS:
                if pattern.search(normalized):
                    signals.append(IntentSignal(
                        signal_type=SIGNAL_CARDINALITY,
                        detected_term=label,
                    required_handle=f"grounding.all_doors.ranked.{metric}.agent",
                    ))
                    break

        if not signals and "door" in normalized:
            m = _DISTANCE_VALUE_PATTERN.search(normalized)
            if m:
                signals.append(IntentSignal(
                    signal_type=SIGNAL_DISTANCE_VALUE,
                    detected_term=f"distance {m.group('distance')}",
                    required_handle=f"grounding.all_doors.ranked.{metric}.agent",
                ))

        if not signals:
            return intent, IntentVerificationResult()

        existing = set(intent.required_capabilities or [])
        to_inject = [s.required_handle for s in signals if s.required_handle not in existing]
        enriched = dataclasses.replace(
            intent,
            required_capabilities=list(existing) + to_inject,
        )
        return enriched, IntentVerificationResult(signals=signals, injected_handles=to_inject)


default_verifier = IntentVerifier()
