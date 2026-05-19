from __future__ import annotations

import re
from dataclasses import dataclass


_ORDINALS: dict[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
}

_DESCENDING_DISTANCE_TERMS = (
    "highest",
    "largest",
    "greatest",
    "maximum",
    "max",
    "farthest",
    "furthest",
    "most distant",
    "most far",
)

_ASCENDING_DISTANCE_TERMS = (
    "lowest",
    "smallest",
    "minimum",
    "min",
    "closest",
    "nearest",
    "shortest",
    "least distant",
)


@dataclass(frozen=True)
class DistanceOrdinalSemantics:
    ordinal_word: str
    ordinal: int
    order: str
    direction_term: str
    metric: str | None

    @property
    def canonical_direction(self) -> str:
        return "farthest" if self.order == "descending" else "closest"

    @property
    def preserved_constraints(self) -> list[str]:
        constraints = [
            self.ordinal_word,
            self.direction_term,
            self.canonical_direction,
            "door",
        ]
        if self.metric is not None:
            constraints.append(self.metric)
        return list(dict.fromkeys(constraints))


def _extract_ordinal(text: str) -> tuple[str, int, int, int] | None:
    numeric_match = re.search(r"\b(\d+)(st|nd|rd|th)\b", text)
    if numeric_match:
        word = numeric_match.group(0)
        value = int(numeric_match.group(1))
        return (word, value, numeric_match.start(), numeric_match.end())
        
    word_pattern = "|".join(re.escape(word) for word in _ORDINALS)
    word_match = re.search(rf"\b({word_pattern})\b", text)
    if word_match:
        word = word_match.group(1)
        value = _ORDINALS[word]
        return (word, value, word_match.start(), word_match.end())
        
    return None


def normalize_distance_ordinal(text: str) -> DistanceOrdinalSemantics | None:
    """Extract ordinal distance ordering from natural operator text.

    Handles phrases such as "second highest manhattan distance" and
    "third smallest distance" without making an LLM output the authority.
    """

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if "door" not in normalized or "distance" not in normalized:
        return None

    extracted = _extract_ordinal(normalized)
    if extracted is None:
        return None

    ordinal_word, ordinal_val, start_idx, end_idx = extracted
    after_ordinal = normalized[end_idx: end_idx + 96]

    for term in _DESCENDING_DISTANCE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", after_ordinal):
            return DistanceOrdinalSemantics(
                ordinal_word=ordinal_word,
                ordinal=ordinal_val,
                order="descending",
                direction_term=term,
                metric=_detect_metric(normalized),
            )

    for term in _ASCENDING_DISTANCE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", after_ordinal):
            return DistanceOrdinalSemantics(
                ordinal_word=ordinal_word,
                ordinal=ordinal_val,
                order="ascending",
                direction_term=term,
                metric=_detect_metric(normalized),
            )

    return None


def _detect_metric(normalized: str) -> str | None:
    if "euclidean" in normalized:
        return "euclidean"
    if "manhattan" in normalized:
        return "manhattan"
    return None


def get_semantic_constraints() -> dict[str, list[str]]:
    """Export the exact deterministic vocabulary constraints so the LLM compiler can advertise them."""
    return {
        "ordinals": list(_ORDINALS.keys()) + ["any numeric ordinal like 11th or 42nd"],
        "descending_distance_terms": list(_DESCENDING_DISTANCE_TERMS),
        "ascending_distance_terms": list(_ASCENDING_DISTANCE_TERMS),
    }
