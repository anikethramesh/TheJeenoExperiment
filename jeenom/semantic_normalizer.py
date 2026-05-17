from __future__ import annotations

import re
from dataclasses import dataclass


_ORDINALS: dict[str, int] = {
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
    metric: str

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
            self.metric,
        ]
        return list(dict.fromkeys(constraints))


def normalize_distance_ordinal(text: str) -> DistanceOrdinalSemantics | None:
    """Extract ordinal distance ordering from natural operator text.

    Handles phrases such as "second highest manhattan distance" and
    "third smallest distance" without making an LLM output the authority.
    """

    normalized = re.sub(r"\s+", " ", text.strip().lower())
    if "door" not in normalized or "distance" not in normalized:
        return None

    ordinal_pattern = "|".join(re.escape(word) for word in _ORDINALS)
    ordinal_match = re.search(rf"\b(?P<ordinal>{ordinal_pattern})\b", normalized)
    if ordinal_match is None:
        return None

    ordinal_word = ordinal_match.group("ordinal")
    after_ordinal = normalized[ordinal_match.end(): ordinal_match.end() + 96]

    for term in _DESCENDING_DISTANCE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", after_ordinal):
            return DistanceOrdinalSemantics(
                ordinal_word=ordinal_word,
                ordinal=_ORDINALS[ordinal_word],
                order="descending",
                direction_term=term,
                metric=_detect_metric(normalized),
            )

    for term in _ASCENDING_DISTANCE_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", after_ordinal):
            return DistanceOrdinalSemantics(
                ordinal_word=ordinal_word,
                ordinal=_ORDINALS[ordinal_word],
                order="ascending",
                direction_term=term,
                metric=_detect_metric(normalized),
            )

    return None


def _detect_metric(normalized: str) -> str:
    if "euclidean" in normalized:
        return "euclidean"
    return "manhattan"
