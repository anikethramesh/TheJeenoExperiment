"""Phase 13A.2.3 — metric-name detection consolidation (behavior-preserving).

The euclidean-first text detection was hand-rolled in `semantic_normalizer._detect_metric`
and re-implemented inline in `operator_station`. This promotes one public
`detect_metric` / `mentions_metric` home, with the known-metric SET sourced from the
canonical `OPERATOR_DISTANCE_METRICS` and the euclidean-first priority preserved exactly.

Scope fence (unchanged behavior, left alone): the divergent per-site default metrics
(filter→euclidean, ranked→manhattan), the deliberately-narrow supported lists
(`["manhattan"]`), the `metric == "manhattan"` compute-branch, the plan-mismatch logic, and
`intent_verifier`'s own PlanningSemantics-driven detector.

Red-bar first: imports the public helpers (not public yet) and pins the euclidean-first
contract so a manhattan-first regression is caught.
"""
from __future__ import annotations

from pathlib import Path

from jeenom.schemas import OPERATOR_DISTANCE_METRICS
from jeenom.semantic_normalizer import detect_metric, mentions_metric

STATION = (Path(__file__).resolve().parents[1] / "jeenom" / "operator_station.py").read_text(
    encoding="utf-8"
)


def test_detect_metric_basic():
    assert detect_metric("rank by euclidean") == "euclidean"
    assert detect_metric("rank by manhattan") == "manhattan"
    assert detect_metric("go to the third door") is None


def test_detect_metric_is_euclidean_first_when_both_present():
    # The original _detect_metric checked euclidean first; ambiguous text must keep
    # resolving to euclidean (a manhattan-first regression would silently change plans).
    assert detect_metric("compare euclidean and manhattan") == "euclidean"
    assert detect_metric("manhattan vs euclidean") == "euclidean"


def test_mentions_metric_matches_or_of_substrings():
    assert mentions_metric("use manhattan") is True
    assert mentions_metric("use euclidean") is True
    assert mentions_metric("nearest door") is False


def test_known_metric_set_is_sourced_from_canonical_list():
    # Single source of names: detection covers exactly the canonical metric set.
    detected = {m for m in OPERATOR_DISTANCE_METRICS if detect_metric(f"x {m} y") == m}
    assert detected == set(OPERATOR_DISTANCE_METRICS)


def test_station_routes_inline_detection_through_the_home():
    assert "mentions_metric" in STATION and "detect_metric" in STATION
