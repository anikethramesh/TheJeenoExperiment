"""Op 4 probe: shadow NLU demoted into IntentCache.

Verifies:
  1. classify_utterance body has no inline regex (no re.compile/match/search)
  2. IntentCache is a distinct class from PlanReuseCache
  3. An IntentCache entry for the metric-definition pattern returns OperatorIntent
  4. TurnOrchestrator holds intent_cache; cached intents route through dispatch
"""
from __future__ import annotations

import ast as _ast
import inspect
import textwrap
from pathlib import Path
from typing import Any

from harness import emit_result, make_session

ROOT = Path(__file__).resolve().parents[1]

_RE_CALL_ATTRS = frozenset({"compile", "match", "search", "fullmatch", "findall", "finditer"})


def _inline_re_calls(fn: object) -> list[str]:
    """Return lines where fn's source calls re.<method>(...)."""
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = _ast.parse(src)
    except Exception:
        return ["<parse error>"]
    hits: list[str] = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, _ast.Attribute)
            and func.attr in _RE_CALL_ATTRS
            and isinstance(func.value, _ast.Name)
            and func.value.id == "re"
        ):
            hits.append(f"line {node.lineno}: re.{func.attr}(...)")
    return hits


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # ── 1. classify_utterance body has no inline regex calls ─────────────────
    try:
        from jeenom.operator_station import classify_utterance
        hits = _inline_re_calls(classify_utterance)
        metrics["op4_classify_no_inline_regex"] = not hits
        details["op4_classify_inline_re_hits"] = hits
    except Exception as exc:
        metrics["op4_classify_no_inline_regex"] = False
        details["op4_classify_import_error"] = f"{type(exc).__name__}: {exc}"

    # ── 2. IntentCache exists and is distinct from PlanReuseCache ─────────────
    try:
        from jeenom.intent_cache import IntentCache
        from jeenom.plan_reuse import PlanReuseCache
        is_distinct = IntentCache is not PlanReuseCache
        metrics["op4_intent_cache_distinct"] = is_distinct
        details["op4_intent_cache_class"] = IntentCache.__name__
        details["op4_plan_reuse_cache_class"] = PlanReuseCache.__name__
    except Exception as exc:
        metrics["op4_intent_cache_distinct"] = False
        details["op4_intent_cache_import_error"] = f"{type(exc).__name__}: {exc}"

    # ── 3. IntentCache entry for metric-definition returns OperatorIntent ──────
    try:
        from jeenom.intent_cache import IntentCache, seed_intent_cache
        from jeenom.schemas import OperatorIntent
        session = make_session()
        cache = IntentCache()
        seed_intent_cache(cache, session.capability_registry)
        # A valid metric-definition pattern
        test_utt = "create a metric called mytest which is min(euclidean, manhattan)"
        result = cache.lookup(test_utt)
        is_operator_intent = isinstance(result, OperatorIntent)
        correct_type = (
            is_operator_intent and result.intent_type == "primitive_definition"
        )
        metrics["op4_cache_produces_primitive_definition"] = correct_type
        details["op4_cache_lookup_result_type"] = type(result).__name__
        details["op4_cache_lookup_intent_type"] = (
            result.intent_type if is_operator_intent else None
        )
    except Exception as exc:
        metrics["op4_cache_produces_primitive_definition"] = False
        details["op4_cache_seed_error"] = f"{type(exc).__name__}: {exc}"

    # ── 4. TurnOrchestrator has intent_cache; cached intents go through dispatch ─
    try:
        import dataclasses as _dc
        from jeenom.turn_orchestrator import TurnOrchestrator
        from jeenom.schemas import OperatorIntent as _OI

        # 4a: intent_cache is a declared dataclass field
        to_field_names = (
            {f.name for f in _dc.fields(TurnOrchestrator)}
            if _dc.is_dataclass(TurnOrchestrator)
            else set()
        )
        has_field = "intent_cache" in to_field_names
        metrics["op4_orchestrator_has_intent_cache_field"] = has_field
        details["op4_to_field_names"] = sorted(to_field_names)

        # 4b: dispatching a cached metric-definition utterance sets last_operator_intent
        session = make_session()
        session.last_operator_intent = None
        utt = "create a metric called probetest which is min(euclidean, manhattan)"
        try:
            session.handle_utterance(utt)
        except Exception:
            pass
        # After handle_utterance for a cached intent, last_operator_intent must be set
        intent_set = session.last_operator_intent is not None
        correct_dispatch = (
            intent_set
            and isinstance(session.last_operator_intent, _OI)
            and session.last_operator_intent.intent_type == "primitive_definition"
        )
        metrics["op4_cached_intent_routes_through_dispatch"] = correct_dispatch
        details["op4_last_operator_intent_type"] = (
            session.last_operator_intent.intent_type
            if isinstance(session.last_operator_intent, _OI)
            else repr(session.last_operator_intent)
        )
    except Exception as exc:
        metrics["op4_orchestrator_has_intent_cache_field"] = False
        metrics["op4_cached_intent_routes_through_dispatch"] = False
        details["op4_dispatch_test_error"] = f"{type(exc).__name__}: {exc}"

    metrics["intent_fidelity_cache_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="intent_fidelity_cache_holds")


if __name__ == "__main__":
    raise SystemExit(main())
