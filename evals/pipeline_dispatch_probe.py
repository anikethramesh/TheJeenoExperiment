"""Op 3b probe: knowledge-type routing in TurnOrchestrator.dispatch."""
from __future__ import annotations

import ast as _ast
import inspect
import textwrap
from pathlib import Path
from typing import Any

from harness import emit_result, make_session

ROOT = Path(__file__).resolve().parents[1]

_KNOWLEDGE_TYPES = frozenset({"claim", "procedure", "provenance", "action", "control"})

_EXPECTED_MAPPING: dict[str, str] = {
    "status_query": "claim",
    "claim_reference": "claim",
    "cache_query": "claim",
    "concept_teach": "procedure",
    "concept_recall": "procedure",
    "procedure_recall": "procedure",
    "sequence_instruction": "procedure",
    "primitive_definition": "provenance",
    "knowledge_update": "provenance",
    "task_instruction": "action",
    "motor_command": "action",
    "motor_sequence": "action",
    "conditional_sense_motor": "action",
    "mission_contract": "action",
    "reset": "control",
    "quit": "control",
    "accept_proposal": "control",
    "reject_proposal": "control",
    "unsupported": "control",
    "ambiguous": "control",
}


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        from jeenom.schemas import OperatorIntent, OPERATOR_INTENT_TYPES
        from jeenom.turn_orchestrator import TurnOrchestrator
    except Exception as exc:
        details["import_error"] = f"{type(exc).__name__}: {exc}"
        metrics["op3b_knowledge_type_property"] = False
        metrics["op3b_all_intent_types_mapped"] = False
        metrics["op3b_dispatch_no_intent_type_chain"] = False
        metrics["op3b_non_control_produces_readiness_graph"] = False
        metrics["pipeline_dispatch_holds"] = False
        return emit_result(metrics, details, pass_metric="pipeline_dispatch_holds")

    # 3b-1: OperatorIntent has a knowledge_type computed property
    has_property = isinstance(
        getattr(OperatorIntent, "knowledge_type", None), property
    )
    metrics["op3b_knowledge_type_property"] = has_property
    details["op3b_knowledge_type_type"] = type(
        getattr(OperatorIntent, "knowledge_type", None)
    ).__name__

    # 3b-2: every OPERATOR_INTENT_TYPE maps to a knowledge_type via the property
    unmapped: list[str] = []
    wrong_type: list[str] = []
    for itype in OPERATOR_INTENT_TYPES:
        try:
            sample = OperatorIntent(intent_type=itype)
            kt = sample.knowledge_type
        except Exception as exc:
            unmapped.append(f"{itype}: {exc}")
            continue
        if kt not in _KNOWLEDGE_TYPES:
            wrong_type.append(f"{itype} → {kt!r} (not in {sorted(_KNOWLEDGE_TYPES)})")
        expected = _EXPECTED_MAPPING.get(itype)
        if expected is not None and kt != expected:
            wrong_type.append(f"{itype}: expected {expected!r}, got {kt!r}")
    metrics["op3b_all_intent_types_mapped"] = not unmapped and not wrong_type
    details["op3b_unmapped_types"] = unmapped
    details["op3b_wrong_type_mappings"] = wrong_type

    # 3b-3: TurnOrchestrator.dispatch has no if/elif chain on intent_type strings
    try:
        dispatch_source = inspect.getsource(TurnOrchestrator.dispatch)
        # dedent so the AST parse doesn't fail on leading indent
        dispatch_source = textwrap.dedent(dispatch_source)
        tree = _ast.parse(dispatch_source)
        intent_type_comparisons: list[str] = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Compare):
                for comp in node.comparators:
                    if isinstance(comp, _ast.Constant) and isinstance(comp.value, str):
                        # Check if any of the OPERATOR_INTENT_TYPES appear as comparators
                        if comp.value in set(OPERATOR_INTENT_TYPES):
                            # Find the left side to confirm it's intent_type access
                            if isinstance(node.left, _ast.Attribute):
                                if node.left.attr == "intent_type":
                                    intent_type_comparisons.append(
                                        f"line {node.lineno}: {comp.value!r}"
                                    )
        metrics["op3b_dispatch_no_intent_type_chain"] = not intent_type_comparisons
        details["op3b_dispatch_intent_type_comparisons"] = intent_type_comparisons
    except Exception as exc:
        metrics["op3b_dispatch_no_intent_type_chain"] = False
        details["op3b_dispatch_ast_error"] = f"{type(exc).__name__}: {exc}"

    # 3b-4: non-control intents produce a ReadinessGraph (last_readiness_graph set)
    non_control_without_graph: list[str] = []
    try:
        session = make_session()
        # Build one OperatorIntent for each non-control knowledge_type and dispatch it
        non_control_samples = [
            # claim
            OperatorIntent(intent_type="cache_query"),
            # procedure
            OperatorIntent(intent_type="concept_teach", concept_name="t", concept_utterance="go to the red door"),
            # provenance
            OperatorIntent(intent_type="knowledge_update", knowledge_update={}),
            # action
            OperatorIntent(intent_type="motor_command", action_name="move_forward", repeat_count=1),
        ]
        for sample_intent in non_control_samples:
            session.last_readiness_graph = None
            try:
                session.turn_orchestrator.dispatch(session, sample_intent, "test")
            except Exception:
                pass  # some intents may raise — we only check graph was set
            if session.last_readiness_graph is None:
                non_control_without_graph.append(sample_intent.intent_type)
    except Exception as exc:
        details["op3b_readiness_graph_error"] = f"{type(exc).__name__}: {exc}"
        non_control_without_graph = ["<session error>"]
    metrics["op3b_non_control_produces_readiness_graph"] = not non_control_without_graph
    details["op3b_non_control_without_graph"] = non_control_without_graph

    metrics["pipeline_dispatch_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="pipeline_dispatch_holds")


if __name__ == "__main__":
    raise SystemExit(main())
