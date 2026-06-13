"""ORPI 12C: KnowledgeBase access is scoped and channel-gated."""
from __future__ import annotations

import ast
import tempfile
from pathlib import Path

from harness import ROOT, emit_result, make_session


KB_METHODS = {"teach", "recall", "forget", "search", "all_concepts", "_is_sequence"}


def _direct_kb_method_calls(rel_path: str) -> list[str]:
    tree = ast.parse((ROOT / rel_path).read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in KB_METHODS:
            continue
        value = func.value
        if isinstance(value, ast.Attribute) and value.attr == "knowledge_base":
            hits.append(f"{rel_path}:{node.lineno}:{func.attr}")
    return hits


def _effect_manifest():
    from jeenom.orpi import OrpiContract, OrpiManifest
    from jeenom.schemas import PrimitiveSpec

    effect_spec = PrimitiveSpec(
        name="meta.effect_vocab",
        primitive_type="meta",
        layer="claims",
        description="Effect vocabulary entry.",
        postconditions=["object_state_delta"],
    )
    return OrpiManifest(
        substrate_id="probe",
        substrate_fingerprint="probe:v1",
        object_vocabulary=[],
        primitives=[OrpiContract.from_primitive_spec(effect_spec)],
    )


def _universal_plan():
    from jeenom.schemas import RequestPlan, RequestPlanStep

    return RequestPlan(
        request_id="request:universal-kb-write",
        original_utterance="state delta recipe",
        objective_type="procedure",
        objective_summary="universal recipe",
        steps=[
            RequestPlanStep(
                step_id="delta",
                layer="claims",
                operation="derive",
                required_handle="object_state_delta",
            )
        ],
        expected_response="answer_query",
    )


def main() -> int:
    from jeenom.knowledge_base import KnowledgeBase
    from jeenom.turn_orchestrator import KnowledgeChannel

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}

    forbidden = []
    for rel_path in (
        "jeenom/operator_station.py",
        "jeenom/representation.py",
        "jeenom/schemas.py",
    ):
        forbidden.extend(_direct_kb_method_calls(rel_path))
    details["forbidden_direct_kb_calls"] = forbidden
    metrics["kb_access_routes_through_channel_static"] = not forbidden

    channel = KnowledgeChannel(KnowledgeBase(storage_path=None))
    site = channel.teach("site fact", "red door is delivery target", writer="operator")
    embodiment = channel.teach(
        "oem dock",
        "dock and charge",
        writer="manifest",
        scope="embodiment",
    )
    universal = channel.teach(
        "state delta",
        "object state delta recipe",
        plan=_universal_plan(),
        writer="synthesizer",
        scope=None,
        manifest=_effect_manifest(),
    )
    try:
        channel.teach("bad flash", "temporary", writer="operator", scope="episodic")
    except ValueError as exc:
        details["episodic_rejection"] = str(exc)
        episodic_rejected = True
    else:
        episodic_rejected = False
    try:
        channel.teach("bad site", "site write", writer="synthesizer", scope="site")
    except PermissionError as exc:
        details["bad_site_writer_rejection"] = str(exc)
        bad_writer_rejected = True
    else:
        bad_writer_rejected = False
    metrics["scope_writer_policy_enforced"] = (
        site.scope == "site"
        and embodiment.scope == "embodiment"
        and universal.scope == "universal"
        and episodic_rejected
        and bad_writer_rejected
    )

    channel.invalidate_site(reason="map_changed")
    site_after = channel.recall("site fact")
    universal_after_site = channel.recall("state delta")
    channel.invalidate_substrate(reason="substrate_changed")
    embodiment_after = channel.recall("oem dock")
    universal_after_substrate = channel.recall("state delta")
    metrics["scope_invalidation_preserves_universal_only"] = (
        site_after is None
        and embodiment_after is None
        and universal_after_site is not None
        and universal_after_substrate is not None
    )

    session = make_session(memory_root=Path(tempfile.mkdtemp()))
    teach_result = session.handle_utterance("remember bingo means go to the red door")
    teach_episode = teach_result.labelled_episode.as_dict()
    metrics["kb_write_emits_to_labelled_episode"] = any(
        item.get("event") == "knowledge_write"
        and item.get("scope") == "site"
        for item in teach_episode["steering"]["knowledge"]
    )
    recall_result = session.handle_utterance("bingo")
    recall_episode = recall_result.labelled_episode.as_dict()
    details["recall_reuse_counters"] = recall_episode["steering"]["kb_reuse_counters"]
    metrics["kb_reuse_counters_recorded_per_scope"] = (
        recall_episode["steering"]["kb_reuse_counters"].get("site", 0) >= 1
    )

    metrics["authority_kb_channel_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="authority_kb_channel_holds")


if __name__ == "__main__":
    raise SystemExit(main())
