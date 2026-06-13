"""ORPI conformance: every operator turn emits a LabelledEpisode."""
from __future__ import annotations

import json

from harness import emit_result, make_session


def _episode_holds(result: object) -> bool:
    episode = getattr(result, "labelled_episode", None)
    payload = episode.as_dict() if episode is not None else {}
    return (
        episode is not None
        and isinstance(payload.get("intent"), (dict, type(None)))
        and isinstance(payload.get("grounding"), dict)
        and isinstance(payload.get("plan"), dict)
        and isinstance(payload.get("authority"), dict)
        and isinstance(payload.get("execution"), dict)
        and isinstance(payload.get("verification"), dict)
        and isinstance(payload.get("attribution"), dict)
        and isinstance(payload.get("steering"), dict)
    )


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}
    session = make_session(env_id="MiniGrid-GoToDoor-8x8-v0", seed=8)
    success_result = session.handle_utterance("how far are all the doors from you")
    refusal_result = session.handle_utterance("pick up the red key")
    task_session = make_session(env_id="MiniGrid-GoToDoor-8x8-v0", seed=8)
    task_result = task_session.handle_utterance("go to the red door")
    success_episode = success_result.labelled_episode.as_dict()
    refusal_episode = refusal_result.labelled_episode.as_dict()
    task_episode = task_result.labelled_episode.as_dict()
    details["success_first_line"] = success_episode["execution"]["message"].splitlines()[0]
    details["refusal_first_line"] = refusal_episode["execution"]["message"].splitlines()[0]
    details["task_first_line"] = task_episode["execution"]["message"].splitlines()[0]
    metrics["successful_turn_emits_labelled_episode"] = _episode_holds(success_result)
    metrics["refusal_turn_emits_labelled_episode"] = _episode_holds(refusal_result)
    metrics["episode_preserves_user_visible_message"] = (
        success_episode["execution"]["message"] == success_result.message
        and refusal_episode["execution"]["message"] == refusal_result.message
    )
    metrics["episode_carries_plan_and_authority_sections"] = (
        success_episode["plan"]["request_plan"] is not None
        and success_episode["authority"]["command"] is not None
    )
    metrics["episode_carries_grounding_required_handles"] = (
        "grounding.all_doors.ranked.manhattan.agent"
        in success_episode["grounding"]["required_handles"]
        and success_episode["plan"]["required_handles"]
        == success_episode["grounding"]["required_handles"]
    )
    metrics["task_episode_carries_verification_summary"] = (
        task_episode["verification"]["task_complete"] is True
        and task_episode["execution"]["runtime_llm_calls_during_render"] == 0
        and task_episode["execution"]["cache_miss_during_render"] == 0
        and task_episode["execution"]["trace_event_count"] > 0
    )
    postconditions = task_episode["verification"]["postcondition_results"]
    passed_postconditions = {
        item["name"]
        for item in postconditions
        if isinstance(item, dict) and item.get("passed") is True
    }
    metrics["task_episode_carries_postcondition_evidence"] = (
        "task_complete" in passed_postconditions
        and "target_visible" in passed_postconditions
        and "adjacency_to_target" in passed_postconditions
    )
    task_attr = task_episode["attribution"]
    metrics["task_episode_attribution_uses_orpi_taxonomy"] = (
        "orpi_attribution" in task_attr
        and task_attr["orpi_attribution"] is None
    )
    metrics["task_episode_minigrid_task_layer_is_degenerate_boolean"] = (
        task_episode["verification"]["verification_method"] == "degenerate_boolean"
        and task_episode["verification"]["postcondition_primitive"] is None
    )
    refusal_attr = refusal_episode["attribution"]
    metrics["refusal_episode_orpi_attribution_is_none"] = (
        "orpi_attribution" in refusal_attr
        and refusal_attr["orpi_attribution"] is None
    )
    from jeenom.orpi import (
        OrpiContract,
        OrpiManifest,
        _map_failure_attribution,
        _postcondition_checker_for_handle,
    )
    from jeenom.schemas import PrimitiveSpec
    action_spec = PrimitiveSpec(
        name="action.move_forward",
        primitive_type="action",
        layer="action",
        description="Move forward one step.",
        postconditions=["position_changed"],
        postcondition_primitive="sensing.parse_grid_objects",
    )
    named_checker_manifest = OrpiManifest(
        substrate_id="probe",
        substrate_fingerprint="probe:v1",
        object_vocabulary=[],
        primitives=[OrpiContract.from_primitive_spec(action_spec)],
    )
    metrics["named_checker_manifest_resolves_postcondition_primitive"] = (
        _postcondition_checker_for_handle("action.move_forward", named_checker_manifest)
        == "sensing.parse_grid_objects"
        and _postcondition_checker_for_handle("action.unknown", named_checker_manifest) is None
        and _postcondition_checker_for_handle(None, named_checker_manifest) is None
    )
    metrics["failure_attribution_maps_correctly"] = (
        _map_failure_attribution("stuck") == "unmet_postcondition"
        and _map_failure_attribution("progress") == "unmet_postcondition"
        and _map_failure_attribution("blocking_claim") == "stale_claim"
        and _map_failure_attribution("timeout") == "substrate_fault"
        and _map_failure_attribution(None) is None
    )
    task_session2 = make_session(env_id="MiniGrid-GoToDoor-8x8-v0", seed=8)
    fail_result = task_session2.handle_utterance("pick up the red key")
    fail_episode = fail_result.labelled_episode.as_dict()
    fail_attr = fail_episode["attribution"]
    metrics["failure_episode_orpi_attribution_maps_failure_category"] = (
        "orpi_attribution" in fail_attr
        and fail_attr["failure_category"] is None
        and fail_attr["orpi_attribution"] is None
    )
    try:
        json.dumps(success_episode, sort_keys=True)
        json.dumps(refusal_episode, sort_keys=True)
        json.dumps(task_episode, sort_keys=True)
    except TypeError as exc:
        details["json_serialization_error"] = str(exc)
        metrics["episodes_are_json_serializable"] = False
    else:
        metrics["episodes_are_json_serializable"] = True
    metrics["orpi_labelled_episode_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="orpi_labelled_episode_holds")


if __name__ == "__main__":
    raise SystemExit(main())
