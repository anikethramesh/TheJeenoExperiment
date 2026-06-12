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
