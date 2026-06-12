"""Phase 9 probe: repair truthfulness.

A repair event marked success must either resume the original request safely or
tell the operator that repair only cleared state and did not execute.
"""
from __future__ import annotations

from harness import emit_result, first_line, make_session, patched_env_builder

from jeenom.schemas import StationActiveClaims


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}

    with patched_env_builder():
        session = make_session(seed=42)
        session.handle_utterance("rank all the doors by manhattan distance")

        if session.active_claims is not None:
            session.active_claims = StationActiveClaims(
                scene_fingerprint=(999, 999, 999),
                ranked_scene_doors=session.active_claims.ranked_scene_doors,
                last_grounded_target=session.active_claims.last_grounded_target,
                last_grounded_rank=session.active_claims.last_grounded_rank,
                last_grounding_query=session.active_claims.last_grounding_query,
            )

        response = session.handle_utterance("go to the closest door")
        repair_events = [event.as_dict() for event in session.last_repair_events]

    success_claimed = any(event["success"] for event in repair_events)
    resumed_execution = "RUN COMPLETE" in response
    disclosed_non_execution = any(
        token in response.lower()
        for token in (
            "repair",
            "stale",
            "cleared",
            "did not execute",
            "re-ground",
            "reground",
        )
    )

    metrics["repair_events_logged"] = len(repair_events) > 0
    metrics["stale_claim_repair_success_claimed"] = any(
        event["mismatch_type"] == "STALE_CLAIMS"
        and event["repair_action"] == "REFRESH_CLAIMS"
        and event["success"]
        for event in repair_events
    )
    metrics["request_resumed_or_non_execution_disclosed"] = (
        resumed_execution or disclosed_non_execution
    )
    metrics["non_execution_disclosed_if_not_resumed"] = (
        resumed_execution or disclosed_non_execution
    )
    metrics["repair_success_is_truthful"] = (
        not success_claimed
        or resumed_execution
        or disclosed_non_execution
    )

    details["response_first_line"] = first_line(response)
    details["repair_events"] = repair_events

    return emit_result(metrics, details, pass_metric="repair_success_is_truthful")


if __name__ == "__main__":
    raise SystemExit(main())
