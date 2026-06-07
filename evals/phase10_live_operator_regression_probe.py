"""Phase 10 live operator regression probe.

This catches the failure mode where structural evals pass while the live
operator loop contradicts itself: an answer is produced from a plan whose
RequestPlan/ReadinessGraph still say unsupported/refuse.
"""
from __future__ import annotations

from typing import Any

from harness import emit_result, first_line, make_session


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        session = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)
        ranked_response = session.handle_utterance(
            "what is the distance of all the doors from you"
        )
        details["ranked_response"] = first_line(ranked_response)
        details["ranked_intent_type"] = (
            session.last_operator_intent.intent_type
            if session.last_operator_intent is not None
            else None
        )
        details["ranked_plan_objective"] = (
            session.last_request_plan.objective_type
            if session.last_request_plan is not None
            else None
        )
        details["ranked_next_action"] = (
            session.last_readiness_graph.next_action
            if session.last_readiness_graph is not None
            else None
        )
        metrics["ranked_query_answers"] = "DOORS RANKED BY MANHATTAN DISTANCE" in ranked_response
        metrics["ranked_query_records_status_query"] = (
            session.last_operator_intent is not None
            and session.last_operator_intent.intent_type == "status_query"
        )
        metrics["ranked_query_records_query_plan"] = (
            session.last_request_plan is not None
            and session.last_request_plan.objective_type == "query"
        )
        metrics["ranked_query_readiness_answers_not_refuses"] = (
            session.last_readiness_graph is not None
            and session.last_readiness_graph.next_action == "answer_query"
        )

        followup_response = session.handle_utterance("what is the euclidean distance")
        details["followup_response"] = first_line(followup_response)
        followup_handles = {
            step.required_handle
            for step in (session.last_request_plan.steps if session.last_request_plan else [])
            if step.required_handle is not None
        }
        details["followup_handles"] = sorted(followup_handles)
        details["followup_next_action"] = (
            session.last_readiness_graph.next_action
            if session.last_readiness_graph is not None
            else None
        )
        metrics["metric_followup_not_unresolved"] = "I didn't understand" not in followup_response
        metrics["metric_followup_uses_previous_context"] = (
            "grounding.all_doors.ranked.euclidean.agent" in followup_handles
        )
        metrics["metric_followup_routes_to_answer_or_synthesis"] = (
            session.last_readiness_graph is not None
            and session.last_readiness_graph.next_action
            in {"answer_query", "propose_synthesis"}
        )

        unsupported = make_session(env_id="MiniGrid-GoToDoor-16x16-v0", seed=8)
        first = unsupported.handle_utterance(
            "synthesize a new distance metric named convenientDistance"
        )
        cache_size_after_first = len(unsupported.request_plan_reuse_cache.entries)
        second = unsupported.handle_utterance("another unknown operator request")
        details["unsupported_first"] = first_line(first)
        details["unsupported_second"] = first_line(second)
        details["unsupported_cache_size"] = len(unsupported.request_plan_reuse_cache.entries)
        details["unsupported_last_reuse"] = (
            unsupported.last_plan_reuse_verdict.verdict
            if unsupported.last_plan_reuse_verdict is not None
            else None
        )
        metrics["unsupported_refuse_plan_not_cached"] = (
            cache_size_after_first == 0
            and len(unsupported.request_plan_reuse_cache.entries) == 0
        )
        metrics["unsupported_turn_not_marked_reused"] = (
            unsupported.last_plan_reuse_verdict is None
        )
        metrics["unsupported_remains_honest_non_execution"] = (
            "I didn't understand" in first and "I didn't understand" in second
        )
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        details["error"] = f"{type(exc).__name__}: {exc}"
        for key in (
            "ranked_query_answers",
            "ranked_query_records_status_query",
            "ranked_query_records_query_plan",
            "ranked_query_readiness_answers_not_refuses",
            "metric_followup_not_unresolved",
            "metric_followup_uses_previous_context",
            "metric_followup_routes_to_answer_or_synthesis",
            "unsupported_refuse_plan_not_cached",
            "unsupported_turn_not_marked_reused",
            "unsupported_remains_honest_non_execution",
        ):
            metrics[key] = False

    metrics["phase10_live_operator_regression_holds"] = all(metrics.values())
    return emit_result(
        metrics,
        details,
        pass_metric="phase10_live_operator_regression_holds",
    )


if __name__ == "__main__":
    raise SystemExit(main())
