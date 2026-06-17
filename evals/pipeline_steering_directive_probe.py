"""Phase 13A: typed steering directives reshape plan assembly.

Hostile probe. Proves that the operator's HOW (budget/risk/stopping-rule), parsed as
a typed SteeringDirective and routed through verification, demonstrably changes the
typed RequestPlan / ReadinessGraph and the outbound LabelledEpisode — not just the
prose. Same WHAT + different steering => different verdict.

Must fail on a repo without the steering layer (no SteeringDirective parsing, no risk
gate in readiness, no budget cap in the Spine loop).
"""
from __future__ import annotations

from typing import Any

from harness import emit_result, make_session


SEED = 8
ENV = "MiniGrid-GoToDoor-8x8-v0"
METRIC_KEYS = (
    "baseline_executes_without_steering",
    "embedded_risk_directive_is_parsed",
    "risk_query_only_blocks_actuation_in_readiness",
    "risk_query_only_issues_no_execution_ticket",
    "embedded_budget_directive_is_parsed",
    "budget_folds_into_plan_step_constraints",
    "budget_cap_halts_execution_with_budget_exhausted",
    "steered_episode_trace_carries_active_directive",
    "baseline_episode_trace_has_no_active_directive",
    "budget_exhausted_maps_to_missing_authority",
)


def _steps(plan: Any) -> list[Any]:
    return list(getattr(plan, "steps", []) or [])


def _run(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    # --- Baseline: no steering -> executes (golden path). ---
    base = make_session(env_id=ENV, seed=SEED)
    base_result = base.handle_utterance("go to the red door")
    base_graph = base.last_readiness_graph
    details["baseline_next_action"] = getattr(base_graph, "next_action", None)
    metrics["baseline_executes_without_steering"] = (
        getattr(base_graph, "next_action", None) == "execute_task"
        and base.last_execution_ticket is not None
    )

    # --- Risk: query_only on a go-to (actuation) task -> authorization withdrawn. ---
    risk_sess = make_session(env_id=ENV, seed=SEED)
    risk_sess.handle_utterance("go to the red door, query only")
    risk_intent = risk_sess.last_operator_intent
    risk_directive = getattr(risk_intent, "steering_directive", None)
    details["risk_directive"] = (
        risk_directive.as_dict() if risk_directive is not None else None
    )
    metrics["embedded_risk_directive_is_parsed"] = (
        risk_directive is not None and risk_directive.risk == "query_only"
    )
    risk_graph = risk_sess.last_readiness_graph
    details["risk_next_action"] = getattr(risk_graph, "next_action", None)
    details["risk_graph_status"] = getattr(risk_graph, "graph_status", None)
    metrics["risk_query_only_blocks_actuation_in_readiness"] = (
        getattr(risk_graph, "graph_status", None) == "needs_authorization"
        and getattr(risk_graph, "next_action", None) != "execute_task"
    )
    metrics["risk_query_only_issues_no_execution_ticket"] = (
        risk_sess.last_execution_ticket is None
    )

    # --- Budget: max_steps cap folds into the plan and halts the Spine loop. ---
    budget_sess = make_session(env_id=ENV, seed=SEED)
    budget_result = budget_sess.handle_utterance("go to the red door using at most 1 step")
    budget_intent = budget_sess.last_operator_intent
    budget_directive = getattr(budget_intent, "steering_directive", None)
    details["budget_directive"] = (
        budget_directive.as_dict() if budget_directive is not None else None
    )
    metrics["embedded_budget_directive_is_parsed"] = (
        budget_directive is not None
        and isinstance(budget_directive.budget, dict)
        and budget_directive.budget.get("max_steps") == 1
    )
    plan_steps = _steps(budget_sess.last_request_plan)
    metrics["budget_folds_into_plan_step_constraints"] = any(
        "steering_budget" in (getattr(step, "constraints", {}) or {})
        for step in plan_steps
    )
    failure = getattr(budget_result, "failure_outcome", None)
    final_state = (budget_sess.last_result or {}).get("final_state", {})
    details["budget_failure_category"] = getattr(failure, "category", None)
    details["budget_task_complete"] = final_state.get("task_complete")
    metrics["budget_cap_halts_execution_with_budget_exhausted"] = (
        final_state.get("task_complete") is False
        and getattr(failure, "category", None) == "budget_exhausted"
    )

    # --- Trace: the active directive is recorded in the LabelledEpisode steering field. ---
    budget_episode = budget_result.labelled_episode.as_dict()
    base_episode = base_result.labelled_episode.as_dict()
    steered_steering = budget_episode.get("steering", {})
    metrics["steered_episode_trace_carries_active_directive"] = (
        isinstance(steered_steering.get("steering_directive"), dict)
        and steered_steering["steering_directive"].get("budget", {}).get("max_steps") == 1
    )
    metrics["baseline_episode_trace_has_no_active_directive"] = (
        base_episode.get("steering", {}).get("steering_directive") is None
    )

    # --- Taxonomy: a budget stop is an authorization limit, not a substrate fault. ---
    from jeenom.orpi import _map_failure_attribution

    metrics["budget_exhausted_maps_to_missing_authority"] = (
        _map_failure_attribution("budget_exhausted") == "missing_authority"
    )


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        _run(metrics, details)
    except Exception as exc:  # pragma: no cover - emitted as eval detail
        details["error"] = f"{type(exc).__name__}: {exc}"
    for key in METRIC_KEYS:
        metrics.setdefault(key, False)
    metrics["steering_directive_holds"] = all(metrics[key] for key in METRIC_KEYS)
    return emit_result(metrics, details, pass_metric="steering_directive_holds")


if __name__ == "__main__":
    raise SystemExit(main())
