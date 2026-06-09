"""Phase 11B hostile primitive and mission eval ladder.

This probe is intentionally adversarial. It does not add production behavior;
it exposes places where the current architecture only works for narrow phrasings
or lets low-level actions bypass the Sense -> Cortex -> Spine flow.
"""
from __future__ import annotations

import re
from typing import Any

from harness import emit_result, first_line, make_session


ENV_ID = "MiniGrid-GoToDoor-16x16-v0"
SEED = 8
RANKED_MANHATTAN = "grounding.all_doors.ranked.manhattan.agent"
INLINE_SUM_TASK = (
    "go to the third farthest door based on the sum of euclidean and "
    "manhattan distance"
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:70]


def _step_ids(plan: Any) -> list[str]:
    if plan is None:
        return []
    return [getattr(step, "step_id", "") for step in getattr(plan, "steps", [])]


def _step_handles(plan: Any) -> set[str]:
    if plan is None:
        return set()
    return {
        getattr(step, "required_handle")
        for step in getattr(plan, "steps", [])
        if getattr(step, "required_handle", None) is not None
    }


def _step_actions(plan: Any) -> list[tuple[str, int]]:
    if plan is None:
        return []
    actions: list[tuple[str, int]] = []
    for step in getattr(plan, "steps", []):
        inputs = getattr(step, "inputs", {}) or {}
        action = inputs.get("action_name")
        count = inputs.get("repeat_count")
        if action is not None:
            actions.append((str(action), int(count or 1)))
    return actions


def _not_unsupported(response: str, session: Any) -> bool:
    intent = getattr(session, "last_operator_intent", None)
    plan = getattr(session, "last_request_plan", None)
    graph = getattr(session, "last_readiness_graph", None)
    lower = response.lower()
    return (
        "i didn't understand" not in lower
        and "unsupported" not in lower
        and getattr(intent, "intent_type", None) != "unsupported"
        and getattr(plan, "objective_type", None) != "unsupported"
        and getattr(graph, "next_action", None) != "refuse"
    )


def _ranked_distance_query_holds(response: str, session: Any) -> bool:
    intent = getattr(session, "last_operator_intent", None)
    plan = getattr(session, "last_request_plan", None)
    graph = getattr(session, "last_readiness_graph", None)
    return (
        _not_unsupported(response, session)
        and getattr(intent, "intent_type", None) == "status_query"
        and getattr(plan, "objective_type", None) == "query"
        and getattr(graph, "next_action", None) == "answer_query"
        and RANKED_MANHATTAN in _step_handles(plan)
        and getattr(session, "last_execution_ticket", None) is None
        and getattr(session, "last_raw_motor_ticket", None) is None
        and getattr(session, "active_claims", None) is not None
        and bool(getattr(session.active_claims, "ranked_scene_doors", []))
    )


def _single_motor_holds(response: str, session: Any, expected_action: str, expected_count: int) -> bool:
    intent = getattr(session, "last_operator_intent", None)
    plan = getattr(session, "last_request_plan", None)
    graph = getattr(session, "last_readiness_graph", None)
    ticket = getattr(session, "last_raw_motor_ticket", None)
    return (
        _not_unsupported(response, session)
        and getattr(intent, "intent_type", None) == "motor_command"
        and getattr(plan, "objective_type", None) == "motor_control"
        and getattr(graph, "next_action", None) == "execute_motor"
        and ticket is not None
        and getattr(ticket, "action_name", None) == expected_action
        and getattr(ticket, "repeat_count", None) == expected_count
        and getattr(session, "last_execution_ticket", None) is None
    )


def _has_pending_or_last_mission(session: Any) -> bool:
    pending = getattr(session, "pending_primitive_definition", None)
    pending_mission = getattr(pending, "mission_plan", None)
    last_mission = getattr(session, "last_mission_execution_plan", None)
    return pending_mission is not None or last_mission is not None


def _record_turn(details: dict[str, Any], label: str, utterance: str, response: str, session: Any) -> None:
    plan = getattr(session, "last_request_plan", None)
    graph = getattr(session, "last_readiness_graph", None)
    intent = getattr(session, "last_operator_intent", None)
    details[label] = {
        "utterance": utterance,
        "first_line": first_line(response),
        "intent_type": getattr(intent, "intent_type", None),
        "plan_objective": getattr(plan, "objective_type", None),
        "next_action": getattr(graph, "next_action", None),
        "handles": sorted(_step_handles(plan)),
        "steps": _step_ids(plan),
        "actions": _step_actions(plan),
        "raw_motor_ticket": (
            {
                "action_name": getattr(session.last_raw_motor_ticket, "action_name", None),
                "repeat_count": getattr(session.last_raw_motor_ticket, "repeat_count", None),
            }
            if getattr(session, "last_raw_motor_ticket", None) is not None
            else None
        ),
        "execution_ticket": getattr(session, "last_execution_ticket", None) is not None,
        "pending_mission": _has_pending_or_last_mission(session),
    }


def _run_sense_paraphrases(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    prompts = [
        "what is in front of me",
        "what am I facing",
        "what object is ahead",
        "sense the cell in front",
        "look forward",
    ]
    for prompt in prompts:
        session = make_session(env_id=ENV_ID, seed=SEED)
        response = session.handle_utterance(prompt)
        label = f"sense_{_slug(prompt)}"
        _record_turn(details, label, prompt, response, session)
        plan = getattr(session, "last_request_plan", None)
        graph = getattr(session, "last_readiness_graph", None)
        metrics[f"{label}_routes_to_query_or_evidence"] = (
            _not_unsupported(response, session)
            and getattr(plan, "objective_type", None) == "query"
            and getattr(graph, "next_action", None) in {"answer_query", "ask_clarification"}
        )
        metrics[f"{label}_does_not_move"] = (
            getattr(session, "last_raw_motor_ticket", None) is None
            and getattr(session, "last_execution_ticket", None) is None
        )


def _run_spine_paraphrases(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    prompts = [
        ("take one step forward", "move_forward", 1),
        ("move forward once", "move_forward", 1),
        ("advance one cell", "move_forward", 1),
        ("step ahead", "move_forward", 1),
        ("go forward one", "move_forward", 1),
    ]
    for prompt, action, count in prompts:
        session = make_session(env_id=ENV_ID, seed=SEED)
        response = session.handle_utterance(prompt)
        label = f"spine_{_slug(prompt)}"
        _record_turn(details, label, prompt, response, session)
        metrics[f"{label}_gets_raw_motor_ticket"] = _single_motor_holds(
            response,
            session,
            action,
            count,
        )


def _run_named_procedure(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    teach_prompts = [
        "when I say fing fam foom, give me the distance to all doors",
        "remember fing fam foom means give me the distance to all doors",
        "define fing fam foom as give me the distance to all doors",
    ]
    invoke_prompts = ["fing fam foom", "do fing fam foom", "run fing fam foom"]
    for teach_prompt in teach_prompts:
        session = make_session(env_id=ENV_ID, seed=SEED)
        teach_response = session.handle_utterance(teach_prompt)
        teach_label = f"procedure_teach_{_slug(teach_prompt)}"
        _record_turn(details, teach_label, teach_prompt, teach_response, session)
        snapshot = session.representation.snapshot()
        procedure = snapshot.procedures.get("fing fam foom")
        plan = getattr(session, "last_request_plan", None)
        graph = getattr(session, "last_readiness_graph", None)
        metrics[f"{teach_label}_stores_representation_procedure"] = (
            procedure is not None and bool(procedure.get("plan"))
        )
        metrics[f"{teach_label}_records_typed_memory_update"] = (
            procedure is not None
            and getattr(plan, "objective_type", None) == "knowledge_update"
            and getattr(graph, "next_action", None) == "update_memory"
        )
        for invoke_prompt in invoke_prompts:
            response = session.handle_utterance(invoke_prompt)
            invoke_label = f"{teach_label}_invoke_{_slug(invoke_prompt)}"
            _record_turn(details, invoke_label, invoke_prompt, response, session)
            metrics[f"{invoke_label}_expands_to_ranked_distance_query"] = (
                procedure is not None and _ranked_distance_query_holds(response, session)
            )


def _run_action_procedures(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    prompts = [
        "go straight two steps and turn left",
        "move forward twice then turn left",
        "advance two cells, then face left",
    ]
    expected = [("move_forward", 2), ("turn_left", 1)]
    for prompt in prompts:
        session = make_session(env_id=ENV_ID, seed=SEED)
        response = session.handle_utterance(prompt)
        label = f"action_procedure_{_slug(prompt)}"
        _record_turn(details, label, prompt, response, session)
        actions = _step_actions(getattr(session, "last_request_plan", None))
        metrics[f"{label}_not_sequence_error"] = "SEQUENCE ERROR" not in response
        metrics[f"{label}_preserves_all_child_actions"] = actions == expected
        metrics[f"{label}_records_parent_sequence_not_final_child_only"] = (
            len(actions) == len(expected)
            and getattr(session, "last_raw_motor_ticket", None) is not None
            and getattr(session.last_raw_motor_ticket, "action_name", None) == expected[-1][0]
        )


def _run_conditional_sense_spine(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    prompts = [
        "if there is a red door in front of me, go forward, otherwise stay",
        "if I am facing a red door, step forward, else do nothing",
        "move only if the object ahead is a red door",
    ]
    for prompt in prompts:
        session = make_session(env_id=ENV_ID, seed=SEED)
        response = session.handle_utterance(prompt)
        label = f"conditional_{_slug(prompt)}"
        _record_turn(details, label, prompt, response, session)
        plan = getattr(session, "last_request_plan", None)
        steps = _step_ids(plan)
        metrics[f"{label}_does_not_execute_raw_motor_before_evidence"] = (
            getattr(session, "last_raw_motor_ticket", None) is None
            and "MOTOR COMPLETE" not in response
        )
        metrics[f"{label}_has_evidence_before_action_plan"] = (
            any("sense" in step or "evidence" in step or "observe" in step for step in steps)
            and any("execute" in step for step in steps)
        )


def _run_distance_paraphrases(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    prompts = [
        "how far are all the doors from you",
        "how far are the doors from you",
        "how far away are the doors",
        "distance to the doors",
        "show me the door distances",
        "how far is each door",
        "what are the distances to the doors",
    ]
    for prompt in prompts:
        session = make_session(env_id=ENV_ID, seed=SEED)
        response = session.handle_utterance(prompt)
        label = f"distance_{_slug(prompt)}"
        _record_turn(details, label, prompt, response, session)
        metrics[f"{label}_routes_to_ranked_distance_query"] = (
            _ranked_distance_query_holds(response, session)
        )

    stateful = make_session(env_id=ENV_ID, seed=SEED)
    scene_response = stateful.handle_utterance("what are the doors you see around you")
    _record_turn(details, "stateful_scene_seed", "what are the doors you see around you", scene_response, stateful)
    for prompt in ["how far are the doors from you", "rank them by distance", "show their distances"]:
        response = stateful.handle_utterance(prompt)
        label = f"stateful_distance_{_slug(prompt)}"
        _record_turn(details, label, prompt, response, stateful)
        metrics[f"{label}_binds_visible_door_set"] = _ranked_distance_query_holds(
            response,
            stateful,
        )


def _run_compound_missions(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    prompts = [
        INLINE_SUM_TASK,
        (
            "find euclidean distance to all doors, find manhattan distance to all "
            "doors, then go to the third farthest by their sum"
        ),
        "go to the second highest door by max(euclidean, manhattan)",
        "create ramesian = euclidean mod 5 and go to the farthest ramesian door",
    ]
    for prompt in prompts:
        session = make_session(env_id=ENV_ID, seed=SEED)
        proposal = session.handle_utterance(prompt)
        label = f"compound_{_slug(prompt)}"
        _record_turn(details, label, prompt, proposal, session)
        metrics[f"{label}_creates_or_uses_mission_plan"] = _has_pending_or_last_mission(session)
        approval = ""
        if getattr(session, "pending_primitive_definition", None) is not None:
            approval = session.handle_utterance("yes")
            _record_turn(details, f"{label}_approval", "yes", approval, session)
        mission = getattr(session, "last_mission_execution_plan", None)
        ticket = getattr(session, "last_execution_ticket", None)
        continuation = getattr(mission, "continuation_request_plan", None)
        continuation_steps = set(_step_ids(continuation))
        metrics[f"{label}_keeps_rank_select_execute_lineage"] = (
            mission is not None
            and {"rank_scene_doors", "select_grounded_target", "execute_task"}.issubset(
                continuation_steps
            )
        )
        metrics[f"{label}_final_ticket_carries_mission_provenance"] = (
            ticket is not None
            and mission is not None
            and getattr(ticket, "mission_id", None) == getattr(mission, "mission_id", None)
            and prompt in getattr(ticket, "provenance", {}).get("original_utterance", "")
        )
        if approval:
            metrics[f"{label}_runtime_llm_free_after_approval"] = (
                getattr(session, "last_result", None) is not None
                and session.last_result.get("runtime_llm_calls_during_render") == 0
            )


def _run_composition_invariants(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    """Red-bar tests for primitive composition architectural invariants.

    Invariant 1 — plan-trust: when a grounding_query_plan is present, text-parsing
    fallbacks (_comparison_from_text_or_plan, metric_from_text_or_plan) must not
    override the structured plan values, even when the utterance contains keywords
    that would normally trigger those parsers.

    Invariant 2 — pre-built continuation plan: the continuation RequestPlan must be
    built at proposal time and stored in MissionExecutionPlan.continuation_request_plan,
    not reconstructed from the original utterance at approval time.

    Invariant 3 — plan reuse on approval: resume_after_approval must reuse the
    pre-built plan (same request_id) rather than re-running build_request_plan.
    """
    from jeenom.request_planner import build_request_plan
    from jeenom.schemas import OperatorIntent

    # Use a fresh session's planning_semantics — guaranteed to be correctly wired.
    _ref_session = make_session(env_id=ENV_ID, seed=SEED)
    semantics = _ref_session.planning_semantics

    # ── Invariant 1a: comparison not extracted from text when plan is present ──
    # Adversarial: utterance contains "above" (comparison word) but the
    # grounding_query_plan explicitly has comparison: None.  No
    # filter_distance_threshold step may appear.
    adversarial_utterance = (
        "create abovemetric = euclidean mod 5 and go above to the farthest abovemetric door"
    )
    continuation_intent_adv = OperatorIntent(
        intent_type="task_instruction",
        task_type="go_to_object",
        grounding_query_plan={
            "object_type": "door",
            "operation": "select",
            "primitive_handle": "grounding.all_doors.ranked.mod_euclidean_5_0.agent",
            "metric": "mod_euclidean_5_0",
            "reference": "agent",
            "order": "descending",
            "ordinal": 1,
            "color": None,
            "exclude_colors": [],
            "distance_value": None,
            "comparison": None,
            "tie_policy": "first",
            "answer_fields": ["target", "distance"],
            "required_capabilities": ["grounding.all_doors.ranked.mod_euclidean_5_0.agent", "task.go_to_object.door"],
            "preserved_constraints": ["inline_metric", "mod_euclidean_5_0", "euclidean"],
            "metric_dependencies": ["euclidean"],
            "derived_metric": True,
            "mission_id": "mission:test",
        },
        capability_status="executable",
        required_capabilities=["grounding.all_doors.ranked.mod_euclidean_5_0.agent", "task.go_to_object.door"],
        confidence=1.0,
        reason="test",
    )
    plan_adv = build_request_plan(adversarial_utterance, continuation_intent_adv, planning_semantics=semantics)
    step_ids_adv = {s.step_id for s in plan_adv.steps}
    metrics["composition_plan_trust_comparison_not_extracted_from_adversarial_text"] = (
        "filter_distance_threshold" not in step_ids_adv
    )
    details["composition_plan_trust_comparison_step_ids"] = sorted(step_ids_adv)

    # ── Invariant 1b: metric from plan used even when not yet in metric_supported ──
    # At proposal time, "mod_euclidean_5_0" is not in the semantics metrics list yet
    # (_install_metric_semantics hasn't run). metric_from_text_or_plan must still
    # return "mod_euclidean_5_0" from the plan, not None or a fallback.
    metric_in_constraints = next(
        (
            step.constraints.get("metric")
            for step in plan_adv.steps
            if step.step_id == "rank_scene_doors"
        ),
        "MISSING",
    )
    metrics["composition_plan_trust_metric_from_plan_not_text"] = (
        metric_in_constraints == "mod_euclidean_5_0"
    )
    details["composition_plan_trust_metric_in_constraints"] = metric_in_constraints

    # ── Invariant 2: continuation plan pre-built at proposal time ──
    # After the first handle_utterance, pending_primitive_definition.mission_plan
    # must already have continuation_request_plan set with the correct step structure.
    session_prebuilt = make_session(env_id=ENV_ID, seed=SEED)
    session_prebuilt.handle_utterance(
        "create ramesian = euclidean mod 5 and go to the farthest ramesian door"
    )
    pending = getattr(session_prebuilt, "pending_primitive_definition", None)
    prebuilt_plan = getattr(getattr(pending, "mission_plan", None), "continuation_request_plan", None)
    prebuilt_step_ids = set(_step_ids(prebuilt_plan))
    metrics["composition_continuation_plan_prebuilt_at_proposal"] = (
        prebuilt_plan is not None
        and "rank_scene_doors" in prebuilt_step_ids
        and "select_grounded_target" in prebuilt_step_ids
        and "execute_task" in prebuilt_step_ids
    )
    metrics["composition_continuation_plan_no_spurious_filter_at_proposal"] = (
        prebuilt_plan is not None
        and "filter_distance_threshold" not in prebuilt_step_ids
    )
    details["composition_prebuilt_step_ids"] = sorted(prebuilt_step_ids)

    # ── Invariant 3: approval reuses pre-built plan (same request_id) ──
    prebuilt_request_id = getattr(prebuilt_plan, "request_id", None)
    session_prebuilt.handle_utterance("yes")
    final_mission = getattr(session_prebuilt, "last_mission_execution_plan", None)
    final_cont_plan = getattr(final_mission, "continuation_request_plan", None)
    final_request_id = getattr(final_cont_plan, "request_id", None)
    metrics["composition_approval_reuses_prebuilt_plan"] = (
        prebuilt_request_id is not None
        and final_request_id == prebuilt_request_id
    )
    details["composition_prebuilt_request_id"] = prebuilt_request_id
    details["composition_final_request_id"] = final_request_id


def _run_negative_controls(metrics: dict[str, bool], details: dict[str, Any]) -> None:
    controls = [
        ("go to the door", "clarify_ambiguous_target"),
        ("how far is the door from you", "clarify_ambiguous_singular_distance"),
        ("pick up the red key", "block_unsupported_task"),
        ("move to the farthest door by walking randomly", "block_random_walk_degradation"),
    ]
    for prompt, expected in controls:
        session = make_session(env_id=ENV_ID, seed=SEED)
        response = session.handle_utterance(prompt)
        label = f"negative_{expected}"
        _record_turn(details, label, prompt, response, session)
        lower = response.lower()
        no_side_effect = (
            getattr(session, "last_execution_ticket", None) is None
            and getattr(session, "last_raw_motor_ticket", None) is None
            and getattr(session, "last_result", None) is None
        )
        metrics[f"{label}_has_no_side_effect"] = no_side_effect
        if expected.startswith("clarify"):
            metrics[f"{label}_clarifies_instead_of_unsupported"] = (
                "clarify" in lower
                or getattr(session, "pending_clarification", None) is not None
            )
        elif expected == "block_random_walk_degradation":
            metrics[f"{label}_does_not_degrade_task_to_answer"] = (
                getattr(getattr(session, "last_operator_intent", None), "intent_type", None)
                != "status_query"
                and "GROUNDING ANSWER" not in response
            )
        else:
            metrics[f"{label}_blocks_or_reports_missing_skills"] = (
                "missing skills" in lower
                or "unsupported" in lower
                or "i didn't understand" in lower
            )


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        _run_sense_paraphrases(metrics, details)
        _run_spine_paraphrases(metrics, details)
        _run_named_procedure(metrics, details)
        _run_action_procedures(metrics, details)
        _run_conditional_sense_spine(metrics, details)
        _run_distance_paraphrases(metrics, details)
        _run_compound_missions(metrics, details)
        _run_composition_invariants(metrics, details)
        _run_negative_controls(metrics, details)
    except Exception as exc:  # pragma: no cover - emitted as eval detail
        details["error"] = f"{type(exc).__name__}: {exc}"
        metrics.setdefault("phase11b_probe_completed_without_exception", False)
    else:
        metrics["phase11b_probe_completed_without_exception"] = True

    details["violated_invariants"] = sorted(
        key for key, value in metrics.items() if value is False
    )
    metrics["phase11b_primitive_ladder_holds"] = all(metrics.values())
    return emit_result(
        metrics,
        details,
        pass_metric="phase11b_primitive_ladder_holds",
    )


if __name__ == "__main__":
    raise SystemExit(main())
