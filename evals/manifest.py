"""Explicit eval manifest for eval_master.py.

Suites:
  all          - every probe in this manifest
  architecture - invariant and architecture probes
  cleanup      - Phase 9 cleanup red-bar probes
  smoke        - historical smoke/regression probes
"""
from __future__ import annotations

EVAL_SPECS: list[dict[str, object]] = [
    {"file": "capability_arbitrator_probe.py", "suites": ["architecture"]},
    {"file": "capability_matcher_probe.py", "suites": ["architecture"]},
    {"file": "capability_registry_probe.py", "suites": ["architecture"]},
    {"file": "eval_golden.py", "suites": ["smoke"]},
    {"file": "eval_phase_3_5.py", "suites": ["smoke"]},
    {"file": "eval_phase_7_5.py", "suites": ["architecture"]},
    {"file": "eval_phase_7_59.py", "suites": ["architecture"]},
    {"file": "eval_phase_7_6.py", "suites": ["architecture"]},
    {"file": "eval_phase_7_8.py", "suites": ["architecture"]},
    {"file": "eval_phase_7_95.py", "suites": ["architecture"]},
    {"file": "intent_verifier_probe.py", "suites": ["architecture"]},
    {"file": "operator_intent_probe.py", "suites": ["architecture"]},
    {"file": "phase810_typed_claims_probe.py", "suites": ["architecture"]},
    {"file": "phase811_mission_contract_probe.py", "suites": ["architecture"]},
    {"file": "phase835_knowledge_base_probe.py", "suites": ["architecture"]},
    {"file": "phase845_concept_intent_probe.py", "suites": ["architecture"]},
    {"file": "phase846_unified_claim_abstraction_probe.py", "suites": ["architecture"]},
    {"file": "phase847_primitive_composition_probe.py", "suites": ["architecture"]},
    {"file": "phase848_sequential_intent_probe.py", "suites": ["architecture"]},
    {"file": "phase849_sequence_instruction_probe.py", "suites": ["architecture"]},
    {"file": "phase84_mismatch_detection_probe.py", "suites": ["architecture"]},
    {"file": "phase850_motor_command_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase852_motor_count_sequence_probe.py", "suites": ["architecture"]},
    {"file": "phase855_motor_implicit_sequence_probe.py", "suites": ["architecture"]},
    {"file": "phase89_command_registry_probe.py", "suites": ["architecture"]},
    {"file": "phase8_environment_assumption_probe.py", "suites": ["architecture"]},
    {"file": "phase8_environment_change_stale_claim_probe.py", "suites": ["architecture"]},
    {"file": "phase8_plan_reuse_probe.py", "suites": ["architecture"]},
    {"file": "phase8_vocabulary_advertising_probe.py", "suites": ["architecture"]},
    {"file": "phase91_operational_repair_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9_episode_semantics_probe.py", "suites": ["cleanup"]},
    {"file": "phase9_eval_coverage_probe.py", "suites": ["cleanup"]},
    {"file": "phase9_motor_safety_probe.py", "suites": ["cleanup"]},
    {"file": "phase9_repair_truthfulness_probe.py", "suites": ["cleanup"]},
    {"file": "phase9_request_plan_gate_probe.py", "suites": ["cleanup"]},
    {"file": "phase9b_substrate_contract_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9d_clarification_resume_gate_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9d_execution_ticket_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9d_memory_write_gate_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9d_mission_child_ticket_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9d_motor_ticket_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9d_schema_enforcement_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9d_static_architecture_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9e_block_boundary_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9e_schema_boundary_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9e_knowledge_surface_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase9e_readiness_snapshot_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase10_command_authority_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase10_side_effect_authority_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase10_substrate_adapter_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase10_operational_context_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase10_domain_helper_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase10_turn_orchestrator_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase10_runtime_package_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase10_context_planning_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "phase10_live_operator_regression_probe.py", "suites": ["architecture", "cleanup"]},
    {"file": "request_plan_probe.py", "suites": ["architecture"]},
]


def select_eval_specs(suite: str) -> list[dict[str, object]]:
    if suite == "all":
        return list(EVAL_SPECS)
    return [
        spec
        for spec in EVAL_SPECS
        if suite in set(spec.get("suites", []))
    ]
