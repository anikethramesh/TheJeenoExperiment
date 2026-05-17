from __future__ import annotations

import unittest

from jeenom.schemas import (
    EnvironmentAssumption,
    EnvironmentIdentity,
    GroundingQueryPlan,
    OperatorIntent,
    PrimitiveManifest,
    PrimitiveCall,
    ProcedureRecipe,
    ReadinessGraph,
    ReadinessNode,
    RequestPlan,
    SchemaValidationError,
    SensePlanTemplate,
    SkillPlanTemplate,
    TargetSelector,
    TaskRequest,
    primitive_params_json_schema,
)


class JeenomSchemaTests(unittest.TestCase):
    def test_environment_identity_round_trip_and_stable_fingerprint(self):
        identity = EnvironmentIdentity(
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            grid_width=8,
            grid_height=8,
            mission="go to the red door",
            task_family="go_to_object",
            scene_fingerprint="agent=(1,1,0);step=0",
            summary={"objects": [{"type": "door", "color": "red", "x": 5, "y": 6}]},
        )
        round_trip = EnvironmentIdentity.from_dict(identity.as_dict())

        self.assertEqual(round_trip.env_id, identity.env_id)
        self.assertEqual(round_trip.summary, identity.summary)
        self.assertEqual(round_trip.fingerprint(), identity.fingerprint())

    def test_environment_identity_fingerprint_uses_stable_fields_only(self):
        base = EnvironmentIdentity(
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            grid_width=8,
            grid_height=8,
            mission="go to the red door",
            task_family="go_to_object",
            scene_fingerprint="agent=(1,1,0);step=0",
            summary={"objects": [{"type": "door", "color": "red", "x": 5, "y": 6}]},
        )
        same_environment_new_scene = EnvironmentIdentity.from_dict(
            {**base.as_dict(), "scene_fingerprint": "agent=(2,1,0);step=3"}
        )
        other_env = EnvironmentIdentity.from_dict(
            {**base.as_dict(), "env_id": "MiniGrid-GoToDoor-16x16-v0"}
        )
        other_seed = EnvironmentIdentity.from_dict({**base.as_dict(), "seed": 43})

        self.assertEqual(base.fingerprint(), same_environment_new_scene.fingerprint())
        self.assertNotEqual(base.fingerprint(), other_env.fingerprint())
        self.assertNotEqual(base.fingerprint(), other_seed.fingerprint())

    def test_environment_assumption_round_trip(self):
        assumption = EnvironmentAssumption(
            assumption_id="env.grid_size",
            kind="grid_size",
            expected={"grid_width": 8, "grid_height": 8},
            source="environment_identity",
            required=True,
            description="Grid size captured with the plan.",
        )
        round_trip = EnvironmentAssumption.from_dict(assumption.as_dict())

        self.assertEqual(round_trip.assumption_id, "env.grid_size")
        self.assertEqual(round_trip.expected["grid_width"], 8)
        self.assertTrue(round_trip.required)

    def test_request_plan_schema_accepts_environment_assumptions(self):
        plan = RequestPlan.from_dict(
            {
                "request_id": "req-env",
                "original_utterance": "go to the red door",
                "objective_type": "task",
                "objective_summary": "navigate",
                "expected_response": "execute_task",
                "preservation_signals": ["color.red", "object_type.door"],
                "environment_assumptions": [
                    {
                        "assumption_id": "env.env_id",
                        "kind": "env_id",
                        "expected": {"env_id": "MiniGrid-GoToDoor-8x8-v0"},
                        "source": "environment_identity",
                        "required": True,
                        "description": "Plan environment id.",
                    }
                ],
                "steps": [
                    {
                        "step_id": "execute_task",
                        "layer": "task",
                        "operation": "execute",
                        "required_handle": "task.go_to_object.door",
                        "outputs": ["task_result"],
                        "environment_assumption_ids": ["env.env_id"],
                    }
                ],
            }
        )

        self.assertEqual(plan.environment_assumptions[0].assumption_id, "env.env_id")
        self.assertEqual(plan.steps[0].environment_assumption_ids, ["env.env_id"])

    def test_readiness_schema_accepts_assumption_diagnostics(self):
        graph = ReadinessGraph.from_dict(
            {
                "request_id": "req-env",
                "graph_status": "environment_assumption_failed",
                "next_action": "refuse",
                "blocking_step_id": "execute_task",
                "explanation": "execute_task: Environment assumption failed",
                "violated_assumption_ids": ["env.env_id"],
                "diagnostic_assumption_ids": ["env.seed"],
                "nodes": [
                    {
                        "step_id": "execute_task",
                        "status": "environment_assumption_failed",
                        "layer": "task",
                        "operation": "execute",
                        "required_handle": "task.go_to_object.door",
                        "reason": "Environment assumption failed",
                        "blocking_dependencies": [],
                        "violated_assumption_ids": ["env.env_id"],
                        "diagnostic_assumption_ids": ["env.seed"],
                    }
                ],
            }
        )

        self.assertEqual(graph.violated_assumption_ids, ["env.env_id"])
        self.assertEqual(graph.diagnostic_assumption_ids, ["env.seed"])
        self.assertIsInstance(graph.nodes[0], ReadinessNode)

    def test_primitive_params_schema_uses_azure_safe_array_shape(self):
        schema = primitive_params_json_schema()

        self.assertEqual(
            schema["required"],
            ["color", "object_type", "target_location"],
        )
        self.assertNotIn("minItems", schema["properties"]["target_location"])
        self.assertNotIn("maxItems", schema["properties"]["target_location"])

    def test_task_request_from_dict_accepts_nullable_required_params(self):
        task = TaskRequest.from_dict(
            {
                "instruction": "go to the red door",
                "task_type": "go_to_object",
                "params": {
                    "color": "red",
                    "object_type": "door",
                    "target_location": None,
                },
                "source": "llm_compiler",
            }
        )

        self.assertEqual(task.params["color"], "red")
        self.assertIsNone(task.params["target_location"])

    def test_procedure_recipe_from_dict_accepts_string_steps(self):
        recipe = ProcedureRecipe.from_dict(
            {
                "task_type": "go_to_object",
                "steps": ["locate_object", "navigate_to_object"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "test recipe",
            }
        )
        self.assertEqual(recipe.steps, ["locate_object", "navigate_to_object"])
        self.assertTrue(recipe.validated)

    def test_sense_plan_template_from_dict_validates_symbols(self):
        template = SensePlanTemplate.from_dict(
            {
                "primitives": ["parse_grid_objects", "build_occupancy_grid"],
                "required_inputs": ["observation", "color"],
                "produces": ["world_sample", "operational_evidence", "percepts"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "test template",
            }
        )
        self.assertEqual(template.primitives[0], "parse_grid_objects")

    def test_skill_plan_template_from_dict_validates_symbols(self):
        template = SkillPlanTemplate.from_dict(
            {
                "primitives": ["plan_grid_path", "execute_next_path_action"],
                "required_inputs": ["agent_pose", "target_location", "occupancy_grid", "direction"],
                "produces": ["execution_report", "execution_context"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "test template",
            }
        )
        self.assertEqual(template.primitives[-1], "execute_next_path_action")

    def test_primitive_call_from_dict_rejects_invalid_target_location_shape(self):
        with self.assertRaises(SchemaValidationError):
            PrimitiveCall.from_dict(
                {
                    "name": "plan_grid_path",
                    "params": {
                        "color": None,
                        "object_type": "door",
                        "target_location": [1, 2, 3],
                    },
                }
            )

    def test_operator_intent_from_dict_accepts_typed_task_intent(self):
        intent = OperatorIntent.from_dict(
            {
                "intent_type": "task_instruction",
                "canonical_instruction": "go to the blue door",
                "task_type": "go_to_object",
                "target": {"color": "blue", "object_type": "door"},
                "target_selector": None,
                "capability_status": "executable",
                "knowledge_update": None,
                "reference": None,
                "status_query": None,
                "control": None,
                "clear_memory": False,
                "confidence": 0.9,
                "reason": "Parsed task.",
            }
        )

        self.assertEqual(intent.intent_type, "task_instruction")
        self.assertEqual(intent.target["color"], "blue")
        self.assertIsInstance(intent.confidence, float)

    def test_operator_intent_rejects_unsupported_object_type(self):
        with self.assertRaises(SchemaValidationError):
            OperatorIntent.from_dict(
                {
                    "intent_type": "task_instruction",
                    "canonical_instruction": "go to the red key",
                    "task_type": "go_to_object",
                    "target": {"color": "red", "object_type": "key"},
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": None,
                    "reference": None,
                    "status_query": None,
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.8,
                    "reason": "Unsupported object.",
                }
            )

    def test_operator_intent_rejects_broad_knowledge_update(self):
        with self.assertRaises(SchemaValidationError):
            OperatorIntent.from_dict(
                {
                    "intent_type": "knowledge_update",
                    "canonical_instruction": None,
                    "task_type": None,
                    "target": None,
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": {"target_color": "red"},
                    "reference": None,
                    "status_query": None,
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.8,
                    "reason": "Broad memory write.",
                }
            )

    def test_operator_intent_accepts_delivery_target_status_query(self):
        intent = OperatorIntent.from_dict(
            {
                "intent_type": "status_query",
                "canonical_instruction": None,
                "task_type": None,
                "target": None,
                "target_selector": None,
                "capability_status": "executable",
                "knowledge_update": None,
                "reference": None,
                "status_query": "delivery_target",
                "control": None,
                "clear_memory": False,
                "confidence": 0.9,
                "reason": "Question about delivery target.",
            }
        )

        self.assertEqual(intent.status_query, "delivery_target")

    def test_target_selector_accepts_closest_door_with_manhattan_agent_reference(self):
        selector = TargetSelector.from_dict(
            {
                "object_type": "door",
                "color": None,
                "exclude_color": None,
                "relation": "closest",
                "distance_metric": "manhattan",
                "distance_reference": "agent",
            }
        )

        self.assertEqual(selector.object_type, "door")
        self.assertEqual(selector.relation, "closest")
        self.assertEqual(selector.distance_metric, "manhattan")

    def test_target_selector_rejects_unsupported_object_type(self):
        with self.assertRaises(SchemaValidationError):
            TargetSelector.from_dict(
                {
                    "object_type": "key",
                    "color": "red",
                    "exclude_color": None,
                    "relation": "unique",
                    "distance_metric": None,
                    "distance_reference": None,
                }
            )

    def test_target_selector_accepts_closest_with_missing_metric_for_clarification(self):
        selector = TargetSelector.from_dict(
            {
                "object_type": "door",
                "color": None,
                "exclude_color": None,
                "relation": "closest",
                "distance_metric": None,
                "distance_reference": "agent",
            }
        )

        self.assertEqual(selector.relation, "closest")
        self.assertIsNone(selector.distance_metric)

    def test_target_selector_accepts_euclidean_metric_for_registry_arbitration(self):
        selector = TargetSelector.from_dict(
            {
                "object_type": "door",
                "color": None,
                "exclude_color": None,
                "relation": "closest",
                "distance_metric": "euclidean",
                "distance_reference": "agent",
            }
        )

        self.assertEqual(selector.distance_metric, "euclidean")

    def test_primitive_manifest_from_dict_accepts_typed_primitive_specs(self):
        manifest = PrimitiveManifest.from_dict(
            {
                "name": "test_manifest",
                "primitives": [
                    {
                        "name": "grounding.example",
                        "primitive_type": "grounding",
                        "layer": "grounding",
                        "description": "Example grounding primitive.",
                        "inputs": ["scene"],
                        "outputs": ["target"],
                        "side_effects": [],
                        "implementation_status": "implemented",
                        "safe_to_synthesize": False,
                        "runtime_binding": {"kind": "python", "value": "example"},
                    }
                ],
            }
        )

        self.assertEqual(manifest.name, "test_manifest")
        self.assertEqual(manifest.primitives[0].primitive_type, "grounding")

    def test_operator_intent_accepts_ground_target_status_query(self):
        intent = OperatorIntent.from_dict(
            {
                "intent_type": "status_query",
                "canonical_instruction": None,
                "task_type": None,
                "target": None,
                "target_selector": {
                    "object_type": "door",
                    "color": None,
                    "exclude_color": None,
                    "relation": "closest",
                    "distance_metric": "manhattan",
                    "distance_reference": "agent",
                },
                "capability_status": "executable",
                "knowledge_update": None,
                "reference": None,
                "status_query": "ground_target",
                "control": None,
                "clear_memory": False,
                "confidence": 0.9,
                "reason": "Question about closest door.",
            }
        )

        self.assertEqual(intent.status_query, "ground_target")
        self.assertEqual(intent.target_selector["relation"], "closest")

    def test_grounding_query_plan_accepts_ranked_second_farthest_plan(self):
        plan = GroundingQueryPlan.from_dict(
            {
                "object_type": "door",
                "operation": "select",
                "primitive_handle": "grounding.all_doors.ranked.manhattan.agent",
                "metric": "manhattan",
                "reference": "agent",
                "order": "descending",
                "ordinal": 2,
                "color": None,
                "exclude_colors": [],
                "distance_value": None,
                "tie_policy": "clarify",
                "answer_fields": ["target", "distance"],
                "required_capabilities": [
                    "grounding.all_doors.ranked.manhattan.agent",
                    "task.go_to_object.door",
                ],
                "preserved_constraints": ["second", "farthest", "door", "manhattan"],
            }
        )

        self.assertEqual(plan.operation, "select")
        self.assertEqual(plan.order, "descending")
        self.assertEqual(plan.ordinal, 2)

    def test_operator_intent_accepts_grounding_query_plan(self):
        intent = OperatorIntent.from_dict(
            {
                "intent_type": "status_query",
                "canonical_instruction": None,
                "task_type": None,
                "target": None,
                "target_selector": None,
                "grounding_query_plan": {
                    "object_type": "door",
                    "operation": "answer",
                    "primitive_handle": "grounding.all_doors.ranked.manhattan.agent",
                    "metric": "manhattan",
                    "reference": "agent",
                    "order": "ascending",
                    "ordinal": None,
                    "color": "red",
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "display",
                    "answer_fields": ["distance"],
                    "required_capabilities": ["grounding.all_doors.ranked.manhattan.agent"],
                    "preserved_constraints": ["red", "door", "distance"],
                },
                "capability_status": "executable",
                "knowledge_update": None,
                "reference": None,
                "status_query": "ground_target",
                "claim_reference": None,
                "control": None,
                "required_capabilities": ["grounding.all_doors.ranked.manhattan.agent"],
                "clear_memory": False,
                "confidence": 0.9,
                "reason": "Typed grounding answer plan.",
            }
        )

        self.assertEqual(intent.grounding_query_plan["color"], "red")
        self.assertEqual(intent.grounding_query_plan["answer_fields"], ["distance"])

    def test_request_plan_schema_accepts_typed_dependency_steps(self):
        plan = RequestPlan.from_dict(
            {
                "request_id": "req-1",
                "original_utterance": "go to the highest euclidean door below 10",
                "objective_type": "task",
                "objective_summary": "rank, filter, select, execute",
                "expected_response": "execute_task",
                "preservation_signals": ["metric.euclidean", "threshold"],
                "steps": [
                    {
                        "step_id": "rank_scene_doors",
                        "layer": "grounding",
                        "operation": "rank",
                        "required_handle": "grounding.all_doors.ranked.euclidean.agent",
                        "implementation_status": "synthesizable",
                        "inputs": {"object_type": "door"},
                        "outputs": ["active_claims.ranked_scene_doors"],
                        "depends_on": [],
                        "constraints": {"metric": "euclidean"},
                        "tie_policy": "clarify",
                        "memory_reads": [],
                        "memory_writes": [],
                        "scene_fingerprint_required": False,
                    },
                    {
                        "step_id": "filter_distance_threshold",
                        "layer": "claims",
                        "operation": "filter",
                        "required_handle": "claims.filter.threshold.euclidean",
                        "implementation_status": "synthesizable",
                        "inputs": {"entries": "active_claims.ranked_scene_doors"},
                        "outputs": ["filtered_candidates"],
                        "depends_on": ["rank_scene_doors"],
                        "constraints": {"comparison": "below", "threshold": 10},
                        "tie_policy": "clarify",
                        "memory_reads": ["active_claims.ranked_scene_doors"],
                        "memory_writes": [],
                        "scene_fingerprint_required": True,
                    },
                ],
            }
        )

        self.assertEqual(plan.objective_type, "task")
        self.assertEqual(plan.steps[1].depends_on, ["rank_scene_doors"])
        self.assertEqual(plan.steps[1].constraints["threshold"], 10)

    def test_readiness_graph_schema_accepts_blocking_node(self):
        graph = ReadinessGraph.from_dict(
            {
                "request_id": "req-1",
                "graph_status": "synthesizable",
                "next_action": "propose_synthesis",
                "blocking_step_id": "rank_scene_doors",
                "explanation": "rank_scene_doors needs synthesis",
                "nodes": [
                    {
                        "step_id": "rank_scene_doors",
                        "status": "synthesizable",
                        "layer": "grounding",
                        "operation": "rank",
                        "required_handle": "grounding.all_doors.ranked.euclidean.agent",
                        "reason": "safe to synthesize",
                        "blocking_dependencies": [],
                    }
                ],
            }
        )

        self.assertEqual(graph.next_action, "propose_synthesis")
        self.assertEqual(graph.nodes[0].status, "synthesizable")

    def test_grounding_query_plan_rejects_missing_ranked_handle(self):
        with self.assertRaises(SchemaValidationError):
            GroundingQueryPlan.from_dict(
                {
                    "object_type": "door",
                    "operation": "select",
                    "primitive_handle": None,
                    "metric": "manhattan",
                    "reference": "agent",
                    "order": "descending",
                    "ordinal": 2,
                    "color": None,
                    "exclude_colors": [],
                    "distance_value": None,
                    "tie_policy": "clarify",
                    "answer_fields": ["target"],
                    "required_capabilities": [],
                    "preserved_constraints": ["second", "farthest"],
                }
            )


if __name__ == "__main__":
    unittest.main()
