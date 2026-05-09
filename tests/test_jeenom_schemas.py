from __future__ import annotations

import unittest

from jeenom.schemas import (
    PrimitiveCall,
    ProcedureRecipe,
    SchemaValidationError,
    SensePlanTemplate,
    SkillPlanTemplate,
    TaskRequest,
    primitive_params_json_schema,
)


class JeenomSchemaTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
