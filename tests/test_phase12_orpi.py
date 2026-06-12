from __future__ import annotations

import json
import unittest

from evals.harness import make_session
from jeenom.minigrid_runtime_package import build_minigrid_runtime_package
from jeenom.orpi import (
    LabelledEpisode,
    OrpiContract,
    assert_no_deliberative_meta_plan_references,
)
from jeenom.schemas import (
    PrimitiveManifest,
    PrimitiveSpec,
    ReadinessGraph,
    ReadinessNode,
    RequestPlan,
    RequestPlanStep,
    SchemaValidationError,
    orpi_primitive_type_for,
)
from jeenom.capability_registry import CapabilityRegistry


class TestPhase12Orpi(unittest.TestCase):
    def test_orpi_mapping_preserves_legacy_layers(self):
        expected = {
            "sensing": "sense",
            "action": "actuation",
            "task": "meta",
            "grounding": "meta",
            "claims": "meta",
        }
        self.assertEqual(
            {key: orpi_primitive_type_for(key) for key in expected},
            expected,
        )
        spec = PrimitiveSpec(
            name="grounding.example",
            primitive_type="grounding",
            layer="grounding",
            description="Example.",
        )
        contract = OrpiContract.from_primitive_spec(spec).as_dict()
        self.assertEqual(contract["primitive_type"], "meta")
        self.assertEqual(contract["source_primitive_type"], "grounding")
        self.assertEqual(contract["layer"], "grounding")

    def test_schema_defaults_orpi_metadata(self):
        meta = PrimitiveSpec(
            name="grounding.example",
            primitive_type="grounding",
            layer="grounding",
            description="Example.",
        )
        action = PrimitiveSpec(
            name="action.move_forward",
            primitive_type="action",
            layer="action",
            description="Move.",
        )
        sense = PrimitiveSpec(
            name="sensing.parse_grid_objects",
            primitive_type="sensing",
            layer="sensing",
            description="Sense.",
        )
        self.assertEqual((meta.mode, meta.cadence, meta.invariant_level), (
            "deterministic",
            "deliberation",
            "intent",
        ))
        self.assertEqual(action.cadence, "control")
        self.assertEqual(action.invariant_level, "object_state")
        self.assertEqual(sense.cadence, "perception")
        with self.assertRaises(SchemaValidationError):
            PrimitiveSpec(
                name="action.bad",
                primitive_type="action",
                layer="action",
                description="Bad.",
                mode="deliberative",
            )

    def test_minigrid_orpi_manifest_serializes_contracts_and_metadata(self):
        package = build_minigrid_runtime_package(
            env_id="MiniGrid-GoToDoor-8x8-v0",
            render_mode="none",
        )
        manifest = package.resolve_orpi_manifest()
        payload = manifest.as_dict()
        names = {contract["name"] for contract in payload["primitives"]}
        self.assertEqual(payload["substrate_id"], "minigrid")
        self.assertEqual(payload["orpi_version"], "0")
        self.assertIn("door", payload["object_vocabulary"])
        self.assertIn("object_index", payload["symbol_mappings"])
        self.assertIn("grid", payload["frames"])
        self.assertIn("actuation", payload["risk_policy"])
        self.assertIn("action.move_forward", names)
        move = next(
            contract
            for contract in payload["primitives"]
            if contract["name"] == "action.move_forward"
        )
        self.assertEqual(move["primitive_type"], "actuation")
        self.assertEqual(move["postcondition_primitive"], "sensing.parse_grid_objects")

    def test_labelled_episode_serializes_success_and_refusal(self):
        session = make_session(env_id="MiniGrid-GoToDoor-8x8-v0", seed=8)
        success = session.handle_utterance("how far are all the doors from you")
        refusal = session.handle_utterance("pick up the red key")
        task_session = make_session(env_id="MiniGrid-GoToDoor-8x8-v0", seed=8)
        task = task_session.handle_utterance("go to the red door")
        self.assertIsInstance(success.labelled_episode, LabelledEpisode)
        self.assertIsInstance(refusal.labelled_episode, LabelledEpisode)
        self.assertIsInstance(task.labelled_episode, LabelledEpisode)
        success_payload = success.labelled_episode.as_dict()
        refusal_payload = refusal.labelled_episode.as_dict()
        task_payload = task.labelled_episode.as_dict()
        json.dumps(success_payload, sort_keys=True)
        json.dumps(refusal_payload, sort_keys=True)
        json.dumps(task_payload, sort_keys=True)
        self.assertEqual(success_payload["execution"]["message"], success.message)
        self.assertEqual(refusal_payload["execution"]["message"], refusal.message)
        self.assertIsNotNone(success_payload["plan"]["request_plan"])
        self.assertIsNotNone(refusal_payload["authority"]["command"])
        self.assertIn(
            "grounding.all_doors.ranked.manhattan.agent",
            success_payload["grounding"]["required_handles"],
        )
        self.assertEqual(
            success_payload["plan"]["required_handles"],
            success_payload["grounding"]["required_handles"],
        )
        self.assertIs(task_payload["verification"]["task_complete"], True)
        self.assertEqual(task_payload["execution"]["runtime_llm_calls_during_render"], 0)
        self.assertEqual(task_payload["execution"]["cache_miss_during_render"], 0)
        self.assertGreater(task_payload["execution"]["trace_event_count"], 0)
        passed_postconditions = {
            item["name"]
            for item in task_payload["verification"]["postcondition_results"]
            if item["passed"] is True
        }
        self.assertIn("task_complete", passed_postconditions)
        self.assertIn("target_visible", passed_postconditions)
        self.assertIn("adjacency_to_target", passed_postconditions)

    def test_deliberative_meta_plan_reference_is_rejected(self):
        primitive = PrimitiveSpec(
            name="meta.repair_with_llm",
            primitive_type="meta",
            layer="cortex",
            description="Deliberative repair.",
            mode="deliberative",
            cadence="deliberation",
            invariant_level="intent",
        )
        registry = CapabilityRegistry(PrimitiveManifest(name="probe", primitives=[primitive]))
        plan = RequestPlan(
            request_id="request:orpi-deliberative",
            original_utterance="repair it",
            objective_type="query",
            objective_summary="probe",
            steps=[
                RequestPlanStep(
                    step_id="deliberative_step",
                    layer="answer",
                    operation="answer",
                    required_handle="meta.repair_with_llm",
                )
            ],
            expected_response="answer_query",
        )
        ReadinessGraph(
            request_id=plan.request_id,
            nodes=[
                ReadinessNode(
                    step_id="deliberative_step",
                    status="executable",
                    layer="answer",
                    operation="answer",
                )
            ],
            graph_status="executable",
            next_action="answer_query",
        )
        with self.assertRaises(SchemaValidationError):
            assert_no_deliberative_meta_plan_references(plan, registry)


if __name__ == "__main__":
    unittest.main()
