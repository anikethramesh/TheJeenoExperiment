from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import LLMCompiler, SmokeTestCompiler
from jeenom.memory import OperationalMemory
from jeenom.minigrid_envs import ensure_custom_minigrid_envs_registered
from jeenom.minigrid_adapter import MiniGridAdapter
from jeenom.plan_cache import PlanCache
from jeenom.run_demo import prewarm_jit_cache, run_episode
from jeenom.schemas import (
    EvidenceFrame,
    ExecutionContext,
    ExecutionContract,
    Percepts,
    PrimitiveCall,
    SchemaValidationError,
    SensePlanTemplate,
    SkillPlanTemplate,
)
from jeenom.sense import MiniGridSense
from jeenom.spine import MiniGridSpine


def build_test_llm_transport():
    def transport(request):
        method = request["method_name"]
        payload = request["user_payload"]

        if method == "compile_task":
            instruction = payload["instruction"]
            return {
                "instruction": instruction,
                "task_type": "go_to_object",
                "params": {
                    "color": "red",
                    "object_type": "door",
                    "target_location": None,
                },
                "source": "llm_compiler",
            }

        if method == "compile_procedure":
            return {
                "task_type": payload["task_request"]["task_type"],
                "steps": ["locate_object", "navigate_to_object", "verify_adjacent", "done"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "Reusable high-level task recipe.",
            }

        if method == "compile_sense_plan":
            needs = set(payload["evidence_frame"]["needs"])
            context = dict(payload["execution_context"]["params"])
            context.update(payload["evidence_frame"]["context"])
            primitives: list[str] = []
            if needs & {"object_location", "agent_pose", "adjacency_to_target", "occupancy_grid"}:
                primitives.extend(["parse_grid_objects", "build_occupancy_grid"])
            if "object_location" in needs:
                primitives.append("find_object_by_color_type")
            if "agent_pose" in needs:
                primitives.append("get_agent_pose")
            if "adjacency_to_target" in needs:
                primitives.append("check_adjacency")

            required_inputs = ["observation"]
            if context.get("color") is not None:
                required_inputs.append("color")
            if context.get("object_type") is not None:
                required_inputs.append("object_type")

            return {
                "primitives": primitives,
                "required_inputs": required_inputs,
                "produces": ["world_sample", "operational_evidence", "percepts"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "Reusable sensing template.",
            }

        if method == "compile_skill_plan":
            skill = payload["execution_contract"]["skill"]
            if skill == "navigate_to_object":
                primitives = ["plan_grid_path", "execute_next_path_action"]
                required_inputs = ["agent_pose", "target_location", "occupancy_grid", "direction"]
            elif skill == "done":
                primitives = ["done"]
                required_inputs = ["adjacency_to_target"]
            else:
                primitives = [skill]
                required_inputs = ["execution_contract"]

            return {
                "primitives": primitives,
                "required_inputs": required_inputs,
                "produces": ["execution_report", "execution_context"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "Reusable skill template.",
            }

        if method == "compile_memory_updates":
            return {"updates": []}

        raise AssertionError(f"Unexpected method: {method}")

    return transport


class BadSenseCompiler(SmokeTestCompiler):
    def compile_sense_plan(self, evidence_frame, execution_context, available_sensing_primitives, memory):
        return SensePlanTemplate(
            primitives=["teleport_sensor"],
            required_inputs=["observation"],
            produces=["world_sample", "operational_evidence", "percepts"],
            source="bad_test_compiler",
            compiler_backend="bad_test_compiler",
            validated=True,
            rationale="Inject an invalid primitive for testing.",
        )


class SparseSenseCompiler(SmokeTestCompiler):
    def compile_sense_plan(self, evidence_frame, execution_context, available_sensing_primitives, memory):
        return SensePlanTemplate(
            primitives=["find_object_by_color_type", "get_agent_pose", "check_adjacency"],
            required_inputs=["observation", "color", "object_type"],
            produces=["world_sample", "operational_evidence", "percepts"],
            source="sparse_test_compiler",
            compiler_backend="sparse_test_compiler",
            validated=True,
            rationale="Omit parse_grid_objects to test runtime prerequisite recovery.",
        )


class BadSkillCompiler(SmokeTestCompiler):
    def compile_skill_plan(self, execution_contract, percepts, available_action_primitives, memory):
        return SkillPlanTemplate(
            primitives=["teleport"],
            required_inputs=["execution_contract"],
            produces=["execution_report", "execution_context"],
            source="bad_test_compiler",
            compiler_backend="bad_test_compiler",
            validated=True,
            rationale="Inject an invalid action primitive for testing.",
        )


class InvalidDirectActionTemplateCompiler(SmokeTestCompiler):
    def compile_skill_plan(self, execution_contract, percepts, available_action_primitives, memory):
        if execution_contract.skill == "navigate_to_object":
            return SkillPlanTemplate(
                primitives=["plan_grid_path", "execute_next_path_action", "done"],
                required_inputs=["agent_pose", "target_location", "occupancy_grid", "direction"],
                produces=["execution_report", "execution_context"],
                source="invalid_test_compiler",
                compiler_backend="invalid_test_compiler",
                validated=True,
                rationale="Incorrectly emits done inside navigate_to_object.",
            )
        if execution_contract.skill == "done":
            return SkillPlanTemplate(
                primitives=["plan_grid_path", "execute_next_path_action"],
                required_inputs=["agent_pose", "target_location", "occupancy_grid", "direction"],
                produces=["execution_report", "execution_context"],
                source="invalid_test_compiler",
                compiler_backend="invalid_test_compiler",
                validated=True,
                rationale="Incorrectly emits pathfinding for done.",
            )
        if execution_contract.skill == "turn_right":
            return SkillPlanTemplate(
                primitives=["plan_grid_path", "turn_right"],
                required_inputs=["execution_contract"],
                produces=["execution_report", "execution_context"],
                source="invalid_test_compiler",
                compiler_backend="invalid_test_compiler",
                validated=True,
                rationale="Incorrectly emits extra planning for direct action.",
            )
        return super().compile_skill_plan(
            execution_contract,
            percepts,
            available_action_primitives,
            memory,
        )


class JeenomMiniGridTests(unittest.TestCase):
    def test_custom_gotodoor_env_registration_allows_12x12(self):
        ensure_custom_minigrid_envs_registered()
        env = FullyObsWrapper(gym.make("MiniGrid-GoToDoor-12x12-v0"))
        try:
            observation, _ = env.reset(seed=42)
            self.assertIsNotNone(observation)
        finally:
            env.close()

    def test_smoke_test_compiler_solves_goto_door(self):
        result = run_episode(
            instruction="go to the red door",
            compiler_name="smoke_test",
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            max_loops=64,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )
        self.assertEqual(result["readiness"]["status"], "executable")
        self.assertTrue(result["final_state"]["task_complete"])
        self.assertEqual(result["persisted_knowledge"]["last_instruction"], "go to the red door")
        self.assertFalse(result["compiler_usage"]["llm_used"])

    def test_smoke_test_compiler_solves_custom_12x12_goto_door(self):
        result = run_episode(
            instruction="go to the red door",
            compiler_name="smoke_test",
            env_id="MiniGrid-GoToDoor-12x12-v0",
            seed=42,
            max_loops=256,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )
        self.assertEqual(result["readiness"]["status"], "executable")
        self.assertTrue(result["final_state"]["task_complete"])

    def test_explicit_instruction_retargets_env_goal_when_requested_door_exists(self):
        result = run_episode(
            instruction="go to the red door",
            compiler_name="smoke_test",
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=1,
            max_loops=128,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )
        self.assertTrue(result["final_state"]["task_complete"])
        self.assertEqual(result["task"]["instruction"], "go to the red door")
        self.assertEqual(result["last_world_sample"]["target_object"]["color"], "red")

    def test_requested_target_absence_fails_cleanly_before_control_loop(self):
        result = run_episode(
            instruction="go to the red door",
            compiler_name="smoke_test",
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=12,
            max_loops=128,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )
        self.assertFalse(result["final_state"]["task_complete"])
        self.assertEqual(result["final_state"]["current_skill"], "abort")
        self.assertEqual(result["final_state"]["last_report"]["reason"], "target_absent")
        self.assertEqual(result["loop_records"], [])
        self.assertTrue(
            any(event["event"] == "target_preflight_failed" for event in result["trace_events"])
        )

    def test_env_mission_run_is_not_overridden_by_stale_target_knowledge(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            memory = OperationalMemory(root=root)
            memory.update_knowledge("target_color", "red")
            memory.update_knowledge("target_type", "door")

            result = run_episode(
                instruction=None,
                compiler_name="smoke_test",
                env_id="MiniGrid-GoToDoor-8x8-v0",
                seed=1,
                max_loops=128,
                render_mode="none",
                memory_root=root,
            )

        self.assertEqual(result["task"]["instruction"], "go to the yellow door")
        self.assertEqual(result["task"]["params"]["color"], "yellow")
        self.assertFalse(result["final_state"]["knowledge_override_active"])
        self.assertTrue(result["final_state"]["task_complete"])

    def test_runtime_rejects_unknown_sensing_primitive(self):
        env = FullyObsWrapper(gym.make("MiniGrid-GoToDoor-8x8-v0"))
        adapter = MiniGridAdapter(env)
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        compiler = BadSenseCompiler()
        sense = MiniGridSense(memory, compiler)
        try:
            adapter.reset(seed=42)
            with self.assertRaises(RuntimeError):
                sense.tick(
                    observation=adapter.observe(),
                    evidence_frame=EvidenceFrame(needs=["object_location"]),
                    execution_context=ExecutionContext(
                        active_skill="idle",
                        params={"color": "red", "object_type": "door"},
                    ),
                )
        finally:
            adapter.close()

    def test_sense_projects_world_model_sample_to_evidence_and_percepts(self):
        env = FullyObsWrapper(gym.make("MiniGrid-GoToDoor-8x8-v0"))
        adapter = MiniGridAdapter(env)
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        compiler = SmokeTestCompiler()
        sense = MiniGridSense(memory, compiler)
        try:
            adapter.reset(seed=42)
            evidence, percepts, sample, plan, cache_meta = sense.tick(
                observation=adapter.observe(),
                evidence_frame=EvidenceFrame(
                    needs=["object_location", "agent_pose", "occupancy_grid", "adjacency_to_target"],
                    context={"color": "red", "object_type": "door"},
                ),
                execution_context=ExecutionContext(
                    active_skill="idle",
                    params={"color": "red", "object_type": "door"},
                ),
            )
            self.assertEqual(sample.__class__.__name__, "WorldModelSample")
            self.assertIn("target_location", evidence.claims)
            self.assertIn("agent_pose", percepts.cues)
            self.assertGreater(len(plan), 0)
            self.assertEqual(cache_meta["cache"], "disabled")
        finally:
            adapter.close()

    def test_sense_recovers_when_plan_omits_parse_grid_objects(self):
        env = FullyObsWrapper(gym.make("MiniGrid-GoToDoor-8x8-v0"))
        adapter = MiniGridAdapter(env)
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        compiler = SparseSenseCompiler()
        sense = MiniGridSense(memory, compiler)
        try:
            adapter.reset(seed=42)
            evidence, percepts, sample, _, _ = sense.tick(
                observation=adapter.observe(),
                evidence_frame=EvidenceFrame(
                    needs=["object_location", "agent_pose", "adjacency_to_target"],
                    context={"color": "red", "object_type": "door"},
                ),
                execution_context=ExecutionContext(
                    active_skill="idle",
                    params={"color": "red", "object_type": "door"},
                ),
            )
            self.assertIsNotNone(sample.agent_pose)
            self.assertIn("agent_pose", evidence.claims)
            self.assertIn("target_location", percepts.cues)
        finally:
            adapter.close()

    def test_spine_produces_execution_report_and_context(self):
        env = FullyObsWrapper(gym.make("MiniGrid-GoToDoor-8x8-v0"))
        adapter = MiniGridAdapter(env)
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        compiler = SmokeTestCompiler()
        sense = MiniGridSense(memory, compiler)
        spine = MiniGridSpine(memory, adapter, compiler)
        try:
            adapter.reset(seed=42)
            _, percepts, _, _, _ = sense.tick(
                observation=adapter.observe(),
                evidence_frame=EvidenceFrame(
                    needs=["object_location", "agent_pose", "occupancy_grid"],
                    context={"color": "red", "object_type": "door"},
                ),
                execution_context=ExecutionContext(
                    active_skill="idle",
                    params={"color": "red", "object_type": "door"},
                ),
            )
            report, context, _, _ = spine.tick(
                execution_contract=ExecutionContract(
                    skill="navigate_to_object",
                    params={
                        "color": "red",
                        "object_type": "door",
                        "target_location": percepts.cues["target_location"],
                    },
                ),
                percepts=percepts,
            )
            self.assertEqual(report.__class__.__name__, "ExecutionReport")
            self.assertEqual(context.__class__.__name__, "ExecutionContext")
        finally:
            adapter.close()

    def test_knowledge_and_episodic_memory_are_separated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            memory = OperationalMemory(root=root)
            memory.update_knowledge("target_color", "blue")
            memory.update_episodic_memory("known_target_location", (1, 2))

            reloaded = OperationalMemory(root=root)
            self.assertEqual(reloaded.knowledge["target_color"], "blue")
            self.assertIsNone(reloaded.episodic_memory["known_target_location"])

    def test_llm_compiler_falls_back_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            compiler = LLMCompiler(api_key=None)
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        task = compiler.compile_task(
            "go to the red door",
            available_task_primitives=__import__(
                "jeenom.primitive_library",
                fromlist=["TASK_PRIMITIVES"],
            ).TASK_PRIMITIVES,
            memory=memory,
        )
        self.assertEqual(task.task_type, "go_to_object")
        self.assertTrue(any("OPENROUTER_API_KEY not set" in log for log in compiler.logs))
        self.assertFalse(compiler.usage_summary()["llm_used"])

    def test_llm_compiler_reports_llm_usage_on_success(self):
        compiler = LLMCompiler(
            api_key="test-key",
            transport=build_test_llm_transport(),
        )
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        task = compiler.compile_task(
            "go to the red door",
            available_task_primitives=__import__(
                "jeenom.primitive_library",
                fromlist=["TASK_PRIMITIVES"],
            ).TASK_PRIMITIVES,
            memory=memory,
        )
        self.assertEqual(task.source, "llm_compiler")
        usage = compiler.usage_summary()
        self.assertTrue(usage["llm_used"])
        self.assertEqual(usage["total_requested_max_tokens"], 256)
        self.assertEqual(usage["call_history"][0]["requested_max_tokens"], 256)
        self.assertTrue(any("compile_task requested max_tokens=256" in log for log in compiler.logs))

    def test_llm_compiler_falls_back_on_invalid_output(self):
        compiler = LLMCompiler(
            api_key="test-key",
            transport=lambda _: "not a valid object",
        )
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        task = compiler.compile_task(
            "go to the red door",
            available_task_primitives=__import__(
                "jeenom.primitive_library",
                fromlist=["TASK_PRIMITIVES"],
            ).TASK_PRIMITIVES,
            memory=memory,
        )
        self.assertEqual(task.task_type, "go_to_object")
        self.assertTrue(any("falling back" in log for log in compiler.logs))

    def test_llm_compiler_rejects_unknown_primitive_and_falls_back(self):
        compiler = LLMCompiler(
            api_key="test-key",
            transport=lambda request: {
                "task_type": "go_to_object",
                "steps": ["teleport"],
                "source": "llm_compiler",
                "compiler_backend": "llm_compiler",
                "validated": True,
                "rationale": "bad output",
            }
            if request["method_name"] == "compile_procedure"
            else build_test_llm_transport()(request),
        )
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        task = compiler.fallback.compile_task(
            "go to the red door",
            __import__("jeenom.primitive_library", fromlist=["TASK_PRIMITIVES"]).TASK_PRIMITIVES,
            memory,
        )
        procedure = compiler.compile_procedure(
            task,
            __import__("jeenom.primitive_library", fromlist=["TASK_PRIMITIVES"]).TASK_PRIMITIVES,
            memory,
        )
        self.assertEqual(
            procedure.steps,
            ["locate_object", "navigate_to_object", "verify_adjacent", "done"],
        )

    def test_spine_corrects_invalid_navigate_template_instead_of_failing(self):
        env = FullyObsWrapper(gym.make("MiniGrid-GoToDoor-8x8-v0"))
        adapter = MiniGridAdapter(env)
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        sense = MiniGridSense(memory, SmokeTestCompiler())
        spine = MiniGridSpine(memory, adapter, BadSkillCompiler())
        try:
            adapter.reset(seed=42)
            _, percepts, _, _, _ = sense.tick(
                observation=adapter.observe(),
                evidence_frame=EvidenceFrame(
                    needs=["object_location", "agent_pose", "occupancy_grid"],
                    context={"color": "red", "object_type": "door"},
                ),
                execution_context=ExecutionContext(
                    active_skill="idle",
                    params={"color": "red", "object_type": "door"},
                ),
            )
            report, _, plan, _ = spine.tick(
                execution_contract=ExecutionContract(
                    skill="navigate_to_object",
                    params={
                        "color": "red",
                        "object_type": "door",
                        "target_location": percepts.cues["target_location"],
                    },
                ),
                percepts=percepts,
            )
            self.assertEqual([step.name for step in plan], ["plan_grid_path", "execute_next_path_action"])
            self.assertIn(report.status, {"running", "succeeded"})
        finally:
            adapter.close()

    def test_llm_mode_with_cache_passes(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        result = run_episode(
            instruction="go to the red door",
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            max_loops=64,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
            use_cache=True,
        )
        self.assertTrue(result["compiler_usage"]["llm_used"])
        self.assertTrue(result["final_state"]["task_complete"])
        self.assertGreaterEqual(result["plan_cache"]["hits"], 1)
        self.assertIn("entries", result["plan_cache"])

    def test_no_cache_produces_repeated_compiler_calls(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        result = run_episode(
            instruction="go to the red door",
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            max_loops=64,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
            use_cache=False,
        )
        sense_calls = sum(
            1
            for call in result["compiler_usage"]["call_history"]
            if call["method_name"] == "compile_sense_plan" and call["backend"] == "llm_compiler"
        )
        skill_calls = sum(
            1
            for call in result["compiler_usage"]["call_history"]
            if call["method_name"] == "compile_skill_plan" and call["backend"] == "llm_compiler"
        )
        self.assertGreaterEqual(sense_calls, len(result["loop_records"]))
        self.assertGreaterEqual(skill_calls, len(result["loop_records"]) - 1)
        self.assertEqual(result["plan_cache"]["hits"], 0)

    def test_cache_enabled_reduces_compiler_calls_below_loop_count(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        result = run_episode(
            instruction="go to the red door",
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            max_loops=64,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
            use_cache=True,
        )
        loop_count = len(result["loop_records"])
        sense_calls = sum(
            1
            for call in result["compiler_usage"]["call_history"]
            if call["method_name"] == "compile_sense_plan" and call["backend"] == "llm_compiler"
        )
        skill_calls = sum(
            1
            for call in result["compiler_usage"]["call_history"]
            if call["method_name"] == "compile_skill_plan" and call["backend"] == "llm_compiler"
        )
        self.assertLess(sense_calls, loop_count)
        self.assertLess(skill_calls, loop_count)
        self.assertGreater(result["plan_cache"]["llm_calls_saved"], 0)

    def test_prewarm_populates_jit_cache_for_common_shapes(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        plan_cache = PlanCache(enabled=True)
        from jeenom.cortex import Cortex

        cortex = Cortex(memory, compiler, plan_cache=plan_cache)
        sense = MiniGridSense(memory, compiler, plan_cache=plan_cache)
        spine = MiniGridSpine(memory, None, compiler, plan_cache=plan_cache)

        task = compiler.compile_task(
            "go to the red door",
            available_task_primitives=__import__(
                "jeenom.primitive_library",
                fromlist=["TASK_PRIMITIVES"],
            ).TASK_PRIMITIVES,
            memory=memory,
        )
        procedure = compiler.compile_procedure(
            task,
            __import__("jeenom.primitive_library", fromlist=["TASK_PRIMITIVES"]).TASK_PRIMITIVES,
            memory,
        )
        cortex.onboard_task(task, procedure)

        summary = prewarm_jit_cache(task, procedure, cortex, sense, spine, plan_cache)
        entry_types = {entry["template_type"] for entry in summary["compiled_templates"]}

        self.assertIn("procedure", entry_types)
        self.assertIn("sense", entry_types)
        self.assertIn("skill", entry_types)
        self.assertGreater(summary["cache_entries"], 0)

    def test_phase_3_5_golden_human_render_prewarm_avoids_runtime_llm_calls(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            result = run_episode(
                instruction="go to the red door",
                compiler_name="llm",
                compiler=compiler,
                env_id="MiniGrid-GoToDoor-8x8-v0",
                seed=42,
                max_loops=64,
                render_mode="human",
                memory_root=Path(tempfile.mkdtemp()),
                use_cache=True,
                prewarm=True,
            )

        self.assertTrue(result["jit_prewarm"])
        self.assertEqual(result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(result["cache_miss_during_render"], 0)
        self.assertTrue(result["final_state"]["task_complete"])
        final_records = [
            record
            for record in result["loop_records"]
            if record["skill_plan"] is not None
        ]
        self.assertGreater(len(final_records), 0)
        self.assertEqual(final_records[-1]["skill_plan"], ["done"])
        self.assertEqual(final_records[-1]["report"]["status"], "succeeded")

    def test_invalid_llm_sense_template_falls_back_before_cache(self):
        fallbacking_transport = build_test_llm_transport()

        def transport(request):
            if request["method_name"] == "compile_sense_plan":
                return {
                    "primitives": ["find_object_by_color_type", "get_agent_pose", "check_adjacency"],
                    "required_inputs": ["observation", "color", "object_type"],
                    "produces": ["world_sample", "operational_evidence", "percepts"],
                    "source": "llm_compiler",
                    "compiler_backend": "llm_compiler",
                    "validated": True,
                    "rationale": "Missing occupancy builder on purpose.",
                }
            return fallbacking_transport(request)

        compiler = LLMCompiler(api_key="test-key", transport=transport)
        result = run_episode(
            instruction="go to the red door",
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            max_loops=64,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
            use_cache=True,
        )

        self.assertTrue(result["final_state"]["task_complete"])
        self.assertTrue(
            any("corrected invalid sense template via fallback" in log for log in result["compiler_logs"])
        )
        self.assertTrue(result["loop_records"][0]["operational_evidence"]["occupancy_grid"])

    def test_cached_sense_template_still_produces_fresh_world_samples(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        result = run_episode(
            instruction="go to the red door",
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            max_loops=64,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
            use_cache=True,
        )
        poses = [
            record["world_sample"]["agent_pose"]
            for record in result["loop_records"]
            if record["world_sample"]["agent_pose"] is not None
        ]
        self.assertGreater(len({(pose["x"], pose["y"], pose["dir"]) for pose in poses}), 1)
        self.assertTrue(any(record["sense_plan_cache"] == "hit" for record in result["loop_records"][1:]))

    def test_cached_skill_template_still_produces_fresh_actions(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        result = run_episode(
            instruction="go to the red door",
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            max_loops=64,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
            use_cache=True,
        )
        actions = [record["action"] for record in result["loop_records"] if record["action"] is not None]
        self.assertIn("turn_right", actions)
        self.assertIn("move_forward", actions)
        self.assertTrue(any(record["skill_plan_cache"] == "hit" for record in result["loop_records"][1:]))

    def test_spine_corrects_invalid_done_template(self):
        compiler = InvalidDirectActionTemplateCompiler()
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        spine = MiniGridSpine(memory, None, compiler, plan_cache=PlanCache(enabled=True))

        template, _ = spine._resolve_template(
            execution_contract=ExecutionContract(
                skill="done",
                params={"color": "red", "object_type": "door", "target_location": (1, 1)},
            ),
            percepts=Percepts(cues={"adjacency_to_target": True}, source="test"),
            loop_index=0,
        )

        self.assertEqual(template.primitives, ["done"])
        self.assertTrue(any("corrected invalid done skill template" in log for log in compiler.logs))

    def test_spine_corrects_invalid_navigate_template(self):
        compiler = InvalidDirectActionTemplateCompiler()
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        spine = MiniGridSpine(memory, None, compiler, plan_cache=PlanCache(enabled=True))

        template, _ = spine._resolve_template(
            execution_contract=ExecutionContract(
                skill="navigate_to_object",
                params={"color": "red", "object_type": "door", "target_location": (1, 1)},
            ),
            percepts=Percepts(cues={"agent_pose": {"x": 0, "y": 0, "dir": 0}}, source="test"),
            loop_index=0,
        )

        self.assertEqual(template.primitives, ["plan_grid_path", "execute_next_path_action"])
        self.assertTrue(any("corrected invalid navigate_to_object skill template" in log for log in compiler.logs))

    def test_spine_corrects_invalid_direct_action_template(self):
        compiler = InvalidDirectActionTemplateCompiler()
        memory = OperationalMemory(root=Path(tempfile.mkdtemp()))
        spine = MiniGridSpine(memory, None, compiler, plan_cache=PlanCache(enabled=True))

        template, _ = spine._resolve_template(
            execution_contract=ExecutionContract(
                skill="turn_right",
                params={"color": "red", "object_type": "door", "target_location": None},
            ),
            percepts=Percepts(cues={}, source="test"),
            loop_index=0,
        )

        self.assertEqual(template.primitives, ["turn_right"])
        self.assertTrue(
            any("corrected invalid direct action skill template: turn_right" in log for log in compiler.logs)
        )


if __name__ == "__main__":
    unittest.main()
