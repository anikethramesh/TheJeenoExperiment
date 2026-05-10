from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import LLMCompiler, SmokeTestCompiler
from jeenom.capability_registry import CapabilityRegistry
from jeenom.memory import OperationalMemory
from jeenom.minigrid_envs import ensure_custom_minigrid_envs_registered
from jeenom.minigrid_adapter import MiniGridAdapter
from jeenom.operator_station import OperatorStationSession, classify_utterance
from jeenom.plan_cache import PlanCache
from jeenom.primitive_library import ACTION_PRIMITIVES, SENSING_PRIMITIVES, TASK_PRIMITIVES
from jeenom.run_demo import prewarm_jit_cache, run_episode
from jeenom.schemas import (
    EvidenceFrame,
    ExecutionContext,
    ExecutionContract,
    GroundedDoorEntry,
    Percepts,
    PrimitiveCall,
    SceneModel,
    SchemaValidationError,
    SensePlanTemplate,
    SkillPlanTemplate,
    StationActiveClaims,
)
from jeenom.sense import MiniGridSense
from jeenom.spine import MiniGridSpine


def build_test_llm_transport():
    def transport(request):
        method = request["method_name"]
        payload = request["user_payload"]

        if method == "compile_operator_intent":
            utterance = payload["utterance"].lower()
            if "capabilit" in utterance or "overview" in utterance:
                return {
                    "intent_type": "status_query",
                    "canonical_instruction": None,
                    "task_type": None,
                    "target": None,
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": None,
                    "reference": None,
                    "status_query": "help",
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.94,
                    "reason": "Parsed capability overview query.",
                }
            if "what do you see around you" in utterance:
                return {
                    "intent_type": "status_query",
                    "canonical_instruction": None,
                    "task_type": None,
                    "target": None,
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": None,
                    "reference": None,
                    "status_query": "scene",
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.94,
                    "reason": "Parsed fuzzy scene query.",
                }
            if "delivery target" in utterance and utterance.strip().startswith("and"):
                return {
                    "intent_type": "status_query",
                    "canonical_instruction": None,
                    "task_type": None,
                    "target": None,
                    "target_selector": None,
                    "capability_status": "unsupported",
                    "knowledge_update": None,
                    "reference": None,
                    "status_query": "delivery_target",
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.9,
                    "reason": "Parsed delivery-target query.",
                }
            if (
                ("closest" in utterance or "nearest" in utterance or "shortest" in utterance)
                and "door" in utterance
            ):
                metric = (
                    "euclidean"
                    if "euclidean" in utterance
                    else "manhattan"
                    if "manhattan" in utterance
                    else None
                )
                return {
                    "intent_type": (
                        "knowledge_update"
                        if "delivery target" in utterance or "make" in utterance
                        else "task_instruction"
                        if "go" in utterance or "navigate" in utterance or "head" in utterance
                        else "status_query"
                    ),
                    "canonical_instruction": None,
                    "task_type": (
                        "go_to_object"
                        if "go" in utterance or "navigate" in utterance or "head" in utterance
                        else None
                    ),
                    "target": None,
                    "target_selector": {
                        "object_type": "door",
                        "color": None,
                        "exclude_colors": [],
                        "relation": "closest",
                        "distance_metric": metric,
                        "distance_reference": "agent" if metric is not None else None,
                    },
                    "capability_status": (
                        "synthesizable"
                        if metric == "euclidean"
                        else "needs_clarification"
                        if metric is None
                        else "executable"
                    ),
                    "knowledge_update": (
                        {"delivery_target": None}
                        if "delivery target" in utterance or "make" in utterance
                        else None
                    ),
                    "reference": None,
                    "status_query": (
                        None
                        if "go" in utterance
                        or "navigate" in utterance
                        or "head" in utterance
                        or "make" in utterance
                        else "ground_target"
                    ),
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.9,
                    "reason": "Parsed closest-door selector.",
                }
            if "not yellow" in utterance and "door" in utterance:
                return {
                    "intent_type": "task_instruction",
                    "canonical_instruction": None,
                    "task_type": "go_to_object",
                    "target": None,
                    "target_selector": {
                        "object_type": "door",
                        "color": None,
                        "exclude_colors": ["yellow"],
                        "relation": "unique",
                        "distance_metric": None,
                        "distance_reference": None,
                    },
                    "capability_status": "executable",
                    "knowledge_update": None,
                    "reference": None,
                    "status_query": None,
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.9,
                    "reason": "Parsed unique not-yellow door selector.",
                }
            if "blue door" in utterance and (
                "head over" in utterance or "go" in utterance or "navigate" in utterance
            ):
                return {
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
                    "confidence": 0.92,
                    "reason": "Parsed fuzzy blue-door task.",
                }
            if "yellow door" in utterance:
                return {
                    "intent_type": "task_instruction",
                    "canonical_instruction": "go to the yellow door",
                    "task_type": "go_to_object",
                    "target": {"color": "yellow", "object_type": "door"},
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": None,
                    "reference": None,
                    "status_query": None,
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.91,
                    "reason": "Parsed fuzzy yellow-door task.",
                }
            if "red door" in utterance and "delivery target" in utterance:
                return {
                    "intent_type": "knowledge_update",
                    "canonical_instruction": None,
                    "task_type": None,
                    "target": None,
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": {
                        "delivery_target": {"color": "red", "object_type": "door"}
                    },
                    "reference": None,
                    "status_query": None,
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.93,
                    "reason": "Parsed delivery target knowledge.",
                }
            if "same one" in utterance or "back to the same" in utterance:
                return {
                    "intent_type": "task_instruction",
                    "canonical_instruction": None,
                    "task_type": "go_to_object",
                    "target": None,
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": None,
                    "reference": "last_target",
                    "status_query": None,
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.85,
                    "reason": "Parsed last-target reference.",
                }
            if "ask you to do last time" in utterance:
                return {
                    "intent_type": "status_query",
                    "canonical_instruction": None,
                    "task_type": None,
                    "target": None,
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": None,
                    "reference": "last_task",
                    "status_query": "last_run",
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.82,
                    "reason": "Parsed last-run status query.",
                }
            if "pick up" in utterance or "key" in utterance:
                return {
                    "intent_type": "unsupported",
                    "canonical_instruction": None,
                    "task_type": None,
                    "target": None,
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": None,
                    "reference": None,
                    "status_query": None,
                    "control": None,
                    "clear_memory": False,
                    "confidence": 1.0,
                    "reason": "Pickup/key capability is unsupported.",
                }
            return {
                "intent_type": "ambiguous",
                "canonical_instruction": None,
                "task_type": None,
                "target": None,
                "target_selector": None,
                "capability_status": "unsupported",
                "knowledge_update": None,
                "reference": None,
                "status_query": None,
                "control": None,
                "clear_memory": False,
                "confidence": 0.2,
                "reason": "Could not safely parse.",
            }

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

    def test_delivery_target_is_canonical_when_legacy_target_fields_disagree(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir) / "memory"
            memory_dir.mkdir()
            (memory_dir / "knowledge.yaml").write_text(
                'target_color: "green"\n'
                'target_type: "door"\n'
                'delivery_target: {"color": "red", "object_type": "door"}\n'
                'last_task_type: null\n'
                'last_instruction: null\n'
            )

            memory = OperationalMemory(root=Path(tmpdir))

            self.assertEqual(memory.knowledge["target_color"], "red")
            self.assertEqual(memory.knowledge["target_type"], "door")
            self.assertEqual(
                memory.knowledge["delivery_target"],
                {"color": "red", "object_type": "door"},
            )

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

    def test_phase_4_larger_gotodoor_human_render_uses_prewarmed_cache(self):
        ensure_custom_minigrid_envs_registered()
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            result = run_episode(
                instruction="go to the red door",
                compiler_name="llm",
                compiler=compiler,
                env_id="MiniGrid-GoToDoor-16x16-v0",
                seed=42,
                max_loops=512,
                render_mode="human",
                memory_root=Path(tempfile.mkdtemp()),
                use_cache=True,
                prewarm=True,
            )

        self.assertTrue(result["jit_prewarm"])
        self.assertEqual(result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(result["cache_miss_during_render"], 0)
        self.assertTrue(result["final_state"]["task_complete"])
        self.assertGreater(len(result["loop_records"]), 5)
        final_records = [
            record
            for record in result["loop_records"]
            if record["skill_plan"] is not None
        ]
        self.assertGreater(len(final_records), 0)
        self.assertEqual(final_records[-1]["skill_plan"], ["done"])
        self.assertEqual(final_records[-1]["report"]["status"], "succeeded")

    def test_operator_station_classifies_natural_language_utterances(self):
        cases = {
            "go to the red door": "task_instruction",
            "go to green door": "task_instruction",
            "can you go to the purple door": "unresolved",
            "go the the yellow door now": "unresolved",
            "go there again": "task_instruction",
            "go to the same door": "task_instruction",
            "repeat the last task": "task_instruction",
            "repeat the previous task": "task_instruction",
            "go to the closest door": "unresolved",
            "which door is closest to you": "unresolved",
            "ok. which of the doors is closest to you": "unresolved",
            "can you calculate the shortest distance to a door": "unresolved",
            "go to the door that is not yellow": "unresolved",
            "can you go to a door that is not yellow": "unresolved",
            "the green door is the delivery target": "knowledge_update",
            "your delivery target is the red door": "knowledge_update",
            "delivery target is red door": "knowledge_update",
            "set delivery target to the yellow door": "knowledge_update",
            "use the purple door as your delivery target": "knowledge_update",
            "target is the blue door": "knowledge_update",
            "remember the red door": "knowledge_update",
            "please remember that the grey door.": "knowledge_update",
            "what do you know?": "status_query",
            "what do you see": "status_query",
            "what doors are available": "status_query",
            "which doors are visible?": "status_query",
            "ok. Which door is closest to you?": "unresolved",
            "what can you do?": "status_query",
            "what happened last run?": "status_query",
            "what was the last target?": "status_query",
            "what was the previous target": "status_query",
            "show cache": "cache_query",
            "reset": "reset",
            "forget everything": "reset",
            "quit": "quit",
        }
        for utterance, expected_kind in cases.items():
            self.assertEqual(classify_utterance(utterance).kind, expected_kind)

    def test_capability_registry_reports_minigrid_manifest_statuses(self):
        registry = CapabilityRegistry.minigrid_default()
        summary = registry.compact_summary()

        self.assertEqual(summary["name"], "minigrid_primitive_registry_v1")
        self.assertIn("task", summary["primitives"])
        self.assertIn("grounding", summary["primitives"])
        self.assertIn("sensing", summary["primitives"])
        self.assertIn("action", summary["primitives"])
        task_names = {item["name"] for item in summary["primitives"]["task"]}
        sensing_names = {item["name"] for item in summary["primitives"]["sensing"]}
        action_names = {item["name"] for item in summary["primitives"]["action"]}
        self.assertTrue({f"task.{name}" for name in TASK_PRIMITIVES}.issubset(task_names))
        self.assertTrue(
            {f"sensing.{name}" for name in SENSING_PRIMITIVES}.issubset(sensing_names)
        )
        self.assertTrue({f"action.{name}" for name in ACTION_PRIMITIVES}.issubset(action_names))
        plan_grid_path = registry.primitive("action.plan_grid_path")
        self.assertIsNotNone(plan_grid_path)
        self.assertEqual(plan_grid_path.runtime_binding["kind"], "python")
        self.assertEqual(plan_grid_path.runtime_binding["value"], "plan_grid_path")
        self.assertEqual(
            registry.readiness_for_selector(
                {
                    "object_type": "door",
                    "color": None,
                    "exclude_colors": [],
                    "relation": "closest",
                    "distance_metric": "manhattan",
                    "distance_reference": "agent",
                }
            )["status"],
            "executable",
        )
        euclidean = registry.readiness_for_selector(
            {
                "object_type": "door",
                "color": None,
                "exclude_colors": [],
                "relation": "closest",
                "distance_metric": "euclidean",
                "distance_reference": "agent",
            }
        )
        self.assertEqual(euclidean["status"], "synthesizable_missing_primitive")
        self.assertEqual(euclidean["primitive"], "grounding.closest_door.euclidean.agent")
        pickup = registry.readiness_for_task(task_type="pickup", object_type="key")
        self.assertEqual(pickup["status"], "unsupported")
        self.assertEqual(pickup["layer"], "task")
        self.assertEqual(pickup["primitive"], "task.pickup.key")

    def test_operator_station_knowledge_update_and_queries_are_readable(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )
        response = session.handle_utterance("the green door is the delivery target")

        self.assertIn("KNOWLEDGE UPDATED", response)
        self.assertEqual(session.memory.knowledge["target_color"], "green")
        self.assertEqual(session.memory.knowledge["target_type"], "door")
        self.assertEqual(
            session.memory.knowledge["delivery_target"],
            {"color": "green", "object_type": "door"},
        )

    def test_operator_station_help_query_answers_from_capability_registry(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )

        response = session.handle_utterance("what can you do")

        self.assertIn("CAPABILITIES", response)
        self.assertIn("registry=minigrid_primitive_registry_v1", response)
        self.assertIn("task.go_to_object.door", response)
        self.assertIn("grounding.closest_door.manhattan.agent", response)
        self.assertIn("sensing.parse_grid_objects", response)
        self.assertIn("action.plan_grid_path", response)
        self.assertIn("grounding.closest_door.euclidean.agent", response)

    def test_operator_station_llm_capability_overview_answers_from_registry(self):
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )

        response = session.handle_utterance("give me an overview of your capabilities")

        self.assertIn("CAPABILITIES", response)
        self.assertIn("task.go_to_object.door", response)
        self.assertIn("sensing.build_occupancy_grid", response)
        self.assertIn("action.execute_next_path_action", response)
        status = session.handle_utterance("what do you know?")
        self.assertIn("delivery_target", status)
        self.assertNotIn("target_color=", status)
        self.assertNotIn("target_type=", status)
        self.assertIn("CACHE", session.handle_utterance("show cache"))

    def test_operator_station_accepts_natural_delivery_target_variants(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )
        response = session.handle_utterance("your delivery target is the red door")

        self.assertIn("KNOWLEDGE UPDATED", response)
        self.assertEqual(
            session.memory.knowledge["delivery_target"],
            {"color": "red", "object_type": "door"},
        )

        response = session.handle_utterance("set delivery target to blue door")

        self.assertIn("KNOWLEDGE UPDATED", response)
        self.assertEqual(
            session.memory.knowledge["delivery_target"],
            {"color": "blue", "object_type": "door"},
        )

    def test_operator_station_canonicalizes_task_instruction_variants(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )

        self.assertEqual(
            session.resolve_task_instruction("go to green door"),
            "go to the green door",
        )
        self.assertEqual(
            session.resolve_task_instruction("navigate to the grey door."),
            "go to the grey door",
        )
        self.assertEqual(
            session.resolve_task_instruction("can you go to the purple door"),
            "go to the purple door",
        )
        self.assertEqual(
            session.resolve_task_instruction("go the the yellow door now"),
            "go to the yellow door",
        )

    def test_operator_station_runs_natural_language_task_from_ready_prompt(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            self.assertEqual(session.startup(), "READY")
            self.assertIsNotNone(session.preview_adapter)
            response = session.handle_utterance("go to the red door")

        self.assertIn("RUN COMPLETE", response)
        self.assertIsNone(session.preview_adapter)
        self.assertIsNotNone(session.task_adapter)
        self.assertIsNotNone(session.last_result)
        self.assertTrue(session.last_result["final_state"]["task_complete"])
        self.assertEqual(session.last_result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(session.last_result["cache_miss_during_render"], 0)
        final_records = [
            record
            for record in session.last_result["loop_records"]
            if record["skill_plan"] is not None
        ]
        self.assertGreater(len(final_records), 0)
        self.assertEqual(final_records[-1]["skill_plan"], ["done"])
        self.assertEqual(final_records[-1]["report"]["status"], "succeeded")
        self.assertIn("RUN COMPLETE", session.handle_utterance("what happened last run?"))
        session.close()
        self.assertIsNone(session.task_adapter)

    def test_operator_station_answers_scene_and_help_fallbacks(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            scene = session.handle_utterance("what do you see")

        self.assertIn("SCENE", scene)
        self.assertIn("doors=", scene)
        help_response = session.handle_utterance("what can you do?")
        self.assertIn("CAPABILITIES", help_response)
        self.assertIn("task.go_to_object.door", help_response)
        self.assertIn("sensing.parse_grid_objects", help_response)
        self.assertIn("action.plan_grid_path", help_response)
        session.close()

    def test_operator_station_resolves_delivery_target_instruction_from_knowledge(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            session.handle_utterance("the red door is the delivery target")
            response = session.handle_utterance("go to the delivery target")

        self.assertIn("RUN COMPLETE", response)
        self.assertIsNotNone(session.last_result)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the red door")
        final_records = [
            record
            for record in session.last_result["loop_records"]
            if record["skill_plan"] is not None
        ]
        self.assertEqual(final_records[-1]["skill_plan"], ["done"])
        session.close()

    def test_operator_station_llm_intent_resolves_fuzzy_task_instruction(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            response = session.handle_utterance("can you please head over to the blue door")

        self.assertIn("RUN COMPLETE", response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the blue door")
        self.assertEqual(session.last_result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(session.last_result["cache_miss_during_render"], 0)
        self.assertTrue(
            any(call["method_name"] == "compile_operator_intent" for call in compiler.call_history)
        )
        session.close()

    def test_operator_station_sends_capability_manifest_to_operator_intent_compiler(self):
        seen_payloads = []

        def transport(request):
            if request["method_name"] == "compile_operator_intent":
                seen_payloads.append(request["user_payload"])
            return build_test_llm_transport()(request)

        compiler = LLMCompiler(api_key="test-key", transport=transport)
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )

        session.handle_utterance("which of the doors is closest to you")

        self.assertEqual(len(seen_payloads), 1)
        manifest = seen_payloads[0]["capability_manifest"]
        self.assertEqual(manifest["name"], "minigrid_primitive_registry_v1")
        grounding_names = {
            item["name"]
            for item in manifest["primitives"]["grounding"]
        }
        self.assertIn("grounding.closest_door.manhattan.agent", grounding_names)
        self.assertIn("grounding.closest_door.euclidean.agent", grounding_names)
        self.assertIn("sensing", manifest["primitives"])
        self.assertIn("action", manifest["primitives"])
        action_plan = next(
            item
            for item in manifest["primitives"]["action"]
            if item["name"] == "action.plan_grid_path"
        )
        self.assertEqual(action_plan["runtime_binding"]["kind"], "python")

    def test_operator_station_llm_intent_updates_delivery_target(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )

        response = session.handle_utterance("that red door is our delivery target")

        self.assertIn("KNOWLEDGE UPDATED", response)
        self.assertEqual(
            session.memory.knowledge["delivery_target"],
            {"color": "red", "object_type": "door"},
        )

    def test_operator_station_llm_intent_resolves_same_one_reference(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.handle_utterance("go to the red door")
            response = session.handle_utterance("go back to the same one")

        self.assertIn("RUN COMPLETE", response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the red door")
        session.close()

    def test_operator_station_llm_intent_resolves_last_task_status_query(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.handle_utterance("go to the red door")
            response = session.handle_utterance("what did I ask you to do last time?")

        self.assertIn("LAST RUN", response)
        self.assertIn("RUN COMPLETE", response)
        session.close()

    def test_operator_station_llm_intent_resolves_fuzzy_scene_query(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            response = session.handle_utterance("Ok. What do you see around you")

        self.assertIn("SCENE", response)
        self.assertIn("doors=", response)
        session.close()

    def test_operator_station_delivery_target_question_does_not_write_memory(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )
        session.handle_utterance("the yellow door is the delivery target")

        response = session.handle_utterance("and the delivery target?")

        self.assertIn("DELIVERY TARGET", response)
        self.assertIn("color=yellow", response)
        self.assertEqual(
            session.memory.knowledge["delivery_target"],
            {"color": "yellow", "object_type": "door"},
        )

    def test_operator_station_question_shaped_knowledge_intent_cannot_write_memory(self):
        def bad_transport(request):
            if request["method_name"] == "compile_operator_intent":
                return {
                    "intent_type": "knowledge_update",
                    "canonical_instruction": None,
                    "task_type": None,
                    "target": None,
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": {
                        "delivery_target": {"color": "red", "object_type": "door"}
                    },
                    "reference": None,
                    "status_query": None,
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.8,
                    "reason": "Incorrectly treated question as update.",
                }
            return build_test_llm_transport()(request)

        compiler = LLMCompiler(api_key="test-key", transport=bad_transport)
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )
        session.handle_utterance("the yellow door is the delivery target")

        response = session.handle_utterance("and the delivery target?")

        self.assertIn("DELIVERY TARGET", response)
        self.assertIn("color=yellow", response)
        self.assertEqual(
            session.memory.knowledge["delivery_target"],
            {"color": "yellow", "object_type": "door"},
        )

    def test_operator_station_unsupported_llm_intent_does_not_execute(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )

        response = session.handle_utterance("pick up the red key")

        self.assertIn("I cannot safely execute that capability yet", response)
        self.assertIsNone(session.last_result)
        self.assertFalse(
            any(call["method_name"] == "compile_task" for call in compiler.call_history)
        )

    def test_operator_station_answers_visible_door_query(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="human",
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            response = session.handle_utterance("which doors are visible?")

        self.assertIn("SCENE", response)
        self.assertIn("doors=", response)
        self.assertIn("yellow door", response)
        session.close()

    def test_operator_station_ground_target_query_reports_closest_manhattan_door(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="human",
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            response = session.handle_utterance("which door is closest by Manhattan distance?")

        self.assertIn("GROUNDED TARGET", response)
        self.assertIn("target=", response)
        self.assertIn("distance=", response)
        self.assertIsNone(session.last_result)
        session.close()

    def test_operator_station_runs_grounded_closest_manhattan_door_task(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="human",
            max_loops=512,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            response = session.handle_utterance(
                "go to the closest door using Manhattan distance"
            )

        self.assertIn("RUN COMPLETE", response)
        self.assertTrue(session.last_result["final_state"]["task_complete"])
        self.assertEqual(session.last_result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(session.last_result["cache_miss_during_render"], 0)
        final_records = [
            record
            for record in session.last_result["loop_records"]
            if record["skill_plan"] is not None
        ]
        self.assertEqual(final_records[-1]["skill_plan"], ["done"])
        self.assertEqual(final_records[-1]["report"]["status"], "succeeded")
        session.close()

    def test_operator_station_closest_without_metric_requests_clarification(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        response = session.handle_utterance("go to the closest door")

        self.assertIn("CLARIFY", response)
        self.assertIn("which distance metric", response)
        self.assertIn("Supported: manhattan", response)
        self.assertIsNotNone(session.pending_clarification)
        self.assertEqual(session.pending_clarification.missing_field, "distance_metric")
        self.assertIsNone(session.last_result)

    def test_operator_station_manhattan_answer_resumes_closest_task(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="human",
            max_loops=512,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            clarify = session.handle_utterance("go to the closest door")
            response = session.handle_utterance("manhattan")

        self.assertIn("CLARIFY", clarify)
        self.assertIsNone(session.pending_clarification)
        self.assertIn("RUN COMPLETE", response)
        self.assertTrue(session.last_result["final_state"]["task_complete"])
        self.assertEqual(session.last_result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(session.last_result["cache_miss_during_render"], 0)
        final_records = [
            record
            for record in session.last_result["loop_records"]
            if record["skill_plan"] is not None
        ]
        self.assertEqual(final_records[-1]["skill_plan"], ["done"])
        session.close()

    def test_operator_station_euclidean_answer_fails_safely_and_clears_pending(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        session.handle_utterance("go to the closest door")
        response = session.handle_utterance("euclidean")

        self.assertEqual(response, "I cannot use Euclidean distance yet. Supported: manhattan.")
        self.assertIsNone(session.pending_clarification)
        self.assertIsNone(session.last_result)

    def test_operator_station_cancel_clears_pending_clarification(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        session.handle_utterance("go to the closest door")
        response = session.handle_utterance("cancel")

        self.assertIn("pending clarification cleared", response)
        self.assertIsNone(session.pending_clarification)
        self.assertIsNone(session.last_result)

    def test_operator_station_reset_clears_pending_clarification(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        session.handle_utterance("go to the closest door")
        response = session.handle_utterance("reset")

        self.assertIn("RESET", response)
        self.assertIsNone(session.pending_clarification)
        self.assertIsNone(session.last_result)

    def test_operator_station_status_and_cache_keep_pending_clarification(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        session.handle_utterance("go to the closest door")
        status = session.handle_utterance("what do you know")
        cache = session.handle_utterance("show cache")

        self.assertIn("STATUS", status)
        self.assertIn("CACHE", cache)
        self.assertIsNotNone(session.pending_clarification)
        self.assertIsNone(session.last_result)

    def test_operator_station_new_task_cancels_pending_clarification_and_runs(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            session.handle_utterance("go to the closest door")
            response = session.handle_utterance("go to the red door")

        self.assertIsNone(session.pending_clarification)
        self.assertIn("RUN COMPLETE", response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the red door")
        session.close()

    def test_operator_station_ambiguous_unique_selector_requests_candidate_clarification(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        response = session.handle_utterance("go to the door that is not yellow")

        self.assertIn("CLARIFY", response)
        self.assertIn("matched multiple doors", response)
        self.assertIn("Options:", response)
        self.assertIsNotNone(session.pending_clarification)
        self.assertEqual(
            session.pending_clarification.clarification_type,
            "target_selector_candidate_choice",
        )
        self.assertIsNone(session.last_result)

    def test_operator_station_candidate_answer_resumes_ambiguous_selector_task(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="human",
            max_loops=512,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            clarify = session.handle_utterance("go to the door that is not yellow")
            response = session.handle_utterance("red")

        self.assertIn("CLARIFY", clarify)
        self.assertIsNone(session.pending_clarification)
        self.assertIn("RUN COMPLETE", response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the red door")
        self.assertTrue(session.last_result["final_state"]["task_complete"])
        self.assertEqual(session.last_result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(session.last_result["cache_miss_during_render"], 0)
        final_records = [
            record
            for record in session.last_result["loop_records"]
            if record["skill_plan"] is not None
        ]
        self.assertEqual(final_records[-1]["skill_plan"], ["done"])
        session.close()

    def test_operator_station_natural_not_yellow_phrase_requests_candidate_clarification(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        response = session.handle_utterance("can you go to a door that is not yellow")

        self.assertIn("CLARIFY", response)
        self.assertIn("matched multiple doors", response)
        self.assertIsNotNone(session.pending_clarification)

    def test_operator_station_natural_closest_query_requests_metric_clarification(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        response = session.handle_utterance("which door is closest to you")

        self.assertIn("CLARIFY", response)
        self.assertIn("which distance metric", response)
        self.assertIsNotNone(session.pending_clarification)

    def test_operator_station_conversational_closest_query_requests_metric_clarification(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        response = session.handle_utterance("ok. Which door is closest to you?")

        self.assertIn("CLARIFY", response)
        self.assertIn("which distance metric", response)
        self.assertIsNotNone(session.pending_clarification)

    def test_operator_station_which_of_the_doors_closest_requests_metric_clarification(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        response = session.handle_utterance("ok. which of the doors is closest to you")

        self.assertIn("CLARIFY", response)
        self.assertIn("which distance metric", response)
        self.assertIsNotNone(session.pending_clarification)

    def test_operator_station_shortest_distance_query_requests_metric_clarification(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        response = session.handle_utterance("can you calculate the shortest distance to a door")

        self.assertIn("CLARIFY", response)
        self.assertIn("which distance metric", response)
        self.assertIsNotNone(session.pending_clarification)

    def test_operator_station_euclidean_closest_reports_synthesizable_missing_primitive(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        response = session.handle_utterance("which door is closest by Euclidean distance")

        self.assertIn("required primitive is not implemented yet", response)
        self.assertIn("grounding.closest_door.euclidean.agent", response)
        self.assertIn("Phase 7.7", response)
        self.assertIsNone(session.last_result)

    def test_operator_station_grounded_selector_can_update_delivery_target(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="human",
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.startup()
            response = session.handle_utterance(
                "make the closest door by Manhattan distance the delivery target"
            )

        self.assertIn("KNOWLEDGE UPDATED", response)
        self.assertIsNotNone(session.memory.knowledge["delivery_target"])
        self.assertEqual(session.memory.knowledge["delivery_target"]["object_type"], "door")
        session.close()

    def test_operator_station_rejects_llm_chosen_target_for_closest_request(self):
        def bad_transport(request):
            if request["method_name"] == "compile_operator_intent":
                return {
                    "intent_type": "task_instruction",
                    "canonical_instruction": "go to the yellow door",
                    "task_type": "go_to_object",
                    "target": {"color": "yellow", "object_type": "door"},
                    "target_selector": None,
                    "capability_status": "executable",
                    "knowledge_update": None,
                    "reference": None,
                    "status_query": None,
                    "control": None,
                    "clear_memory": False,
                    "confidence": 0.99,
                    "reason": "Incorrectly chose a closest target directly.",
                }
            return build_test_llm_transport()(request)

        compiler = LLMCompiler(api_key="test-key", transport=bad_transport)
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=8,
            render_mode="none",
            memory_root=Path(tempfile.mkdtemp()),
        )

        response = session.handle_utterance("go to the closest door using Manhattan distance")

        self.assertIn("valid target selector", response)
        self.assertIsNone(session.last_result)

    def test_operator_station_missing_reference_fails_safely(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )

        self.assertIn(
            "I do not have a previous target yet",
            session.handle_utterance("go there again"),
        )
        self.assertIn(
            "I do not have a previous successful task yet",
            session.handle_utterance("repeat the last task"),
        )
        self.assertEqual(session.handle_utterance("what was the last target?"), "LAST TARGET: none")

    def test_operator_station_stores_last_target_after_success(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            response = session.handle_utterance("go to the red door")

        self.assertIn("RUN COMPLETE", response)
        self.assertEqual(
            session.memory.episodic_memory["last_target"],
            {"color": "red", "object_type": "door"},
        )
        self.assertEqual(
            session.memory.episodic_memory["last_successful_instruction"],
            "go to the red door",
        )
        last_target = session.handle_utterance("what was the last target?")
        self.assertIn("LAST TARGET", last_target)
        self.assertIn("color=red", last_target)
        self.assertIn("object_type=door", last_target)
        self.assertIn("instruction=go to the red door", last_target)
        session.close()

    def test_operator_station_go_there_again_resolves_after_success(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.handle_utterance("go to the red door")
            response = session.handle_utterance("go there again")

        self.assertIn("RUN COMPLETE", response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the red door")
        self.assertTrue(session.last_result["final_state"]["task_complete"])
        self.assertEqual(session.last_result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(session.last_result["cache_miss_during_render"], 0)
        session.close()

    def test_operator_station_repeat_last_task_resolves_after_success(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.handle_utterance("go to the red door")
            response = session.handle_utterance("repeat the last task")

        self.assertIn("RUN COMPLETE", response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the red door")
        self.assertTrue(session.last_result["final_state"]["task_complete"])
        self.assertEqual(session.last_result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(session.last_result["cache_miss_during_render"], 0)
        session.close()

    def test_operator_station_failed_run_does_not_overwrite_last_target(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="none",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        session.handle_utterance("go to the red door")
        self.assertEqual(
            session.memory.episodic_memory["last_target"],
            {"color": "red", "object_type": "door"},
        )

        session.env_id = "MiniGrid-GoToDoor-16x16-v0"
        session.seed = 12
        response = session.handle_utterance("go to the red door")

        self.assertIn("RUN FAILED", response)
        self.assertEqual(
            session.memory.episodic_memory["last_target"],
            {"color": "red", "object_type": "door"},
        )

    def test_operator_station_reset_clears_reference_context_but_keeps_delivery_target(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.handle_utterance("the red door is the delivery target")
            session.handle_utterance("go to the red door")
            self.assertIsNotNone(session.last_result)
            reset = session.handle_utterance("reset")
            missing = session.handle_utterance("go there again")
            response = session.handle_utterance("go to the delivery target")

        self.assertIn("durable knowledge kept", reset)
        self.assertIsNotNone(session.last_result)
        self.assertIn("I do not have a previous target yet", missing)
        self.assertIn("RUN COMPLETE", response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the red door")
        session.close()

    def test_operator_station_reset_clears_last_result(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            session.handle_utterance("go to the red door")
            self.assertIsNotNone(session.last_result)
            response = session.handle_utterance("reset")

        self.assertIn("episodic state cleared", response)
        self.assertIsNone(session.last_result)
        self.assertIsNone(session.memory.episodic_memory["last_target"])
        session.close()

    def test_operator_station_delivery_target_persists_across_station_restart(self):
        memory_root = Path(tempfile.mkdtemp())
        first_session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=memory_root,
            render_mode="none",
        )
        first_session.handle_utterance("the red door is the delivery target")

        second_session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=memory_root,
            render_mode="none",
        )

        self.assertEqual(
            second_session.memory.knowledge["delivery_target"],
            {"color": "red", "object_type": "door"},
        )
        self.assertEqual(
            second_session.resolve_task_instruction("go to the delivery target"),
            "go to the red door",
        )

    def test_operator_station_clear_memory_clears_delivery_target(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )
        session.handle_utterance("the red door is the delivery target")
        response = session.handle_utterance("forget everything")

        self.assertIn("durable knowledge cleared", response)
        self.assertIsNone(session.memory.knowledge["delivery_target"])
        self.assertIsNone(session.memory.knowledge["target_color"])
        self.assertIsNone(session.memory.knowledge["target_type"])
        self.assertIn(
            "I do not have a delivery target yet",
            session.handle_utterance("go to the delivery target"),
        )

    def test_operator_station_reports_missing_delivery_target_safely(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )
        response = session.handle_utterance("go to the delivery target")

        self.assertIn("I do not have a delivery target yet", response)

    def test_operator_station_startup_opens_idle_preview_before_task(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode="human",
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            self.assertEqual(session.startup(), "READY")
            self.assertIsNotNone(session.preview_adapter)

        session.close()
        self.assertIsNone(session.preview_adapter)

    def test_operator_station_reports_target_absent_without_followup_question(self):
        compiler = LLMCompiler(api_key="test-key", transport=build_test_llm_transport())
        session = OperatorStationSession(
            compiler_name="llm",
            compiler=compiler,
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=12,
            render_mode="human",
            max_loops=64,
            memory_root=Path(tempfile.mkdtemp()),
        )
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            response = session.handle_utterance("go to the red door")

        self.assertIn("RUN FAILED", response)
        self.assertIn("reason=target_absent", response)
        self.assertIn("available_targets=", response)

    def test_operator_station_reset_keeps_durable_knowledge_by_default(self):
        session = OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            memory_root=Path(tempfile.mkdtemp()),
            render_mode="none",
        )
        session.handle_utterance("remember the red door")
        response = session.handle_utterance("reset")

        self.assertIn("durable knowledge kept", response)
        self.assertEqual(session.memory.knowledge["target_color"], "red")

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


class TestSceneModel(unittest.TestCase):
    """Phase 7.57 — Persistent SceneModel projection and grounding."""

    def _make_session(self, render_mode="none"):
        return OperatorStationSession(
            compiler=LLMCompiler(api_key="test-key", transport=build_test_llm_transport()),
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode=render_mode,
            memory_root=Path(tempfile.mkdtemp()),
        )

    def _run_with_env(self, fn):
        with patch(
            "jeenom.run_demo.build_env",
            side_effect=lambda env_id, render_mode: FullyObsWrapper(gym.make(env_id)),
        ):
            return fn()

    def test_sense_tick_populates_scene_model(self):
        """After a task's sense tick, memory.scene_model must be set."""
        session = self._make_session()
        self.assertIsNone(session.memory.scene_model)
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self.assertIsNotNone(session.memory.scene_model)
        scene = session.memory.scene_model
        self.assertIsInstance(scene, SceneModel)
        self.assertEqual(scene.source, "task_sense")

    def test_scene_model_contains_agent_pose(self):
        """SceneModel must record the agent position after sensing."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        scene = session.memory.scene_model
        self.assertIsNotNone(scene)
        self.assertIsInstance(scene.agent_x, int)
        self.assertIsInstance(scene.agent_y, int)
        self.assertIsInstance(scene.agent_dir, int)

    def test_scene_model_contains_door_objects(self):
        """SceneModel objects must include at least one door."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        scene = session.memory.scene_model
        doors = scene.find(object_type="door")
        self.assertGreater(len(doors), 0)
        for door in doors:
            self.assertEqual(door.object_type, "door")
            self.assertIsInstance(door.x, int)
            self.assertIsInstance(door.y, int)

    def test_idle_sense_builds_scene_model_before_any_task(self):
        """_ensure_scene_model() must build a scene model via idle sense if none exists."""
        session = self._make_session()
        self.assertIsNone(session.memory.scene_model)
        self._run_with_env(lambda: session._ensure_scene_model())
        self.assertIsNotNone(session.memory.scene_model)
        self.assertEqual(session.memory.scene_model.source, "idle_sense")

    def test_scene_summary_uses_scene_model_not_env_reset(self):
        """scene_summary must answer from SceneModel and include agent position."""
        session = self._make_session(render_mode="human")
        response = self._run_with_env(lambda: session.handle_utterance("what do you see"))
        self.assertIn("SCENE", response)
        self.assertIn("agent=", response)
        self.assertIn("source=", response)
        self.assertIn("doors=", response)

    def test_grounding_uses_scene_model_for_closest_door(self):
        """Grounding closest-door query must use SceneModel, not a fresh env reset."""
        session = self._make_session(render_mode="human")

        reset_calls = []
        original_reset = MiniGridAdapter.reset

        def tracking_reset(self_adapter, seed=None):
            reset_calls.append(seed)
            return original_reset(self_adapter, seed=seed)

        with patch.object(MiniGridAdapter, "reset", tracking_reset):
            self._run_with_env(lambda: session.startup())
            reset_calls.clear()  # ignore startup reset
            result = session.handle_utterance(
                "which door is closest by manhattan distance"
            )

        # No adapter.reset() should fire during the grounding query
        self.assertEqual(reset_calls, [], msg="grounding called adapter.reset()")
        self.assertIn("GROUNDED TARGET", result)
        self.assertIn("distance=", result)

    def test_unique_door_grounding_returns_distance(self):
        """Grounding a unique color-specific door must now return distance."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        grounded = session.ground_target_selector(
            {"object_type": "door", "color": "red", "exclude_colors": [], "relation": "unique",
             "distance_metric": None, "distance_reference": None}
        )
        self.assertTrue(grounded["ok"])
        self.assertIsNotNone(grounded.get("distance"))
        self.assertIsInstance(grounded["distance"], int)

    def test_scene_model_cleared_on_reset(self):
        """reset() must clear scene_model so the next query builds a fresh one."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self.assertIsNotNone(session.memory.scene_model)
        session.reset()
        self.assertIsNone(session.memory.scene_model)

    def test_scene_model_source_is_task_sense_after_task(self):
        """After a completed task, scene_model.source must be 'task_sense'."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self.assertEqual(session.memory.scene_model.source, "task_sense")

    def test_scene_model_source_is_idle_sense_before_task(self):
        """Before any task, idle sense pass must produce source='idle_sense'."""
        session = self._make_session(render_mode="human")
        self._run_with_env(lambda: session.startup())
        # Clear any scene_model startup might have created (preview sense)
        session.memory.scene_model = None
        self._run_with_env(lambda: session._ensure_scene_model())
        self.assertEqual(session.memory.scene_model.source, "idle_sense")


class TestStationActiveClaims(unittest.TestCase):
    """Phase 7.58 — Station Active Claims: typed, session-scoped claims from grounding."""

    def _make_session(self, render_mode: str = "none") -> OperatorStationSession:
        return OperatorStationSession(
            compiler=SmokeTestCompiler(),
            compiler_name="smoke",
            env_id="MiniGrid-GoToDoor-8x8-v0",
            seed=42,
            render_mode=render_mode,
            memory_root=Path(tempfile.mkdtemp()),
        )

    def _run_with_env(self, fn):
        def fake_build_env(env_id, render_mode):
            return FullyObsWrapper(gym.make(env_id))
        with patch("jeenom.run_demo.build_env", side_effect=fake_build_env):
            return fn()

    def test_claims_written_after_closest_grounding(self):
        """After grounding closest door, active_claims must be set."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        self.assertIsNotNone(session.active_claims)

    def test_claims_have_ranked_doors(self):
        """Active claims must contain a non-empty ranked_scene_doors list."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        self.assertIsNotNone(session.active_claims)
        self.assertGreater(len(session.active_claims.ranked_scene_doors), 0)

    def test_claims_last_grounded_target_is_set(self):
        """After closest grounding, last_grounded_target must be a GroundedDoorEntry."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        self.assertIsNotNone(session.active_claims)
        self.assertIsInstance(session.active_claims.last_grounded_target, GroundedDoorEntry)

    def test_claims_last_grounded_rank_is_zero(self):
        """The first grounded target must have rank 0."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        self.assertIsNotNone(session.active_claims)
        self.assertEqual(session.active_claims.last_grounded_rank, 0)

    def test_next_closest_resolves_from_claims(self):
        """claim_reference=next_closest must resolve to rank-1 door."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        result = self._run_with_env(lambda: session.handle_utterance("next closest door"))
        self.assertIsNotNone(result)
        self.assertIn("CLAIM", result.upper())

    def test_claims_cleared_on_reset(self):
        """reset() must clear active_claims."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        self.assertIsNotNone(session.active_claims)
        session.reset()
        self.assertIsNone(session.active_claims)

    def test_claims_cleared_on_new_task(self):
        """Starting a new task must clear active_claims at the start of run_task."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        # active_claims may have been written again by the sense tick in run_task,
        # but prior claims from the grounding query are not carried over.
        # The important invariant: active_claims was cleared at task start.
        # We verify indirectly: last_grounding_query should not be from the grounding round.
        if session.active_claims is not None:
            # claims were refreshed by the task's sense tick — last_grounding_query was reset
            self.assertIsNone(session.active_claims.last_grounding_query.get("relation"))

    def test_stale_claims_fail_safely(self):
        """If scene has changed, claim resolution must return an error dict, not raise."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        self.assertIsNotNone(session.active_claims)
        # Manually corrupt the fingerprint to simulate a stale claims scenario
        import dataclasses
        stale = dataclasses.replace(
            session.active_claims,
            scene_fingerprint=(-1, -1, -1),
        )
        session.active_claims = stale
        result = session._resolve_claim_reference("next_closest")
        self.assertFalse(result.get("ok", True))

    def test_claims_compact_summary_is_dict(self):
        """compact_summary() must return a dict with known keys."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        self.assertIsNotNone(session.active_claims)
        summary = session.active_claims.compact_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("last_grounded_target", summary)
        self.assertIn("ranked_doors", summary)

    def test_is_valid_for_checks_fingerprint(self):
        """StationActiveClaims.is_valid_for() must return False for a different fingerprint."""
        session = self._make_session()
        self._run_with_env(lambda: session.handle_utterance("go to the red door"))
        self._run_with_env(lambda: session.handle_utterance(
            "which door is closest by manhattan distance"
        ))
        self.assertIsNotNone(session.active_claims)
        scene = session.memory.scene_model
        self.assertTrue(session.active_claims.is_valid_for(scene))
        # Build a fake scene with a different fingerprint
        import dataclasses
        fake_scene = dataclasses.replace(scene, agent_x=scene.agent_x + 5, step_count=9999)
        self.assertFalse(session.active_claims.is_valid_for(fake_scene))


class TestGroundingResultComposition(unittest.TestCase):
    """Phase 7.7 — compose answers/tasks from registered ranked-door grounding claims."""

    def _make_session(
        self,
        render_mode: str = "none",
        *,
        seed: int = 8,
    ) -> OperatorStationSession:
        return OperatorStationSession(
            compiler=SmokeTestCompiler(),
            compiler_name="smoke",
            env_id="MiniGrid-GoToDoor-16x16-v0",
            seed=seed,
            render_mode=render_mode,
            memory_root=Path(tempfile.mkdtemp()),
            max_loops=512,
        )

    def _run_with_env(self, fn):
        def fake_build_env(env_id, render_mode):
            return FullyObsWrapper(gym.make(env_id))

        with patch("jeenom.run_demo.build_env", side_effect=fake_build_env):
            return fn()

    def _assert_cached_success(self, session: OperatorStationSession, response: str) -> None:
        self.assertIn("RUN COMPLETE", response)
        self.assertIn("final_skill_plan=['done']", response)
        self.assertIsNotNone(session.last_result)
        self.assertTrue(session.last_result["final_state"]["task_complete"])
        self.assertEqual(session.last_result["runtime_llm_calls_during_render"], 0)
        self.assertEqual(session.last_result["cache_miss_during_render"], 0)

    def test_closest_and_farthest_query_composes_ranked_claim_answer(self):
        session = self._make_session()
        response = self._run_with_env(
            lambda: session.handle_utterance("which door is closest and which is farthest")
        )

        self.assertIn("GROUNDING ANSWER", response)
        self.assertIn("closest=purple door@(4,0) distance=5", response)
        self.assertIn("farthest=", response)
        self.assertIn("blue door@(12,3) distance=8", response)
        self.assertIn("red door@(10,7) distance=8", response)
        self.assertIn("tie=", response)
        self.assertIsNone(session.last_result)
        self.assertIsNotNone(session.active_claims)

    def test_go_to_farthest_door_clarifies_when_farthest_is_tied(self):
        session = self._make_session()
        response = self._run_with_env(lambda: session.handle_utterance("go to the farthest door"))

        self.assertIn("CLARIFY", response)
        self.assertIn("multiple farthest", response)
        self.assertIn("blue door@(12,3)", response)
        self.assertIn("red door@(10,7)", response)
        self.assertIsNone(session.last_result)
        self.assertIsNotNone(session.pending_clarification)

        resumed = self._run_with_env(lambda: session.handle_utterance("red"))
        self._assert_cached_success(session, resumed)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the red door")

    def test_go_to_second_closest_door_composes_ranked_target_and_executes(self):
        session = self._make_session()
        response = self._run_with_env(
            lambda: session.handle_utterance("go to the second closest door")
        )

        self._assert_cached_success(session, response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the yellow door")

    def test_second_farthest_door_does_not_degrade_to_farthest(self):
        session = self._make_session(seed=12)
        response = self._run_with_env(
            lambda: session.handle_utterance("can you navigate to the second farthest door")
        )

        self.assertIn("CLARIFY", response)
        self.assertIn("ordinal falls inside a distance tie", response)
        self.assertIn("green door@(9,0)", response)
        self.assertIn("yellow door@(11,2)", response)
        self.assertIsNone(session.last_result)

    def test_distance_reference_uses_ranked_claims_and_executes_unique_match(self):
        session = self._make_session()
        response = self._run_with_env(
            lambda: session.handle_utterance("can you go to the door with a distance of 7")
        )

        self._assert_cached_success(session, response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the yellow door")

    def test_color_reference_after_ranked_display_uses_active_claims(self):
        session = self._make_session()
        ranked = self._run_with_env(
            lambda: session.handle_utterance("rank all the doors by manhattan distance")
        )
        self.assertIn("DOORS RANKED BY MANHATTAN DISTANCE FROM AGENT", ranked)
        self.assertIsNotNone(session.active_claims)

        response = self._run_with_env(lambda: session.handle_utterance("go to the red one"))
        self._assert_cached_success(session, response)
        self.assertEqual(session.last_result["task"]["instruction"], "go to the red door")

    def test_ranked_composition_uses_registered_primitive_handle(self):
        session = self._make_session()
        self._run_with_env(
            lambda: session.handle_utterance("which door is closest and which is farthest")
        )

        self.assertIsNotNone(session.active_claims)
        self.assertEqual(
            session.active_claims.last_grounding_query.get("primitive"),
            "grounding.all_doors.ranked.manhattan.agent",
        )


if __name__ == "__main__":
    unittest.main()
