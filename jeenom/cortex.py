from __future__ import annotations

from dataclasses import asdict

from .primitive_library import ACTION_PRIMITIVES, SENSING_PRIMITIVES, TASK_PRIMITIVES, produced_evidence
from .schemas import (
    EvidenceFrame,
    ExecutionContract,
    ReadinessReport,
    TraceEvent,
)


class Readiness:
    def __init__(self, memory):
        self.memory = memory

    def check(self, procedure, substrate: str = "minigrid"):
        available_evidence = produced_evidence(SENSING_PRIMITIVES) | {
            "occupancy_grid",
            "object_location",
            "agent_pose",
            "adjacency_to_target",
        }
        available_actions = set(ACTION_PRIMITIVES)

        missing_task_primitives = []
        missing_evidence = set()
        missing_actions = set()

        for step_name in procedure.steps:
            spec = TASK_PRIMITIVES.get(step_name)
            if spec is None:
                missing_task_primitives.append(step_name)
                continue

            for evidence in spec.consumes:
                if evidence not in available_evidence:
                    missing_evidence.add(evidence)

            for action in spec.required_action_primitives:
                if action not in available_actions:
                    missing_actions.add(action)

        status = "executable"
        if missing_task_primitives:
            status = "impossible"
        elif missing_evidence or missing_actions:
            status = "partial"

        return ReadinessReport(
            status=status,
            task_type=procedure.task_type,
            missing_task_primitives=sorted(missing_task_primitives),
            missing_evidence=sorted(missing_evidence),
            missing_actions=sorted(missing_actions),
            recipe_steps=list(procedure.steps),
        )


class Cortex:
    def __init__(self, memory, compiler, plan_cache=None):
        self.memory = memory
        self.compiler = compiler
        self.plan_cache = plan_cache
        self.readiness = Readiness(memory)
        self.claims = {}
        self.trace: list[TraceEvent] = []
        self.task_request = None
        self.procedure = None
        self.resolved_task_params = {}
        self.last_world_sample = None
        self.execution_state = {
            "step_index": 0,
            "task_complete": False,
            "current_skill": None,
            "last_report": None,
            "knowledge_override_active": False,
        }

    def onboard_task(self, task_request, procedure):
        readiness = self.readiness.check(procedure, substrate="minigrid")
        if readiness.status == "impossible":
            raise RuntimeError(f"Task is not executable: {asdict(readiness)}")

        self.task_request = task_request
        self.procedure = procedure
        self.resolved_task_params = self.memory.resolve_target_params(task_request.params)
        self.memory.reset_episode(clear_reference_context=False)

        color_override = (
            task_request.params.get("color") is not None
            and self.resolved_task_params.get("color") != task_request.params.get("color")
        )
        type_override = (
            task_request.params.get("object_type") is not None
            and self.resolved_task_params.get("object_type") != task_request.params.get("object_type")
        )
        self.execution_state["knowledge_override_active"] = color_override or type_override

        self.record_trace(
            "task_onboarded",
            {
                "task_request": asdict(task_request),
                "resolved_task_params": dict(self.resolved_task_params),
                "procedure": list(procedure.steps),
                "readiness": asdict(readiness),
            },
        )
        return readiness

    def make_evidence_frame(self):
        self._advance_completed_steps()
        active_step = self._current_step_name()

        if active_step is None:
            return EvidenceFrame(
                needs=[],
                context=dict(self.resolved_task_params),
                step_index=self.execution_state["step_index"],
            )
        if active_step == "locate_object":
            needs = ["object_location", "agent_pose", "occupancy_grid"]
        elif active_step == "navigate_to_object":
            needs = ["object_location", "agent_pose", "occupancy_grid", "adjacency_to_target"]
        elif active_step == "verify_adjacent":
            needs = ["agent_pose", "object_location", "adjacency_to_target"]
        else:
            needs = ["adjacency_to_target"]

        frame = EvidenceFrame(
            needs=needs,
            context=dict(self.resolved_task_params),
            active_step=active_step,
            step_index=self.execution_state["step_index"],
        )
        self.record_trace(
            "evidence_frame_created",
            {"evidence_needs": list(needs), "context": dict(frame.context)},
            step_name=active_step,
        )
        return frame

    def update_from_evidence(self, evidence, world_sample=None):
        self.claims.update(evidence.claims)
        if world_sample is not None:
            self.last_world_sample = world_sample
        self.record_trace(
            "evidence_update",
            {"claims": evidence.claims},
            step_name=self._current_step_name(),
        )
        self._advance_completed_steps()

    def choose_execution_contract(self):
        self._advance_completed_steps()
        active_step = self._current_step_name()
        if active_step is None:
            self.execution_state["task_complete"] = True
            return None

        if active_step == "done" and self.execution_state.get("knowledge_override_active"):
            self.execution_state["task_complete"] = True
            self.record_trace(
                "done_skipped_for_knowledge_override",
                {
                    "resolved_task_params": dict(self.resolved_task_params),
                    "reason": "MiniGrid mission reward disagrees with durable knowledge target.",
                },
                step_name=active_step,
            )
            return None

        params = dict(self.resolved_task_params)
        params["target_location"] = self.claims.get("target_location")

        if active_step == "locate_object":
            skill = "turn_right"
        elif active_step in {"navigate_to_object", "verify_adjacent"}:
            skill = "navigate_to_object" if self.claims.get("target_location") else "turn_right"
        elif active_step == "done":
            skill = "done"
        else:
            skill = "abort"

        contract = ExecutionContract(
            skill=skill,
            params=params,
            stop_conditions=["adjacent_to_target", "task_complete"],
            source="cortex",
        )
        self.execution_state["current_skill"] = skill
        self.record_trace(
            "execution_contract_issued",
            asdict(contract),
            step_name=active_step,
        )
        return contract

    def update_from_report(self, report):
        self.execution_state["last_report"] = asdict(report)
        if self.execution_state["current_skill"] == "done" and report.status == "succeeded":
            self.execution_state["task_complete"] = True

        self.record_trace(
            "execution_report",
            asdict(report),
            step_name=self._current_step_name(),
        )

    def finalize(self):
        final_state = {
            "task_request": asdict(self.task_request) if self.task_request else None,
            "resolved_task_params": dict(self.resolved_task_params),
            "execution_state": dict(self.execution_state),
            "last_world_sample": self.last_world_sample.summary() if self.last_world_sample else None,
        }
        trace_payload = [asdict(event) for event in self.trace]
        updates = self.compiler.compile_memory_updates(
            final_state=final_state,
            final_claims=dict(self.claims),
            trace=trace_payload,
            memory=self.memory,
        )
        self.memory.apply_memory_updates(updates)
        self.record_trace(
            "memory_updates_applied",
            {"updates": [asdict(update) for update in updates]},
        )
        return updates

    def record_trace(self, event: str, payload: dict, step_name: str | None = None) -> None:
        self.trace.append(
            TraceEvent(
                event=event,
                payload=payload,
                loop_index=len(self.trace),
                step_name=step_name,
            )
        )

    def _advance_completed_steps(self):
        while True:
            active_step = self._current_step_name()
            if active_step is None:
                self.execution_state["task_complete"] = True
                return
            if not self._step_complete(active_step):
                return
            self.execution_state["step_index"] += 1

    def _current_step_name(self):
        if self.procedure is None:
            return None
        idx = self.execution_state["step_index"]
        if idx >= len(self.procedure.steps):
            return None
        return self.procedure.steps[idx]

    def _step_complete(self, step_name):
        if step_name == "locate_object":
            return bool(self.claims.get("target_location"))
        if step_name in {"navigate_to_object", "verify_adjacent"}:
            return bool(self.claims.get("adjacency_to_target"))
        if step_name == "done":
            return bool(self.execution_state.get("task_complete"))
        return False
