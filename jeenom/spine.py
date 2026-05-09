from __future__ import annotations

from collections import deque

from .plan_cache import skill_key
from .primitive_library import ACTION_PRIMITIVES
from .schemas import ExecutionContext, ExecutionReport, PrimitiveCall, SkillPlanTemplate


DIR_TO_VEC = {
    0: (1, 0),
    1: (0, 1),
    2: (-1, 0),
    3: (0, -1),
}
VEC_TO_DIR = {value: key for key, value in DIR_TO_VEC.items()}
DIRECT_ACTION_SKILLS = {
    "done",
    "turn_left",
    "turn_right",
    "move_forward",
    "pickup",
    "drop",
    "toggle",
}


class MiniGridSpine:
    def __init__(self, memory, adapter, compiler, plan_cache=None):
        self.memory = memory
        self.adapter = adapter
        self.compiler = compiler
        self.plan_cache = plan_cache
        self.active_skill = None

    def tick(
        self,
        execution_contract,
        percepts,
        loop_index: int = 0,
        allow_llm_compile: bool = True,
    ):
        self.active_skill = execution_contract.skill
        template, cache_meta = self._resolve_template(
            execution_contract=execution_contract,
            percepts=percepts,
            loop_index=loop_index,
            allow_llm_compile=allow_llm_compile,
        )
        plan = self.instantiate_template(template, execution_contract)
        report = self.execute_plan(execution_contract, plan, percepts)

        if self.plan_cache is not None and self.plan_cache.enabled:
            cache_key = cache_meta["cache_key"]
            if report.status == "failed":
                invalidated = self.plan_cache.record_failure(
                    cache_key,
                    immediate=report.reason in {
                        "no_path_found",
                        f"unknown_action_primitive:{report.progress.get('contract')}",
                    }
                    or (report.reason or "").startswith("unknown_action_primitive:"),
                )
                cache_meta["invalidated"] = invalidated
            else:
                self.plan_cache.clear_failures(cache_key)
                cache_meta["invalidated"] = False

        context = ExecutionContext(
            active_skill=execution_contract.skill,
            params=dict(execution_contract.params),
        )
        return report, context, plan, cache_meta

    def _resolve_template(
        self,
        execution_contract,
        percepts,
        loop_index: int,
        allow_llm_compile: bool = True,
    ):
        cache_key = skill_key(execution_contract)
        if self.plan_cache is not None:
            entry = self.plan_cache.lookup(cache_key)
            if entry is not None:
                try:
                    self._validate_template(execution_contract, entry.template)
                except RuntimeError:
                    self.plan_cache.invalidate(cache_key)
                else:
                    return entry.template, {
                        "cache": "hit",
                        "source": "cache",
                        "compiler_source": entry.source,
                        "cache_key": cache_key,
                        "compiler_backend": entry.compiler_backend,
                        "runtime_compiler_call": False,
                    }

        compile_backend = self.compiler
        if not allow_llm_compile and getattr(self.compiler, "name", None) == "llm_compiler":
            compile_backend = getattr(self.compiler, "fallback", self.compiler)
            self.compiler.log(
                "Human render cache miss for skill plan; using local fallback instead of llm_compiler."
            )

        template = compile_backend.compile_skill_plan(
            execution_contract=execution_contract,
            percepts=percepts,
            available_action_primitives=ACTION_PRIMITIVES,
            memory=self.memory,
        )
        template = self._coerce_template_for_contract(execution_contract, template)
        self._validate_template(execution_contract, template)

        cache_status = "disabled"
        if self.plan_cache is not None and self.plan_cache.enabled:
            self.plan_cache.store(
                key=cache_key,
                template_type="skill",
                template=template,
                created_at_loop=loop_index,
            )
            cache_status = "miss"

        return template, {
            "cache": cache_status,
            "source": template.source,
            "compiler_source": template.source,
            "cache_key": cache_key,
            "compiler_backend": template.compiler_backend,
            "runtime_compiler_call": True,
        }

    def instantiate_template(self, template: SkillPlanTemplate, execution_contract):
        plan: list[PrimitiveCall] = []
        for primitive in template.primitives:
            if primitive == "plan_grid_path":
                plan.append(
                    PrimitiveCall(
                        name=primitive,
                        params={
                            "color": execution_contract.params.get("color"),
                            "object_type": execution_contract.params.get("object_type"),
                            "target_location": execution_contract.params.get("target_location"),
                        },
                    )
                )
            else:
                plan.append(PrimitiveCall(name=primitive))
        return plan

    def execute_plan(self, execution_contract, plan, percepts):
        runtime = {"planned_action_names": [], "path": []}

        for step in plan:
            if step.name not in ACTION_PRIMITIVES:
                return ExecutionReport(
                    status="failed",
                    reason=f"unknown_action_primitive:{step.name}",
                    progress={"contract": execution_contract.skill},
                )

            if step.name == "plan_grid_path":
                runtime = self._plan_grid_path(step.params, percepts)
                if not runtime["planned_action_names"]:
                    return ExecutionReport(
                        status="failed",
                        reason="no_path_found",
                        progress={"contract": execution_contract.skill, "path": runtime["path"]},
                    )
                continue

            if step.name == "execute_next_path_action":
                action_name = runtime["planned_action_names"][0]
                return self._execute_env_action(action_name, execution_contract, runtime)

            return self._execute_env_action(step.name, execution_contract, runtime)

        return ExecutionReport(status="failed", reason="empty_skill_plan")

    def _validate_template(self, execution_contract, template: SkillPlanTemplate) -> None:
        for primitive in template.primitives:
            if primitive not in ACTION_PRIMITIVES:
                raise RuntimeError(f"Unknown action primitive at runtime: {primitive}")
        skill = execution_contract.skill
        if skill == "navigate_to_object" and template.primitives != [
            "plan_grid_path",
            "execute_next_path_action",
        ]:
            raise RuntimeError(
                "navigate_to_object skill templates must be exactly "
                "['plan_grid_path', 'execute_next_path_action']"
            )
        if skill == "done" and template.primitives != ["done"]:
            raise RuntimeError("Done skill templates must be exactly ['done']")
        if skill in DIRECT_ACTION_SKILLS and template.primitives != [skill]:
            raise RuntimeError(f"Direct action skill templates must be exactly ['{skill}']")

    def _coerce_template_for_contract(self, execution_contract, template: SkillPlanTemplate) -> SkillPlanTemplate:
        skill = execution_contract.skill
        if skill == "navigate_to_object" and template.primitives != [
            "plan_grid_path",
            "execute_next_path_action",
        ]:
            self.compiler.log("corrected invalid navigate_to_object skill template")
            return SkillPlanTemplate(
                primitives=["plan_grid_path", "execute_next_path_action"],
                required_inputs=["agent_pose", "target_location", "occupancy_grid", "direction"],
                produces=["execution_report", "execution_context"],
                source=template.source,
                compiler_backend=template.compiler_backend,
                validated=True,
                rationale="Corrected invalid navigate_to_object template to the canonical path plan.",
            )
        if skill == "done" and template.primitives != ["done"]:
            self.compiler.log("corrected invalid done skill template")
            return SkillPlanTemplate(
                primitives=["done"],
                required_inputs=["adjacency_to_target"],
                produces=["execution_report", "execution_context"],
                source=template.source,
                compiler_backend=template.compiler_backend,
                validated=True,
                rationale="Corrected invalid done skill template to the canonical direct action.",
            )
        if skill in DIRECT_ACTION_SKILLS and template.primitives != [skill]:
            self.compiler.log(f"corrected invalid direct action skill template: {skill}")
            return SkillPlanTemplate(
                primitives=[skill],
                required_inputs=[],
                produces=["execution_report", "execution_context"],
                source=template.source,
                compiler_backend=template.compiler_backend,
                validated=True,
                rationale="Corrected invalid direct action template to the canonical singleton action.",
            )
        return template

    def _execute_env_action(self, action_name, execution_contract, runtime):
        action_spec = ACTION_PRIMITIVES.get(action_name)
        if action_spec is None or action_spec.runtime_kind != "env_action":
            return ExecutionReport(
                status="failed",
                reason=f"unknown_action_primitive:{action_name}",
                progress={"contract": execution_contract.skill},
            )

        observation, reward, terminated, truncated, info = self.adapter.act(int(action_spec.runtime_value))
        done = terminated or truncated

        if execution_contract.skill == "done":
            if done and reward > 0:
                status = "succeeded"
            elif done:
                status = "failed"
            else:
                status = "running"
        elif done:
            status = "failed"
        else:
            status = "running"

        return ExecutionReport(
            status=status,
            progress={
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
                "executed_action": action_name,
                "path": runtime.get("path"),
                "step_count": observation.step_count,
            },
            source="spine",
        )

    def _plan_grid_path(self, params, percepts):
        agent_pose = percepts.cues.get("agent_pose")
        target_location = params.get("target_location") or percepts.cues.get("target_location")
        target_object = percepts.cues.get("target_object") or {}
        passable_positions = set(percepts.cues.get("passable_positions") or set())
        grid_size = percepts.cues.get("grid_size")

        if not agent_pose or target_location is None or grid_size is None:
            return {"planned_action_names": [], "path": []}

        start = (agent_pose["x"], agent_pose["y"])
        target_type = params.get("object_type") or target_object.get("type")
        goal_positions = self._navigation_goals(
            target_location=target_location,
            target_type=target_type,
            passable_positions=passable_positions,
            grid_size=grid_size,
        )
        path = self._bfs_path(start, goal_positions, passable_positions)
        if not path:
            return {"planned_action_names": [], "path": []}

        return {
            "planned_action_names": self._path_to_actions(path, agent_pose["dir"]),
            "path": path,
        }

    def _navigation_goals(self, target_location, target_type, passable_positions, grid_size):
        width, height = grid_size
        if target_type == "door":
            neighbors = []
            tx, ty = target_location
            for dx, dy in DIR_TO_VEC.values():
                nx, ny = tx + dx, ty + dy
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if (nx, ny) in passable_positions:
                    neighbors.append((nx, ny))
            return neighbors
        return [tuple(target_location)]

    def _bfs_path(self, start, goals, passable_positions):
        goal_set = set(goals)
        if not goal_set:
            return []
        if start in goal_set:
            return [start]

        frontier = deque([start])
        parents = {start: None}

        while frontier:
            current = frontier.popleft()
            if current in goal_set:
                return self._reconstruct_path(current, parents)

            cx, cy = current
            for dx, dy in DIR_TO_VEC.values():
                nxt = (cx + dx, cy + dy)
                if nxt in parents or nxt not in passable_positions:
                    continue
                parents[nxt] = current
                frontier.append(nxt)

        return []

    def _reconstruct_path(self, goal, parents):
        path = [goal]
        current = goal
        while parents[current] is not None:
            current = parents[current]
            path.append(current)
        path.reverse()
        return path

    def _path_to_actions(self, path, direction):
        action_names = []
        current_dir = direction
        current_pos = path[0]

        for next_pos in path[1:]:
            move = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])
            desired_dir = VEC_TO_DIR[move]
            action_names.extend(self._turn_actions(current_dir, desired_dir))
            current_dir = desired_dir
            action_names.append("move_forward")
            current_pos = next_pos

        return action_names

    def _turn_actions(self, current_dir, desired_dir):
        diff = (desired_dir - current_dir) % 4
        if diff == 0:
            return []
        if diff == 1:
            return ["turn_right"]
        if diff == 2:
            return ["turn_right", "turn_right"]
        return ["turn_left"]
