from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import MemoryUpdate


class OperationalMemory:
    def __init__(self, root: Path | None = None):
        project_root = root or Path(__file__).resolve().parent.parent
        self.memory_dir = project_root / "memory"
        self.memory_dir.mkdir(exist_ok=True)
        self.knowledge_path = self.memory_dir / "knowledge.yaml"

        self.understanding = {
            "go_to_object": {
                "description": "Navigate to a target object described by the operator or mission.",
                "default_recipe": [
                    "locate_object",
                    "navigate_to_object",
                    "verify_adjacent",
                    "done",
                ],
            },
            "search_for_object": {
                "description": "Search for a target object and navigate to it.",
                "default_recipe": [
                    "locate_object",
                    "navigate_to_object",
                    "verify_adjacent",
                    "done",
                ],
            },
        }

        self.embodied_intelligence = {
            "minigrid": {
                "substrate": "MiniGrid",
                "action_space": "discrete",
                "observation_space": "fully_observable_grid_encoding",
                "notes": "The runtime executes validated action primitives deterministically.",
            }
        }

        self.knowledge = {
            "target_color": None,
            "target_type": None,
            "last_task_type": None,
            "last_instruction": None,
        }
        self.knowledge.update(self._load_knowledge())

        self.episodic_memory = {
            "known_target_location": None,
            "last_world_sample": None,
            "last_trace_length": 0,
        }

    def resolve_target_params(self, task_params: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(task_params)
        if resolved.get("color") is None and self.knowledge.get("target_color") is not None:
            resolved["color"] = self.knowledge["target_color"]
        if resolved.get("object_type") is None and self.knowledge.get("target_type") is not None:
            resolved["object_type"] = self.knowledge["target_type"]
        return resolved

    def update_knowledge(self, key: str, value: Any, persist: bool = True) -> None:
        self.knowledge[key] = value
        if persist:
            self._persist_knowledge()

    def update_episodic_memory(self, key: str, value: Any) -> None:
        self.episodic_memory[key] = value

    def reset_episode(self) -> None:
        self.episodic_memory = {
            "known_target_location": None,
            "last_world_sample": None,
            "last_trace_length": 0,
        }

    def apply_memory_updates(self, updates: list[MemoryUpdate]) -> None:
        should_persist = False
        for update in updates:
            if update.scope == "knowledge":
                self.knowledge[update.key] = update.value
                should_persist = True
            elif update.scope == "episodic_memory":
                self.episodic_memory[update.key] = update.value
            else:
                raise ValueError(f"Unknown memory update scope: {update.scope}")

        if should_persist:
            self._persist_knowledge()

    def serializable_snapshot(self) -> dict[str, Any]:
        return {
            "understanding": self.understanding,
            "embodied_intelligence": self.embodied_intelligence,
            "knowledge": self.knowledge,
            "episodic_memory": self.episodic_memory,
        }

    def _load_knowledge(self) -> dict[str, Any]:
        if not self.knowledge_path.exists():
            return {}

        loaded: dict[str, Any] = {}
        for line in self.knowledge_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, raw_value = stripped.split(":", 1)
            value_text = raw_value.strip()
            try:
                loaded[key.strip()] = json.loads(value_text)
            except json.JSONDecodeError:
                loaded[key.strip()] = value_text or None
        return loaded

    def _persist_knowledge(self) -> None:
        lines = [
            f"{key}: {json.dumps(self.knowledge.get(key))}"
            for key in ["target_color", "target_type", "last_task_type", "last_instruction"]
        ]
        self.knowledge_path.write_text("\n".join(lines) + "\n")
