from __future__ import annotations

from typing import Any, Protocol

from .capability_registry import CapabilityRegistry
from .llm_compiler import CompilerBackend
from .memory import OperationalMemory
from .orpi import OrpiManifest
from .plan_cache import PlanCache


class SubstrateAdapter(Protocol):
    """Concrete HOW boundary for environment/tool-specific bindings."""

    def capability_registry(self) -> CapabilityRegistry:
        ...

    def orpi_manifest(self) -> OrpiManifest:
        ...

    def create_sense(
        self,
        memory: OperationalMemory,
        compiler: CompilerBackend,
        plan_cache: PlanCache,
    ) -> Any:
        ...

    def create_spine(
        self,
        memory: OperationalMemory,
        compiler: CompilerBackend,
        plan_cache: PlanCache,
    ) -> Any:
        ...

    def known_action_names(self) -> list[str]:
        ...

    def is_action_known(self, action_name: str) -> bool:
        ...

    def prewarm_templates(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def open_preview(self, *, seed: int) -> None:
        ...

    def pump_render_window(self) -> None:
        ...

    def close_preview(self) -> None:
        ...

    def close_task_window(self) -> None:
        ...

    def has_preview_window(self) -> bool:
        ...

    def has_task_window(self) -> bool:
        ...

    def sense_idle_scene(self, sense: Any, *, seed: int) -> None:
        ...

    def run_task_episode(self, **kwargs: Any) -> dict[str, Any]:
        ...

    def run_motor_actions(self, *, seed: int, actions: list[str]) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...
