from __future__ import annotations

import unittest
from typing import Any

from jeenom.capability_registry import CapabilityRegistry
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import OperationalContext, PrimitiveManifest


class ProbeSubstrate:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def capability_registry(self) -> CapabilityRegistry:
        return self._registry

    def create_sense(self, memory: Any, compiler: Any, plan_cache: Any) -> object:
        return object()

    def create_spine(self, memory: Any, compiler: Any, plan_cache: Any) -> object:
        return object()

    def known_action_names(self) -> list[str]:
        return []

    def is_action_known(self, action_name: str) -> bool:
        return False

    def prewarm_templates(self, **kwargs: Any) -> dict[str, Any]:
        return {"compiled_templates": []}

    def open_preview(self, *, seed: int) -> None:
        return None

    def pump_render_window(self) -> None:
        return None

    def close_preview(self) -> None:
        return None

    def close_task_window(self) -> None:
        return None

    def has_preview_window(self) -> bool:
        return False

    def has_task_window(self) -> bool:
        return False

    def sense_idle_scene(self, sense: Any, *, seed: int) -> None:
        return None

    def run_task_episode(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "task_complete": False,
            "runtime_llm_calls": 0,
            "cache_misses": 0,
            "final_report": {"status": "unsupported", "reason": "probe runtime"},
        }

    def run_motor_actions(self, *, seed: int, actions: list[str]) -> dict[str, Any]:
        return {"success": False, "actions_executed": []}

    def close(self) -> None:
        return None


class ProbeDomainHelper:
    def __init__(self, operational_context: OperationalContext) -> None:
        self.operational_context = operational_context
        self.supported_colors: tuple[str, ...] = ()

    def normalize_color(self, color: str) -> str:
        return color


class TestPhase10RuntimePackage(unittest.TestCase):
    def _registry(self) -> CapabilityRegistry:
        return CapabilityRegistry(
            PrimitiveManifest.from_dict(
                {"name": "phase10_runtime_test_manifest", "primitives": []}
            )
        )

    def test_station_accepts_injected_runtime_package(self):
        from jeenom.runtime_package import RuntimePackage

        context = OperationalContext(
            context_id="probe.symbolic",
            substrate_id="probe",
            object_vocabulary=["token"],
            attribute_vocabulary=["name"],
        )
        registry = self._registry()
        runtime_package = RuntimePackage(
            substrate=ProbeSubstrate(registry),
            operational_context=context,
            domain_helper=ProbeDomainHelper(context),
            capability_registry=registry,
        )

        session = OperatorStationSession(
            compiler_name="smoke_test",
            render_mode="none",
            runtime_package=runtime_package,
        )

        self.assertIs(session.runtime_package, runtime_package)
        self.assertIs(session.substrate, runtime_package.substrate)
        self.assertIs(session.domain_helper, runtime_package.domain_helper)
        self.assertEqual(session.operational_context.context_id, "probe.symbolic")
        self.assertIs(session.capability_registry, registry)

    def test_injected_runtime_preserves_public_command_result(self):
        from jeenom.runtime_package import RuntimePackage

        context = OperationalContext(
            context_id="probe.symbolic",
            substrate_id="probe",
            object_vocabulary=["token"],
            attribute_vocabulary=["name"],
        )
        registry = self._registry()
        session = OperatorStationSession(
            compiler_name="smoke_test",
            render_mode="none",
            runtime_package=RuntimePackage(
                substrate=ProbeSubstrate(registry),
                operational_context=context,
                domain_helper=ProbeDomainHelper(context),
                capability_registry=registry,
            ),
        )

        result = session.handle_utterance("help")

        self.assertTrue(result.message)
        self.assertIs(session.last_command_result, result)

    def test_default_minigrid_runtime_package_factory(self):
        from jeenom.minigrid_runtime_package import build_minigrid_runtime_package
        from jeenom.runtime_package import RuntimePackage

        package = build_minigrid_runtime_package(
            env_id="MiniGrid-GoToDoor-8x8-v0",
            render_mode="none",
        )

        self.assertIsInstance(package, RuntimePackage)
        self.assertEqual(package.operational_context.context_id, "minigrid.goto-door")
        self.assertIs(package.domain_helper.operational_context, package.operational_context)


if __name__ == "__main__":
    unittest.main()
