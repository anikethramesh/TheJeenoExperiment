"""Phase 10G probe: OperatorStationSession accepts an injected runtime package."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from harness import emit_result


ROOT = Path(__file__).resolve().parents[1]


class ProbeSubstrate:
    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def capability_registry(self) -> Any:
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
    def __init__(self, operational_context: Any) -> None:
        self.operational_context = operational_context
        self.supported_colors: tuple[str, ...] = ()

    def normalize_color(self, color: str) -> str:
        return color


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        from jeenom.capability_registry import CapabilityRegistry
        from jeenom.operator_station import OperatorStationSession
        from jeenom.runtime_package import RuntimePackage
        from jeenom.schemas import OperationalContext, PrimitiveManifest
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        CapabilityRegistry = None  # type: ignore[assignment]
        OperatorStationSession = None  # type: ignore[assignment]
        OperationalContext = None  # type: ignore[assignment]
        PrimitiveManifest = None  # type: ignore[assignment]
        RuntimePackage = None  # type: ignore[assignment]
        details["runtime_import_error"] = f"{type(exc).__name__}: {exc}"

    metrics["runtime_package_schema_exists"] = RuntimePackage is not None

    try:
        from jeenom.minigrid_runtime_package import build_minigrid_runtime_package
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        build_minigrid_runtime_package = None  # type: ignore[assignment]
        details["minigrid_runtime_import_error"] = f"{type(exc).__name__}: {exc}"

    metrics["default_minigrid_runtime_factory_exists"] = build_minigrid_runtime_package is not None

    if RuntimePackage and OperationalContext and PrimitiveManifest and CapabilityRegistry and OperatorStationSession:
        try:
            context = OperationalContext(
                context_id="probe.symbolic",
                substrate_id="probe",
                object_vocabulary=["token"],
                attribute_vocabulary=["name"],
            )
            registry = CapabilityRegistry(
                PrimitiveManifest.from_dict(
                    {"name": "phase10_runtime_probe_manifest", "primitives": []}
                )
            )
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
            result = session.handle_utterance("help")
            metrics["station_accepts_injected_runtime_package"] = (
                session.runtime_package is runtime_package
                and session.substrate is runtime_package.substrate
                and session.operational_context.context_id == "probe.symbolic"
                and session.domain_helper is runtime_package.domain_helper
            )
            metrics["injected_runtime_preserves_command_result"] = (
                bool(result.message)
                and session.last_command_result is result
            )
            metrics["injected_runtime_uses_package_registry"] = session.capability_registry is registry
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["injected_runtime_error"] = f"{type(exc).__name__}: {exc}"
            metrics["station_accepts_injected_runtime_package"] = False
            metrics["injected_runtime_preserves_command_result"] = False
            metrics["injected_runtime_uses_package_registry"] = False
    else:
        metrics["station_accepts_injected_runtime_package"] = False
        metrics["injected_runtime_preserves_command_result"] = False
        metrics["injected_runtime_uses_package_registry"] = False

    if RuntimePackage and build_minigrid_runtime_package:
        try:
            package = build_minigrid_runtime_package(
                env_id="MiniGrid-GoToDoor-8x8-v0",
                render_mode="none",
            )
            metrics["default_minigrid_runtime_is_package"] = isinstance(package, RuntimePackage)
            metrics["default_minigrid_runtime_has_context_and_helper"] = (
                package.operational_context.context_id == "minigrid.goto-door"
                and package.domain_helper.operational_context is package.operational_context
            )
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["default_runtime_error"] = f"{type(exc).__name__}: {exc}"
            metrics["default_minigrid_runtime_is_package"] = False
            metrics["default_minigrid_runtime_has_context_and_helper"] = False
    else:
        metrics["default_minigrid_runtime_is_package"] = False
        metrics["default_minigrid_runtime_has_context_and_helper"] = False

    station_source = (ROOT / "jeenom" / "operator_station.py").read_text()
    forbidden_inline_runtime = [
        "from .minigrid_domain_helper import MiniGridDomainHelper",
        "from .minigrid_operational_context import MiniGridOperationalContext",
        "from .minigrid_substrate_adapter import MiniGridSubstrateAdapter",
        "MiniGridDomainHelper(",
        "MiniGridOperationalContext.default(",
        "MiniGridSubstrateAdapter(",
    ]
    remaining = [needle for needle in forbidden_inline_runtime if needle in station_source]
    details["remaining_inline_minigrid_runtime_wiring"] = remaining
    metrics["station_no_longer_inlines_minigrid_runtime_wiring"] = not remaining

    metrics["phase10_runtime_package_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase10_runtime_package_holds")


if __name__ == "__main__":
    raise SystemExit(main())
