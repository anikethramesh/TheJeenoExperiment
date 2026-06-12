"""ORPI conformance: MiniGrid registers a manifest at adapter/runtime init.

The permissive-window-is-never-live guarantee requires that the manifest exists on
the adapter *before* any other method is called on it.  We verify this by inspecting
the adapter's internal state immediately after construction, without calling
orpi_manifest() first.
"""
from __future__ import annotations

from harness import emit_result


def main() -> int:
    from jeenom.minigrid_operational_context import MiniGridOperationalContext
    from jeenom.minigrid_substrate_adapter import MiniGridSubstrateAdapter
    from jeenom.minigrid_runtime_package import build_minigrid_runtime_package
    from jeenom.orpi import OrpiManifest

    metrics: dict[str, bool] = {}
    details: dict[str, object] = {}

    # Check the adapter directly: manifest must be present before any method call.
    fresh_adapter = MiniGridSubstrateAdapter(
        env_id="MiniGrid-GoToDoor-8x8-v0",
        render_mode="none",
    )
    manifest_at_init = fresh_adapter._orpi_manifest  # noqa: SLF001
    metrics["manifest_exists_immediately_after_adapter_init"] = isinstance(
        manifest_at_init, OrpiManifest
    )

    # Check the runtime package path.
    package = build_minigrid_runtime_package(
        env_id="MiniGrid-GoToDoor-8x8-v0",
        render_mode="none",
    )
    adapter_manifest = package.substrate.orpi_manifest()
    runtime_manifest = package.resolve_orpi_manifest()
    payload = runtime_manifest.as_dict()
    details["manifest_keys"] = sorted(payload)
    details["symbol_mapping_keys"] = sorted(payload["symbol_mappings"])
    metrics["adapter_exposes_orpi_manifest_method"] = isinstance(adapter_manifest, OrpiManifest)
    metrics["runtime_package_caches_registered_manifest"] = runtime_manifest is adapter_manifest
    metrics["manifest_has_substrate_identity"] = (
        payload["substrate_id"] == "minigrid"
        and bool(payload["substrate_fingerprint"])
        and payload["orpi_version"] == "0"
    )
    metrics["manifest_has_symbol_frames_units_risk_policy"] = (
        bool(payload["symbol_mappings"].get("object_index"))
        and bool(payload["symbol_mappings"].get("color_index"))
        and "grid" in payload["frames"]
        and bool(payload["units"])
        and "actuation" in payload["risk_policy"]
    )
    metrics["orpi_manifest_registration_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="orpi_manifest_registration_holds")


if __name__ == "__main__":
    raise SystemExit(main())
