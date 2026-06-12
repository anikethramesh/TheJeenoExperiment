"""Phase 10D probe: OperationalContext is a typed situation boundary."""
from __future__ import annotations

from typing import Any

from harness import emit_result, make_session


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        import jeenom.schemas as schemas
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        schemas = None  # type: ignore[assignment]
        details["schemas_import_error"] = f"{type(exc).__name__}: {exc}"

    context_cls = getattr(schemas, "OperationalContext", None) if schemas else None
    metrics["operational_context_schema_exists"] = context_cls is not None

    try:
        import jeenom.minigrid_operational_context as minigrid_context
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        minigrid_context = None  # type: ignore[assignment]
        details["minigrid_context_import_error"] = f"{type(exc).__name__}: {exc}"

    minigrid_cls = (
        getattr(minigrid_context, "MiniGridOperationalContext", None)
        if minigrid_context is not None
        else None
    )
    metrics["minigrid_operational_context_module_exists"] = minigrid_context is not None
    metrics["minigrid_operational_context_type_exists"] = minigrid_cls is not None

    if minigrid_cls is not None and context_cls is not None:
        try:
            ctx = minigrid_cls.default(env_id="MiniGrid-GoToDoor-8x8-v0")
            full = ctx.as_dict()
            compact = ctx.compact_slice("go to the red door")
            details["context_fingerprint"] = ctx.fingerprint()
            details["compact_slice"] = compact
            metrics["minigrid_context_is_schema_instance"] = isinstance(ctx, context_cls)
            metrics["context_fingerprint_is_stable_string"] = (
                isinstance(ctx.fingerprint(), str)
                and len(ctx.fingerprint()) >= 16
                and ctx.fingerprint() == minigrid_cls.default(
                    env_id="MiniGrid-GoToDoor-8x8-v0"
                ).fingerprint()
            )
            metrics["context_has_domain_vocabulary"] = (
                "door" in ctx.object_vocabulary
                and "color" in ctx.attribute_vocabulary
                and any(
                    task.get("task_type") == "go_to_object"
                    for task in ctx.task_families
                )
            )
            metrics["compact_slice_is_smaller_than_full_context"] = (
                len(repr(compact)) < len(repr(full))
            )
            metrics["compact_slice_not_full_context_dump"] = not any(
                key in compact
                for key in ("display_rules", "claim_rules", "procedure_hints", "metadata")
            )
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["minigrid_context_behavior_error"] = f"{type(exc).__name__}: {exc}"
            metrics["minigrid_context_is_schema_instance"] = False
            metrics["context_fingerprint_is_stable_string"] = False
            metrics["context_has_domain_vocabulary"] = False
            metrics["compact_slice_is_smaller_than_full_context"] = False
            metrics["compact_slice_not_full_context_dump"] = False
    else:
        metrics["minigrid_context_is_schema_instance"] = False
        metrics["context_fingerprint_is_stable_string"] = False
        metrics["context_has_domain_vocabulary"] = False
        metrics["compact_slice_is_smaller_than_full_context"] = False
        metrics["compact_slice_not_full_context_dump"] = False

    if context_cls is not None and minigrid_cls is not None:
        try:
            session = make_session()
            ctx = getattr(session, "operational_context", None)
            metrics["operator_station_has_operational_context"] = isinstance(
                ctx,
                context_cls,
            )
            metrics["station_context_matches_minigrid_context"] = isinstance(
                ctx,
                minigrid_cls,
            )
            metrics["station_context_has_fingerprint"] = (
                isinstance(getattr(session, "context_fingerprint", None), str)
                and session.context_fingerprint == ctx.fingerprint()
            )
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["station_context_error"] = f"{type(exc).__name__}: {exc}"
            metrics["operator_station_has_operational_context"] = False
            metrics["station_context_matches_minigrid_context"] = False
            metrics["station_context_has_fingerprint"] = False
    else:
        metrics["operator_station_has_operational_context"] = False
        metrics["station_context_matches_minigrid_context"] = False
        metrics["station_context_has_fingerprint"] = False

    metrics["phase10_operational_context_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase10_operational_context_holds")


if __name__ == "__main__":
    raise SystemExit(main())
