"""Phase 10E probe: MiniGrid domain meaning is delegated out of the station."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from harness import emit_result, make_session


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        from jeenom.minigrid_domain_helper import MiniGridDomainHelper
        from jeenom.minigrid_operational_context import MiniGridOperationalContext
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        MiniGridDomainHelper = None  # type: ignore[assignment]
        MiniGridOperationalContext = None  # type: ignore[assignment]
        details["domain_helper_import_error"] = f"{type(exc).__name__}: {exc}"

    metrics["domain_helper_module_exists"] = MiniGridDomainHelper is not None

    if MiniGridDomainHelper is not None and MiniGridOperationalContext is not None:
        try:
            context = MiniGridOperationalContext.default(env_id="MiniGrid-GoToDoor-8x8-v0")
            helper = MiniGridDomainHelper(context)
            details["supported_colors"] = list(helper.supported_colors)
            metrics["helper_consumes_operational_context"] = helper.operational_context is context
            metrics["helper_reads_domain_vocabulary"] = (
                "door" in helper.object_types
                and "red" in helper.supported_colors
                and helper.normalize_color("gray") == "grey"
            )
            parsed = helper.parse_exact_go_to_object_utterance("go to the gray door")
            metrics["helper_parses_domain_target"] = (
                parsed is not None
                and parsed["color"] == "grey"
                and parsed["object_type"] == "door"
            )
            metrics["helper_formats_ranked_claims"] = (
                "DOORS RANKED BY MANHATTAN DISTANCE FROM AGENT"
                in helper.format_ranked_doors_from_entries(
                    [
                        {"color": "red", "object_type": "door", "x": 1, "y": 2, "distance": 3}
                    ],
                    metric="manhattan",
                )
            )
        except Exception as exc:  # pragma: no cover - emitted as probe detail
            details["domain_helper_behavior_error"] = f"{type(exc).__name__}: {exc}"
            metrics["helper_consumes_operational_context"] = False
            metrics["helper_reads_domain_vocabulary"] = False
            metrics["helper_parses_domain_target"] = False
            metrics["helper_formats_ranked_claims"] = False
    else:
        metrics["helper_consumes_operational_context"] = False
        metrics["helper_reads_domain_vocabulary"] = False
        metrics["helper_parses_domain_target"] = False
        metrics["helper_formats_ranked_claims"] = False

    try:
        session = make_session()
        helper = getattr(session, "domain_helper", None)
        metrics["operator_station_has_domain_helper"] = MiniGridDomainHelper is not None and isinstance(
            helper,
            MiniGridDomainHelper,
        )
        metrics["station_domain_helper_matches_context"] = (
            helper is not None
            and getattr(helper, "operational_context", None) is session.operational_context
        )
    except Exception as exc:  # pragma: no cover - emitted as probe detail
        details["station_domain_helper_error"] = f"{type(exc).__name__}: {exc}"
        metrics["operator_station_has_domain_helper"] = False
        metrics["station_domain_helper_matches_context"] = False

    station_source = (ROOT / "jeenom" / "operator_station.py").read_text()
    forbidden_station_helpers = [
        "SUPPORTED_COLORS =",
        "def _normalize_color(",
        "def _color_reference_in_utterance(",
        "def _entry_label(",
        "def _format_ranked_doors_from_claims(",
        "def _color_answer(",
        "def _is_manhattan_answer(",
        "def _is_euclidean_answer(",
        "def _metric_from_grounding_handle(",
    ]
    remaining = [needle for needle in forbidden_station_helpers if needle in station_source]
    details["remaining_station_domain_helpers"] = remaining
    metrics["station_obvious_domain_helpers_extracted"] = not remaining

    metrics["phase10_domain_helper_holds"] = all(metrics.values())
    return emit_result(metrics, details, pass_metric="phase10_domain_helper_holds")


if __name__ == "__main__":
    raise SystemExit(main())
