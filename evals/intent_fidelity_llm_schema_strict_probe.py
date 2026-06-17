"""Every JSON schema sent to the LLM under `strict: True` must satisfy OpenAI's
strict structured-output rules, or the provider rejects the call with HTTP 400 and
the compiler silently falls back to the deterministic NLU — the LLM never runs.

This is a regression guard for a real, invisible failure: `operator_intent_json_schema`
grew a `steering_directive` field (and earlier a `primitive_definition`) whose nested
objects lacked `additionalProperties: false` and whose parents omitted the new keys from
`required`. Because the compiler swallows the 400 and falls back, no behavioural eval
caught it — every probe runs offline against the deterministic compiler. This probe
exercises the schema *structure* directly (no API call), so the blind spot is closed.

OpenAI strict mode requires, for every object that declares `properties`:
  1. `additionalProperties: false`
  2. EVERY declared property listed in `required` (optionality is expressed via a
     nullable type, e.g. `["string", "null"]`, never by omission from `required`).

The probe collects every schema the compiler/arbitrator/synthesizer wrap with
`strict: True` and asserts both rules recursively. It must fail on a repo that
reintroduces either violation.
"""
from __future__ import annotations

from typing import Any

from harness import emit_result


# --- The exact schemas wrapped with strict:True across the LLM-bound call sites. ---
# llm_compiler.py: response_format.json_schema.strict = True for all six.
# capability_arbitrator.py + primitive_synthesizer.py: their own strict paths.
def _collect_strict_schemas() -> dict[str, dict[str, Any]]:
    from jeenom.schemas import (
        task_request_json_schema,
        procedure_recipe_json_schema,
        sense_plan_json_schema,
        skill_plan_json_schema,
        memory_updates_json_schema,
        operator_intent_json_schema,
        primitive_call_json_schema,
        primitive_params_json_schema,
    )
    from jeenom.capability_arbitrator import arbitration_decision_json_schema
    from jeenom.primitive_synthesizer import synthesis_response_json_schema

    prims = ["go_to_object", "pickup_object", "toggle_object", "drop_object"]
    tasks = ["go_to_object", "pickup_object"]

    return {
        "task_request": task_request_json_schema(tasks),
        "procedure_recipe": procedure_recipe_json_schema(prims, tasks),
        "sense_plan": sense_plan_json_schema(prims),
        "skill_plan": skill_plan_json_schema(prims),
        "memory_updates": memory_updates_json_schema(),
        "operator_intent": operator_intent_json_schema(),
        "primitive_call": primitive_call_json_schema(prims),
        "primitive_params": primitive_params_json_schema(),
        "arbitration_decision": arbitration_decision_json_schema(),
        "synthesis_response": synthesis_response_json_schema(),
    }


def _is_object_type(schema: dict[str, Any]) -> bool:
    typ = schema.get("type")
    if typ == "object":
        return True
    if isinstance(typ, list) and "object" in typ:
        return True
    # A schema with `properties` but no explicit type is still an object to OpenAI.
    return "properties" in schema


def _strict_violations(schema: Any, path: str = "") -> list[str]:
    """Return every OpenAI strict-mode violation in `schema`, deepest-first paths."""
    issues: list[str] = []
    if isinstance(schema, dict):
        if _is_object_type(schema):
            here = path or "(root)"
            # Rule 1: every object — even one with no declared properties — must
            # forbid extra keys. A bare {"type": "object"} is the exact shape that
            # tripped the original HTTP 400.
            if schema.get("additionalProperties") is not False:
                issues.append(f"{here}: object schema missing additionalProperties: false")
            # Rule 2: every declared property must be required (optionality via
            # nullable type, never via omission from `required`).
            props = schema.get("properties", {})
            required = set(schema.get("required", []))
            for key in props:
                if key not in required:
                    issues.append(f"{here}: property '{key}' is not listed in `required`")
            for key, sub in props.items():
                issues.extend(_strict_violations(sub, f"{path}.{key}"))
        items = schema.get("items")
        if isinstance(items, dict):
            issues.extend(_strict_violations(items, f"{path}[]"))
        for combinator in ("anyOf", "oneOf", "allOf"):
            for index, sub in enumerate(schema.get(combinator, [])):
                issues.extend(_strict_violations(sub, f"{path}.{combinator}[{index}]"))
    return issues


def main() -> int:
    metrics: dict[str, bool] = {}
    details: dict[str, Any] = {}

    try:
        schemas = _collect_strict_schemas()
    except Exception as exc:  # pragma: no cover - import/shape failure surfaced as red
        details["collect_error"] = f"{type(exc).__name__}: {exc}"
        metrics["all_strict_schemas_openai_compliant"] = False
        return emit_result(metrics, details, pass_metric="all_strict_schemas_openai_compliant")

    for name, schema in schemas.items():
        violations = _strict_violations(schema)
        metrics[f"{name}_is_strict_compliant"] = not violations
        if violations:
            details[f"{name}_violations"] = violations

    # The schema that actually regressed — call it out by name so a future break is
    # legible in the metric list, not buried in a generic aggregate.
    metrics["operator_intent_is_strict_compliant"] = not _strict_violations(
        schemas["operator_intent"]
    )

    metrics["all_strict_schemas_openai_compliant"] = all(metrics.values())
    return emit_result(
        metrics, details, pass_metric="all_strict_schemas_openai_compliant"
    )


if __name__ == "__main__":
    raise SystemExit(main())
