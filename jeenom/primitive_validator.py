"""Phase 7.8 — Primitive Validator.

Validates a synthesized Python grounding primitive against deterministic SceneModel
fixtures before it can be registered and executed.

Architecture:
  SynthesisResult (code)
    → PrimitiveValidator.validate(result, contract)
    → ValidationResult (passed, failures)
    → CapabilityRegistry.register_synthesized(handle, fn)  if passed

Contract:
  - Input: (scene: SceneModel, selector: dict) → list[tuple[float, SceneObject]]
  - Output must be sorted ascending by distance (index 0 = nearest).
  - Empty scene must return [].
  - Color and exclude_colors filters from selector must be respected.
  - Must not raise for any valid fixture input.

No substrate imports. Uses SceneModel directly (pure Python dataclass).
"""
from __future__ import annotations

import math
import types
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ValidationFixture:
    name: str
    scene_kwargs: dict[str, Any]
    selector: dict[str, Any]
    expected_colors_in_order: list[str] | None = None  # None = any non-empty result OK
    expected_empty: bool = False
    expected_count: int | None = None


@dataclass
class ValidationResult:
    handle: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    compiled_fn: Callable | None = None


def _make_scene(
    agent_x: int,
    agent_y: int,
    doors: list[dict[str, Any]],
) -> Any:
    """Build a minimal SceneModel-like object for validation fixtures.

    Avoids importing from schemas to keep this module substrate-independent.
    The fixture scene only needs .find() and .agent_x/.agent_y attributes.
    """
    from jeenom.schemas import SceneModel, SceneObject

    objects = [
        SceneObject(
            object_type=d.get("object_type", "door"),
            color=d["color"],
            x=d["x"],
            y=d["y"],
        )
        for d in doors
    ]
    return SceneModel(
        agent_x=agent_x,
        agent_y=agent_y,
        agent_dir=0,
        grid_width=16,
        grid_height=16,
        objects=objects,
        source="validation_fixture",
    )


EUCLIDEAN_FIXTURES: list[ValidationFixture] = [
    ValidationFixture(
        name="empty_scene_returns_empty",
        scene_kwargs={"agent_x": 1, "agent_y": 1, "doors": []},
        selector={"object_type": "door", "color": None, "exclude_colors": []},
        expected_empty=True,
    ),
    ValidationFixture(
        name="single_door_returned",
        scene_kwargs={
            "agent_x": 1,
            "agent_y": 1,
            "doors": [{"color": "red", "x": 4, "y": 5}],
        },
        selector={"object_type": "door", "color": None, "exclude_colors": []},
        expected_count=1,
        expected_colors_in_order=["red"],
    ),
    ValidationFixture(
        name="nearest_door_is_first",
        scene_kwargs={
            "agent_x": 1,
            "agent_y": 1,
            # red: euclidean = sqrt((3-1)^2 + (1-1)^2) = 2.0
            # blue: euclidean = sqrt((1-1)^2 + (5-1)^2) = 4.0
            "doors": [
                {"color": "blue", "x": 1, "y": 5},
                {"color": "red", "x": 3, "y": 1},
            ],
        },
        selector={"object_type": "door", "color": None, "exclude_colors": []},
        expected_colors_in_order=["red", "blue"],
    ),
    ValidationFixture(
        name="farther_door_is_last",
        scene_kwargs={
            "agent_x": 0,
            "agent_y": 0,
            # purple: euclidean = sqrt(9+16) = 5.0
            # green: euclidean = sqrt(1+1) = sqrt(2) ≈ 1.41
            "doors": [
                {"color": "purple", "x": 3, "y": 4},
                {"color": "green", "x": 1, "y": 1},
            ],
        },
        selector={"object_type": "door", "color": None, "exclude_colors": []},
        expected_colors_in_order=["green", "purple"],
    ),
    ValidationFixture(
        name="color_filter_respected",
        scene_kwargs={
            "agent_x": 1,
            "agent_y": 1,
            "doors": [
                {"color": "red", "x": 2, "y": 2},
                {"color": "blue", "x": 3, "y": 3},
            ],
        },
        selector={"object_type": "door", "color": "red", "exclude_colors": []},
        expected_colors_in_order=["red"],
    ),
    ValidationFixture(
        name="exclude_color_respected",
        scene_kwargs={
            "agent_x": 1,
            "agent_y": 1,
            "doors": [
                {"color": "yellow", "x": 2, "y": 2},
                {"color": "green", "x": 4, "y": 4},
            ],
        },
        selector={"object_type": "door", "color": None, "exclude_colors": ["yellow"]},
        expected_colors_in_order=["green"],
    ),
    ValidationFixture(
        name="distance_is_euclidean_not_manhattan",
        scene_kwargs={
            "agent_x": 0,
            "agent_y": 0,
            # Manhattan of red=(1,4) → 5; Manhattan of blue=(3,3) → 6
            # Euclidean of red=(1,4) → sqrt(17) ≈ 4.12; Euclidean of blue=(3,3) → sqrt(18) ≈ 4.24
            # Both orderings agree for this pair; use a pair where they differ:
            # red=(0,5): manhattan=5, euclidean=5
            # blue=(4,3): manhattan=7, euclidean=5 (tie in euclidean)
            # Let's use: red=(1,4) eucl≈4.12, green=(3,2) eucl≈3.6 → green first
            "doors": [
                {"color": "red", "x": 1, "y": 4},   # eucl=sqrt(17)≈4.12
                {"color": "green", "x": 3, "y": 2},  # eucl=sqrt(13)≈3.6
            ],
        },
        selector={"object_type": "door", "color": None, "exclude_colors": []},
        expected_colors_in_order=["green", "red"],
    ),
]


FIXTURE_SETS: dict[str, list[ValidationFixture]] = {
    "closest_door.euclidean.agent": EUCLIDEAN_FIXTURES,
}

_GENERIC_GROUNDING_FIXTURES: list[ValidationFixture] = [
    ValidationFixture(
        name="empty_scene_returns_empty",
        scene_kwargs={"agent_x": 1, "agent_y": 1, "doors": []},
        selector={"object_type": "door", "color": None, "exclude_colors": []},
        expected_empty=True,
    ),
    ValidationFixture(
        name="returns_list",
        scene_kwargs={
            "agent_x": 1,
            "agent_y": 1,
            "doors": [{"color": "red", "x": 3, "y": 3}],
        },
        selector={"object_type": "door", "color": None, "exclude_colors": []},
        expected_count=1,
    ),
]


def _get_fixtures(handle: str) -> list[ValidationFixture]:
    name = handle.split("grounding.")[-1] if "grounding." in handle else handle
    return FIXTURE_SETS.get(name, _GENERIC_GROUNDING_FIXTURES)


class PrimitiveValidator:
    """Validates a synthesized grounding primitive against deterministic fixtures.

    Executes the generated code in a restricted namespace. No MiniGrid or env imports.
    """

    ALLOWED_BUILTINS = {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter",
        "float", "int", "isinstance", "len", "list", "map", "max",
        "min", "None", "range", "round", "set", "sorted", "str",
        "sum", "tuple", "zip",
    }

    def validate(
        self,
        handle: str,
        function_name: str,
        code: str,
    ) -> ValidationResult:
        fixtures = _get_fixtures(handle)
        compiled_fn, compile_error = self._compile(function_name, code)
        if compiled_fn is None:
            return ValidationResult(
                handle=handle,
                passed=False,
                failures=[f"Compilation failed: {compile_error}"],
            )

        failures: list[str] = []
        for fixture in fixtures:
            failure = self._run_fixture(compiled_fn, fixture)
            if failure:
                failures.append(failure)

        return ValidationResult(
            handle=handle,
            passed=len(failures) == 0,
            failures=failures,
            compiled_fn=compiled_fn if not failures else None,
        )

    def _compile(
        self,
        function_name: str,
        code: str,
    ) -> tuple[Callable | None, str]:
        namespace: dict[str, Any] = {
            "__builtins__": {k: __builtins__[k] for k in self.ALLOWED_BUILTINS if k in (  # type: ignore[index]
                __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)  # type: ignore[arg-type]
            )},
            "math": math,
        }
        try:
            exec(compile(code, f"<synthesized:{function_name}>", "exec"), namespace)  # noqa: S102
        except SyntaxError as exc:
            return None, f"SyntaxError: {exc}"
        except Exception as exc:  # noqa: BLE001
            return None, f"Exec error: {exc}"

        fn = namespace.get(function_name)
        if fn is None:
            return None, f"Function '{function_name}' not found in generated code."
        if not callable(fn):
            return None, f"'{function_name}' is not callable."
        return fn, ""

    def _run_fixture(
        self,
        fn: Callable,
        fixture: ValidationFixture,
    ) -> str | None:
        scene = _make_scene(**fixture.scene_kwargs)
        try:
            result = fn(scene, fixture.selector)
        except Exception as exc:  # noqa: BLE001
            return f"[{fixture.name}] raised {type(exc).__name__}: {exc}"

        if not isinstance(result, list):
            return f"[{fixture.name}] expected list, got {type(result).__name__}"

        if fixture.expected_empty and len(result) != 0:
            return f"[{fixture.name}] expected empty list, got {len(result)} items"

        if fixture.expected_count is not None and len(result) != fixture.expected_count:
            return (
                f"[{fixture.name}] expected {fixture.expected_count} items, "
                f"got {len(result)}"
            )

        if fixture.expected_colors_in_order is not None:
            actual_colors = []
            for item in result:
                if not (isinstance(item, tuple) and len(item) == 2):
                    return (
                        f"[{fixture.name}] result items must be (distance, SceneObject) "
                        f"tuples, got {type(item)}"
                    )
                dist, obj = item
                if not isinstance(dist, (int, float)):
                    return (
                        f"[{fixture.name}] distance must be numeric, got {type(dist)}"
                    )
                actual_colors.append(getattr(obj, "color", None))

            if actual_colors != fixture.expected_colors_in_order:
                return (
                    f"[{fixture.name}] expected order {fixture.expected_colors_in_order}, "
                    f"got {actual_colors}"
                )

            # Verify ascending order
            dists = [item[0] for item in result]
            if dists != sorted(dists):
                return (
                    f"[{fixture.name}] result is not sorted ascending by distance: {dists}"
                )

        return None


default_validator = PrimitiveValidator()
