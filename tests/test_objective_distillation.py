"""Tests for objective distillation — SelectionObjective-based validation.

These tests verify that:
  1. SelectionObjective is parsed correctly from LLM output dicts.
  2. IntentVerifier uses the objective-based path (pure enum logic) when
     selection_objective is present — no vocabulary scanning.
  3. Inversion is detected via objective.direction vs plan.order — not text.
  4. Handle injection uses objective.metric — not superlative vocabulary.
  5. Vocabulary fallback still fires when selection_objective is absent.
  6. Dual-direction plans (closest AND farthest) are not flagged as inversions.
  7. Future domains work: "maximum" direction works regardless of attribute name.
"""
from __future__ import annotations

import unittest

from jeenom.schemas import (
    OperatorIntent,
    SchemaValidationError,
    SelectionObjective,
    SELECTION_DIRECTIONS,
)
from jeenom.intent_verifier import IntentVerifier


def _make_intent(
    order: str | None,
    direction: str | None = None,
    ordinal: int = 1,
    metric: str | None = "manhattan",
    attribute: str = "distance",
    answer_fields: list[str] | None = None,
) -> OperatorIntent:
    """Build a minimal task_instruction intent with a grounding_query_plan."""
    ranked_handle = f"grounding.all_doors.ranked.{metric or 'manhattan'}.agent"
    plan: dict = {
        "object_type": "door",
        "operation": "select",
        "primitive_handle": ranked_handle,
        "metric": metric,
        "reference": "agent",
        "order": order,
        "ordinal": ordinal,
        "color": None,
        "exclude_colors": [],
        "distance_value": None,
        "tie_policy": "clarify",
        "answer_fields": answer_fields or ["target", "distance"],
        "required_capabilities": [ranked_handle, "task.go_to_object.door"],
        "preserved_constraints": [],
    }
    selection_obj = (
        SelectionObjective(
            attribute=attribute,
            direction=direction,
            ordinal=ordinal,
            metric=metric,
        )
        if direction is not None
        else None
    )
    return OperatorIntent(
        intent_type="task_instruction",
        task_type="go_to_object",
        grounding_query_plan=plan,
        required_capabilities=[ranked_handle, "task.go_to_object.door"],
        selection_objective=selection_obj,
        confidence=0.9,
        reason="test",
    )


class TestSelectionObjectiveParsing(unittest.TestCase):
    """SelectionObjective.from_dict() validates and rejects invalid input."""

    def test_maximum_direction(self):
        obj = SelectionObjective.from_dict({
            "attribute": "distance",
            "direction": "maximum",
            "ordinal": 1,
            "metric": "manhattan",
        })
        self.assertEqual(obj.direction, "maximum")
        self.assertEqual(obj.attribute, "distance")
        self.assertEqual(obj.ordinal, 1)
        self.assertEqual(obj.metric, "manhattan")

    def test_minimum_direction(self):
        obj = SelectionObjective.from_dict({
            "attribute": "distance",
            "direction": "minimum",
            "ordinal": 2,
            "metric": None,
        })
        self.assertEqual(obj.direction, "minimum")
        self.assertEqual(obj.ordinal, 2)
        self.assertIsNone(obj.metric)

    def test_none_input_returns_none(self):
        self.assertIsNone(SelectionObjective.from_dict(None))

    def test_invalid_direction_raises(self):
        with self.assertRaises(SchemaValidationError) as ctx:
            SelectionObjective.from_dict({
                "attribute": "distance",
                "direction": "ascending",  # WRONG: ascending/descending not valid here
                "ordinal": 1,
                "metric": None,
            })
        self.assertIn("minimum", str(ctx.exception))
        self.assertIn("maximum", str(ctx.exception))

    def test_descending_direction_raises(self):
        with self.assertRaises(SchemaValidationError):
            SelectionObjective.from_dict({
                "attribute": "distance",
                "direction": "descending",
                "ordinal": 1,
                "metric": None,
            })

    def test_future_attribute_accepted(self):
        """SelectionObjective.attribute is open — any string works."""
        obj = SelectionObjective.from_dict({
            "attribute": "temperature",
            "direction": "maximum",
            "ordinal": 1,
            "metric": None,
        })
        self.assertEqual(obj.attribute, "temperature")

    def test_ordinal_defaults_to_1(self):
        obj = SelectionObjective.from_dict({
            "attribute": "distance",
            "direction": "maximum",
            "metric": None,
        })
        self.assertEqual(obj.ordinal, 1)

    def test_from_dict_roundtrip_via_operator_intent(self):
        """OperatorIntent.from_dict() parses selection_objective correctly."""
        raw = {
            "intent_type": "task_instruction",
            "task_type": "go_to_object",
            "target": None,
            "target_selector": None,
            "clear_memory": False,
            "selection_objective": {
                "attribute": "distance",
                "direction": "maximum",
                "ordinal": 1,
                "metric": "manhattan",
            },
            "grounding_query_plan": {
                "object_type": "door",
                "operation": "select",
                "primitive_handle": "grounding.all_doors.ranked.manhattan.agent",
                "metric": "manhattan",
                "reference": "agent",
                "order": "descending",
                "ordinal": 1,
                "color": None,
                "exclude_colors": [],
                "distance_value": None,
                "tie_policy": "clarify",
                "answer_fields": ["target", "distance"],
                "required_capabilities": [
                    "grounding.all_doors.ranked.manhattan.agent",
                    "task.go_to_object.door",
                ],
                "preserved_constraints": [],
            },
            "required_capabilities": [
                "grounding.all_doors.ranked.manhattan.agent",
                "task.go_to_object.door",
            ],
            "capability_status": "executable",
            "confidence": 0.9,
            "reason": "",
        }
        intent = OperatorIntent.from_dict(raw)
        self.assertIsNotNone(intent.selection_objective)
        self.assertEqual(intent.selection_objective.direction, "maximum")
        self.assertEqual(intent.selection_objective.attribute, "distance")

    def test_missing_selection_objective_gives_none(self):
        """OperatorIntent without selection_objective field parses fine (backwards compat)."""
        raw = {
            "intent_type": "status_query",
            "status_query": "scene",
            "target": None,
            "target_selector": None,
            "clear_memory": False,
            "grounding_query_plan": None,
            "required_capabilities": [],
            "capability_status": "executable",
            "confidence": 0.9,
            "reason": "",
        }
        intent = OperatorIntent.from_dict(raw)
        self.assertIsNone(intent.selection_objective)


class TestObjectiveBasedInversionDetection(unittest.TestCase):
    """IntentVerifier uses selection_objective.direction for inversion — not text."""

    def setUp(self):
        self.verifier = IntentVerifier()

    def _enrich(self, utterance: str, intent: OperatorIntent):
        return self.verifier.enrich(utterance, intent)

    def test_maximum_with_descending_is_valid(self):
        """direction=maximum + order=descending: no inversion."""
        intent = _make_intent(order="descending", direction="maximum")
        _, result = self._enrich("go to the farthest door", intent)
        self.assertFalse(result.inversion_detected, result.inversion_reason)

    def test_minimum_with_ascending_is_valid(self):
        """direction=minimum + order=ascending: no inversion."""
        intent = _make_intent(order="ascending", direction="minimum")
        _, result = self._enrich("go to the closest door", intent)
        self.assertFalse(result.inversion_detected, result.inversion_reason)

    def test_maximum_with_ascending_is_inversion(self):
        """direction=maximum + order=ascending: inversion detected via objective path."""
        intent = _make_intent(order="ascending", direction="maximum")
        _, result = self._enrich("go to the farthest door", intent)
        self.assertTrue(result.inversion_detected)
        self.assertIn("maximum", result.inversion_reason)
        self.assertIn("descending", result.inversion_reason)

    def test_minimum_with_descending_is_inversion(self):
        """direction=minimum + order=descending: inversion detected via objective path."""
        intent = _make_intent(order="descending", direction="minimum")
        _, result = self._enrich("go to the closest door", intent)
        self.assertTrue(result.inversion_detected)
        self.assertIn("minimum", result.inversion_reason)
        self.assertIn("ascending", result.inversion_reason)

    def test_objective_path_does_not_need_utterance_text(self):
        """Objective-based check fires even with a non-English-like utterance.

        This proves the check is purely structural — it does NOT scan the
        utterance text. 'FOOBAR_DOMAIN_WORD' is an unknown word but the
        objective carries direction=maximum, so the inversion is still caught.
        """
        intent = _make_intent(order="ascending", direction="maximum")
        _, result = self._enrich("FOOBAR_DOMAIN_WORD door target", intent)
        self.assertTrue(
            result.inversion_detected,
            "Objective-based check must fire even when utterance has no vocabulary match.",
        )

    def test_dual_direction_answer_plan_not_flagged(self):
        """closest+farthest plan with answer_fields covering both: not an inversion."""
        intent = _make_intent(
            order="ascending",
            direction="maximum",
            answer_fields=["farthest", "closest"],
        )
        _, result = self._enrich("which door is closest and which is farthest", intent)
        self.assertFalse(
            result.inversion_detected,
            "Dual-direction answer plan must not be flagged as inversion.",
        )

    def test_future_attribute_temperature_inversion_detected(self):
        """Inversion detection works for any attribute — 'temperature' example.

        This is the core scalability proof: adding a new domain ('temperature')
        requires NO changes to IntentVerifier. The objective carries direction;
        the check is attribute-agnostic.
        """
        intent = _make_intent(order="ascending", direction="maximum", attribute="temperature")
        _, result = self._enrich("go to the hottest room", intent)
        self.assertTrue(
            result.inversion_detected,
            "direction=maximum + order=ascending must be detected for any attribute.",
        )

    def test_vocabulary_fallback_when_no_objective(self):
        """When selection_objective is None, vocabulary-based check still catches inversions."""
        intent = _make_intent(order="ascending", direction=None)  # no objective
        _, result = self._enrich("go to the farthest door", intent)
        self.assertTrue(
            result.inversion_detected,
            "Vocabulary fallback must still detect inversion when objective is absent.",
        )

    def test_no_inversion_when_objective_absent_and_no_direction_term(self):
        """No false positives from vocabulary fallback on neutral utterances."""
        intent = _make_intent(order="ascending", direction=None)
        _, result = self._enrich("go to the door", intent)
        self.assertFalse(result.inversion_detected)


class TestObjectiveBasedHandleInjection(unittest.TestCase):
    """When selection_objective is set, handle injection uses objective.metric."""

    def setUp(self):
        self.verifier = IntentVerifier()

    def test_handle_injected_from_objective_metric(self):
        """selection_objective.metric drives the injected handle — not superlative scan."""
        intent = OperatorIntent(
            intent_type="task_instruction",
            task_type="go_to_object",
            grounding_query_plan={
                "object_type": "door",
                "operation": "select",
                "primitive_handle": "grounding.all_doors.ranked.manhattan.agent",
                "metric": "manhattan",
                "reference": "agent",
                "order": "descending",
                "ordinal": 1,
                "color": None,
                "exclude_colors": [],
                "distance_value": None,
                "tie_policy": "clarify",
                "answer_fields": ["target", "distance"],
                "required_capabilities": [],  # intentionally empty
                "preserved_constraints": [],
            },
            required_capabilities=[],  # empty — verifier must inject
            selection_objective=SelectionObjective(
                attribute="distance",
                direction="maximum",
                ordinal=1,
                metric="manhattan",
            ),
            confidence=0.9,
            reason="test",
        )
        enriched, result = self.verifier.enrich("go somewhere", intent)
        self.assertIn(
            "grounding.all_doors.ranked.manhattan.agent",
            enriched.required_capabilities,
            "Handle must be injected from selection_objective.metric.",
        )
        self.assertEqual(result.injected_handles, ["grounding.all_doors.ranked.manhattan.agent"])

    def test_no_duplicate_injection_when_handle_already_present(self):
        """Handle injection skips if handle already in required_capabilities."""
        handle = "grounding.all_doors.ranked.manhattan.agent"
        intent = OperatorIntent(
            intent_type="task_instruction",
            task_type="go_to_object",
            grounding_query_plan={
                "object_type": "door",
                "operation": "select",
                "primitive_handle": handle,
                "metric": "manhattan",
                "reference": "agent",
                "order": "descending",
                "ordinal": 1,
                "color": None,
                "exclude_colors": [],
                "distance_value": None,
                "tie_policy": "clarify",
                "answer_fields": ["target", "distance"],
                "required_capabilities": [handle],
                "preserved_constraints": [],
            },
            required_capabilities=[handle],
            selection_objective=SelectionObjective(
                attribute="distance",
                direction="maximum",
                ordinal=1,
                metric="manhattan",
            ),
            confidence=0.9,
            reason="test",
        )
        enriched, result = self.verifier.enrich("go to the farthest door", intent)
        self.assertEqual(
            enriched.required_capabilities.count(handle),
            1,
            "Handle must not be duplicated.",
        )


if __name__ == "__main__":
    unittest.main()
