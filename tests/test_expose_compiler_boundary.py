"""Expose Problem 3: architecture gates are never tested against real LLM output.

All existing evals use SmokeTestCompiler, which constructs OperatorIntent objects
directly. A real LLM goes through LLMCompiler, which calls:
    transport(payload) → raw dict → OperatorIntent.from_dict(raw_dict) → gates

These tests use LLMCompiler(transport=adversarial_transport) to exercise that
real pipeline against three failure modes:

  A. Semantic inversion: LLM returns valid schema but with relation="closest"
     when the utterance said "farthest". IntentVerifier should catch this.

  B. Hallucinated handle: LLM returns valid schema with a made-up primitive.
     ReadinessGraph should block with missing_skills.

  C. Silent fallback: LLM returns a dict that fails OperatorIntent.from_dict().
     LLMCompiler catches the exception and silently falls back to SmokeTestCompiler.
     The test proves this fallback is observable via call_history.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import LLMCompiler, SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession


def _build_env(env_id: str, render_mode: str):
    return FullyObsWrapper(gym.make(env_id))


def _make_session(compiler, **kwargs) -> OperatorStationSession:
    defaults = dict(
        compiler=compiler,
        compiler_name="test",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    defaults.update(kwargs)
    return OperatorStationSession(**defaults)


def _utterance_from(payload: dict[str, Any]) -> str:
    return payload.get("user_payload", {}).get("utterance", "")


# ── Transport A: semantic inversion ──────────────────────────────────────────

def _transport_semantic_inversion(payload: dict[str, Any]) -> dict[str, Any]:
    """Returns valid OperatorIntent JSON but silently degrades 'farthest' → 'closest'.

    Uses the correct grounding_query_plan structure (matching what SmokeTestCompiler
    produces) but with order='ascending' (closest-first) instead of 'descending'
    (farthest-first). This is a schema-valid, semantically wrong output.
    """
    utterance = _utterance_from(payload)
    if "farthest" in utterance.lower() or "furthest" in utterance.lower():
        return {
            "intent_type": "task_instruction",
            "task_type": "go_to_object",
            "canonical_instruction": utterance,
            "target": None,
            "target_selector": None,
            "clear_memory": False,
            # selection_objective correctly says "maximum" (farthest),
            # but grounding_query_plan.order is "ascending" (inverted).
            # IntentVerifier must detect this objective↔plan contradiction.
            "selection_objective": {
                "attribute": "distance",
                "direction": "maximum",   # correct: farthest = maximum
                "ordinal": 1,
                "metric": "manhattan",
            },
            "grounding_query_plan": {
                "object_type": "door",
                "operation": "select",
                "primitive_handle": "grounding.all_doors.ranked.manhattan.agent",
                "metric": "manhattan",
                "reference": "agent",
                "order": "ascending",    # WRONG: maximum-distance requires descending
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
                "preserved_constraints": ["farthest", "door", "manhattan"],
            },
            "required_capabilities": [
                "grounding.all_doors.ranked.manhattan.agent",
                "task.go_to_object.door",
            ],
            "capability_status": "executable",
            "confidence": 0.9,
            "reason": "LLM silently degraded farthest to closest",
        }
    # Pass-through for warmup / other utterances
    return {
        "intent_type": "status_query",
        "status_query": "scene",
        "confidence": 0.9,
        "reason": "",
    }


# ── Transport B: hallucinated primitive handle ────────────────────────────────

def _transport_hallucinated_handle(payload: dict[str, Any]) -> dict[str, Any]:
    """Returns valid OperatorIntent JSON with a hallucinated primitive_handle.

    The hallucinated handle is in grounding_query_plan.primitive_handle — the field
    the station uses to build the RequestPlan step's required_handle. ReadinessGraph
    checks that handle against the registry and must block with missing_skills.

    NOTE: putting a hallucinated handle only in required_capabilities is NOT enough —
    the station builds its own RequestPlan from structural fields and ignores the
    LLM's self-reported required_capabilities for routing decisions.

    IMPORTANT: tests use "go to the mystery door" because:
    - "go to the [color] door" matches the deterministic fast path (classify_utterance
      line 283) and bypasses the LLM compiler entirely.
    - "go to the closest door" is now a deterministic ranked-distance plan, so it
      correctly bypasses the LLM compiler too.
    - "go to the mystery door" is unresolved locally, so the LLM compiler is invoked.
    """
    utterance = _utterance_from(payload)
    if "mystery door" in utterance.lower() or "farthest" in utterance.lower():
        return {
            "intent_type": "task_instruction",
            "task_type": "go_to_object",
            "canonical_instruction": utterance,
            "target": None,
            "target_selector": None,
            "clear_memory": False,
            "grounding_query_plan": {
                "object_type": "door",
                "operation": "select",
                "primitive_handle": "grounding.hallucinated_primitive.does_not_exist",
                "metric": "manhattan",
                "reference": "agent",
                "order": "ascending",
                "ordinal": None,
                "color": None,
                "exclude_colors": [],
                "distance_value": None,
                "tie_policy": "clarify",
                "answer_fields": ["target", "distance"],
                "required_capabilities": [
                    "grounding.hallucinated_primitive.does_not_exist",
                    "task.go_to_object.door",
                ],
                "preserved_constraints": ["mystery", "door"],
            },
            "required_capabilities": [
                "grounding.hallucinated_primitive.does_not_exist",
                "task.go_to_object.door",
            ],
            "capability_status": "executable",
            "confidence": 0.9,
            "reason": "",
        }
    return {
        "intent_type": "status_query",
        "status_query": "scene",
        "confidence": 0.9,
        "reason": "",
    }


# ── Transport C: schema failure → silent fallback ─────────────────────────────

def _transport_invalid_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """Returns a dict that fails OperatorIntent.from_dict() — invalid intent_type."""
    return {
        "intent_type": "totally_invalid_type_that_does_not_exist",
        "confidence": 0.9,
        "reason": "",
    }


def _llm_compiler(transport) -> LLMCompiler:
    # api_key must be set to something non-empty so LLMCompiler doesn't immediately
    # set fallback_reason and bypass the transport entirely (llm_compiler.py:1401-1403).
    return LLMCompiler(api_key="test-key", transport=transport)


class TestSemanticInversionIsBlocked(unittest.TestCase):
    """Farthest→closest degradation from LLM must not silently execute."""

    def setUp(self):
        self.patcher = patch("jeenom.run_demo.build_env", side_effect=_build_env)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_inversion_goes_through_from_dict_before_gates(self):
        """Semantic inversion arrives via OperatorIntent.from_dict(), not direct construction.

        This is the key difference from the previous test approach (which subclassed
        SmokeTestCompiler and constructed OperatorIntent directly). With LLMCompiler
        + transport, the raw dict is validated by from_dict() first.

        The test verifies the inversion dict is structurally valid (passes from_dict)
        but semantically wrong — proving the schema layer alone cannot catch this.
        """
        from jeenom.schemas import OperatorIntent
        inverted = {
            "intent_type": "task_instruction",
            "task_type": "go_to_object",
            "target": None,
            "target_selector": None,
            "clear_memory": False,
            # selection_objective correctly says "maximum" — the LLM distilled
            # the intent right. But the plan says "ascending". Schema can't catch this.
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
                "order": "ascending",    # WRONG: maximum requires descending
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
                "preserved_constraints": ["farthest", "door", "manhattan"],
            },
            "required_capabilities": [
                "grounding.all_doors.ranked.manhattan.agent",
                "task.go_to_object.door",
            ],
            "capability_status": "executable",
            "confidence": 0.9,
            "reason": "",
        }
        # This must NOT raise — the inversion is schema-valid. Gates must catch it.
        intent = OperatorIntent.from_dict(inverted)
        self.assertEqual(intent.intent_type, "task_instruction")
        self.assertEqual(intent.grounding_query_plan["order"], "ascending")
        # selection_objective.direction == "maximum" but plan.order == "ascending":
        # only IntentVerifier's objective-based check can catch this contradiction.

    def test_inversion_does_not_execute_as_closest(self):
        """LLM-compiled inversion (farthest→closest) must not result in RUN COMPLETE.

        If this FAILS it means IntentVerifier did not block the inverted intent —
        the station executed navigation to the closest door when the operator asked
        for the farthest (Blueprint Rule 9 violation).

        The warmup uses SmokeTestCompiler to populate grounding claims before
        switching to LLMCompiler. Without claims the test would pass vacuously
        (no claims → _compose_grounding_query_plan returns an error, not RUN COMPLETE)
        while the actual inversion slips through validation undetected.
        """
        session = _make_session(_llm_compiler(_transport_semantic_inversion))
        # Populate grounding claims via SmokeTestCompiler so the inversion
        # actually reaches _compose_grounding_query_plan with real scene data.
        session.compiler = SmokeTestCompiler()
        session.handle_utterance("rank all the doors by manhattan distance")
        session.compiler = _llm_compiler(_transport_semantic_inversion)

        response = session.handle_utterance("go to the farthest door")

        self.assertNotIn(
            "RUN COMPLETE",
            response,
            f"Station silently executed farthest→closest inversion via LLM path.\n"
            f"ReadinessGraph status: {getattr(session.last_readiness_graph, 'graph_status', None)}\n"
            f"Response: {response[:400]}",
        )

    def test_inversion_produces_gate_block_not_silent_acceptance(self):
        """After LLM delivers inverted intent, ReadinessGraph must not be 'executable'.

        Acceptable outcomes: needs_clarification, missing_skills, synthesizable.
        Unacceptable: graph_status='executable' (inversion silently accepted).
        """
        session = _make_session(_llm_compiler(_transport_semantic_inversion))
        session.compiler = SmokeTestCompiler()
        session.handle_utterance("rank all the doors by manhattan distance")
        session.compiler = _llm_compiler(_transport_semantic_inversion)
        session.handle_utterance("go to the farthest door")

        graph_status = getattr(session.last_readiness_graph, "graph_status", None)
        self.assertNotEqual(
            graph_status,
            "executable",
            f"ReadinessGraph accepted semantically inverted LLM output as executable.\n"
            f"graph_status={graph_status}",
        )


class TestHallucinatedHandleIsBlocked(unittest.TestCase):
    """A hallucinated primitive handle from the LLM must be blocked by ReadinessGraph."""

    def setUp(self):
        self.patcher = patch("jeenom.run_demo.build_env", side_effect=_build_env)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_hallucinated_handle_does_not_execute(self):
        """ReadinessGraph blocks an unregistered primitive_handle from LLM output.

        The hallucinated handle must be in grounding_query_plan.primitive_handle —
        that is what the station uses to build the RequestPlan step's required_handle.
        A hallucinated handle only in required_capabilities is silently ignored
        because the station constructs RequestPlan from structural intent fields.

        "go to the mystery door" is used instead of "go to the red door" because
        the latter matches the deterministic fast path and never calls the LLM compiler.
        """
        session = _make_session(_llm_compiler(_transport_hallucinated_handle))
        session.handle_utterance("what do you see")
        response = session.handle_utterance("go to the mystery door")

        self.assertNotIn(
            "RUN COMPLETE",
            response,
            f"Station executed task with hallucinated primitive handle from LLM.\n"
            f"Response: {response[:400]}",
        )

    def test_hallucinated_handle_sets_missing_skills_status(self):
        """ReadinessGraph must report missing_skills (or similar) for unknown handle."""
        session = _make_session(_llm_compiler(_transport_hallucinated_handle))
        session.handle_utterance("what do you see")
        session.handle_utterance("go to the mystery door")

        graph_status = getattr(session.last_readiness_graph, "graph_status", None)
        self.assertIn(
            graph_status,
            {"missing_skills", "unsupported", "needs_clarification"},
            f"ReadinessGraph did not block hallucinated handle from LLM.\n"
            f"graph_status={graph_status}",
        )


class TestSilentFallbackIsObservable(unittest.TestCase):
    """When LLM output fails schema validation, fallback must be observable.

    LLMCompiler._compile_or_fallback catches ALL exceptions at line 1883 and
    silently calls SmokeTestCompiler. The operator gets a response but doesn't
    know the LLM failed. This test proves the fallback happened via call_history.
    """

    def setUp(self):
        self.patcher = patch("jeenom.run_demo.build_env", side_effect=_build_env)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_schema_failure_does_not_raise(self):
        """Station must not crash when LLM returns schema-invalid output."""
        compiler = _llm_compiler(_transport_invalid_schema)
        session = _make_session(compiler)
        try:
            response = session.handle_utterance("go to the red door")
        except Exception as exc:
            self.fail(f"Station raised exception on schema-invalid LLM output: {exc}")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response.strip()), 0)

    def test_schema_failure_is_recorded_in_call_history(self):
        """Fallback after schema failure must appear in compiler call_history.

        LLMCompiler.record_call() is called with used_fallback=True when the
        exception path fires (llm_compiler.py:1886-1893). This test verifies
        the fallback is at least observable by inspecting the compiler's own log.

        If this FAILS, the fallback is truly silent — no record of LLM failure exists.
        """
        compiler = _llm_compiler(_transport_invalid_schema)
        session = _make_session(compiler)
        session.handle_utterance("go to the red door")

        fallback_calls = [c for c in compiler.call_history if c.get("used_fallback")]
        self.assertGreater(
            len(fallback_calls),
            0,
            f"No fallback recorded in call_history after schema-invalid LLM output.\n"
            f"call_history: {compiler.call_history}",
        )

    def test_fallback_reason_is_logged(self):
        """Fallback reason (the exception message) must appear in compiler logs."""
        compiler = _llm_compiler(_transport_invalid_schema)
        session = _make_session(compiler)
        session.handle_utterance("go to the red door")

        fallback_log_entries = [
            entry for entry in compiler.logs
            if "falling back" in entry.lower() or "fallback" in entry.lower()
        ]
        self.assertGreater(
            len(fallback_log_entries),
            0,
            f"No fallback log entry after schema-invalid LLM output.\n"
            f"compiler.logs: {compiler.logs}",
        )


if __name__ == "__main__":
    unittest.main()
