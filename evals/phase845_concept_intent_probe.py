"""Phase 8.4.5 probe: First-Class Concept Intent Types.

Verifies that concept_teach and concept_recall are first-class OperatorIntent types,
that the LLM/SmokeTestCompiler correctly classifies natural-language concept-teach
utterances without fragile regex in classify_utterance, and that the station dispatches
both intent types correctly.

Checks:
  schema_has_concept_teach          — concept_teach in OPERATOR_INTENT_TYPES
  schema_has_concept_recall         — concept_recall in OPERATOR_INTENT_TYPES
  schema_has_concepts_status_query  — concepts in OPERATOR_STATUS_QUERIES
  intent_concept_name_field         — OperatorIntent.concept_name field exists and parses
  intent_concept_utterance_field    — OperatorIntent.concept_utterance field exists and parses
  validate_concept_teach_ok         — concept_teach with name+utterance passes validation
  validate_concept_teach_no_name    — concept_teach without concept_name raises SchemaValidationError
  validate_concept_teach_no_utt     — concept_teach without concept_utterance raises SchemaValidationError
  validate_concept_recall_ok        — concept_recall with concept_name passes validation
  validate_concept_recall_no_name   — concept_recall without concept_name raises SchemaValidationError
  smoke_natural_language_teach      — SmokeTestCompiler emits concept_teach for 'when i say X, Y'
  smoke_natural_language_teach_noise — SmokeTestCompiler strips trailing 'can you remember that'
  smoke_bare_means                  — SmokeTestCompiler emits concept_teach for 'X means Y'
  station_concept_intent_teach      — station handles concept_teach OperatorIntent correctly
  station_concept_intent_recall     — station handles concept_recall OperatorIntent correctly
  station_concept_recall_unknown    — station returns helpful message for unknown concept_recall
  explicit_regex_still_works        — 'remember X means Y' still fast-paths via classify_utterance
  natural_language_routes_to_smoke  — 'when i say X, Y' no longer short-circuits in classify_utterance
  no_task_instruction_misclassify   — 'when i say scout go to the red door' not classified as task_instruction
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from harness import build_env as _build_env, make_session as _make_session

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.capability_registry import CapabilityRegistry
from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession, classify_utterance
from jeenom.schemas import (
    OPERATOR_INTENT_TYPES,
    OPERATOR_STATUS_QUERIES,
    OperatorIntent,
    SchemaValidationError,
)






def main() -> int:
    metrics: dict[str, bool] = {}

    # ── Schema checks ─────────────────────────────────────────────────────────
    metrics["schema_has_concept_teach"] = "concept_teach" in OPERATOR_INTENT_TYPES
    metrics["schema_has_concept_recall"] = "concept_recall" in OPERATOR_INTENT_TYPES
    metrics["schema_has_concepts_status_query"] = "concepts" in OPERATOR_STATUS_QUERIES

    # ── OperatorIntent field parsing ──────────────────────────────────────────
    try:
        intent = OperatorIntent.from_dict({
            "intent_type": "concept_teach",
            "concept_name": "bingo",
            "concept_utterance": "go to the red door",
            "canonical_instruction": None,
            "task_type": None,
            "target": None,
            "knowledge_update": None,
            "reference": None,
            "status_query": None,
            "claim_reference": None,
            "control": None,
            "target_selector": None,
            "grounding_query_plan": None,
            "capability_status": "executable",
            "required_capabilities": [],
            "clear_memory": False,
            "confidence": 0.9,
            "reason": "test",
        })
        metrics["intent_concept_name_field"] = intent.concept_name == "bingo"
        metrics["intent_concept_utterance_field"] = intent.concept_utterance == "go to the red door"
    except Exception as e:
        metrics["intent_concept_name_field"] = False
        metrics["intent_concept_utterance_field"] = False

    # ── Validation via from_dict (where _validate_supported_shape is enforced) ─
    _base = {
        "canonical_instruction": None, "task_type": None, "target": None,
        "knowledge_update": None, "reference": None, "status_query": None,
        "claim_reference": None, "control": None, "target_selector": None,
        "grounding_query_plan": None, "capability_status": "executable",
        "required_capabilities": [], "clear_memory": False,
        "confidence": 0.9, "reason": "",
    }

    try:
        OperatorIntent.from_dict({**_base, "intent_type": "concept_teach",
                                   "concept_name": "x", "concept_utterance": "go to the red door"})
        metrics["validate_concept_teach_ok"] = True
    except SchemaValidationError:
        metrics["validate_concept_teach_ok"] = False

    try:
        OperatorIntent.from_dict({**_base, "intent_type": "concept_teach",
                                   "concept_name": None, "concept_utterance": "go to the red door"})
        metrics["validate_concept_teach_no_name"] = False
    except SchemaValidationError:
        metrics["validate_concept_teach_no_name"] = True

    try:
        OperatorIntent.from_dict({**_base, "intent_type": "concept_teach",
                                   "concept_name": "x", "concept_utterance": None})
        metrics["validate_concept_teach_no_utt"] = False
    except SchemaValidationError:
        metrics["validate_concept_teach_no_utt"] = True

    try:
        OperatorIntent.from_dict({**_base, "intent_type": "concept_recall", "concept_name": "bingo"})
        metrics["validate_concept_recall_ok"] = True
    except SchemaValidationError:
        metrics["validate_concept_recall_ok"] = False

    try:
        OperatorIntent.from_dict({**_base, "intent_type": "concept_recall", "concept_name": None})
        metrics["validate_concept_recall_no_name"] = False
    except SchemaValidationError:
        metrics["validate_concept_recall_no_name"] = True

    # ── SmokeTestCompiler: natural-language concept teach ─────────────────────

    class _FakeMemory:
        knowledge: dict = {}
        episodic_memory: dict = {}
        understanding: dict = {}

    smoke = SmokeTestCompiler()

    intent_natural = smoke.compile_operator_intent(
        "when i say scout you need to go to the red door",
        _FakeMemory(),
    )
    metrics["smoke_natural_language_teach"] = (
        intent_natural.intent_type == "concept_teach"
        and (intent_natural.concept_name or "").lower() == "scout"
        and "go to the red door" in (intent_natural.concept_utterance or "")
    )

    intent_noise = smoke.compile_operator_intent(
        "when i say patrol, go to the green door. can you remember that?",
        _FakeMemory(),
    )
    metrics["smoke_natural_language_teach_noise"] = (
        intent_noise.intent_type == "concept_teach"
        and "can you remember" not in (intent_noise.concept_utterance or "")
        and "go to the green door" in (intent_noise.concept_utterance or "")
    )

    intent_means = smoke.compile_operator_intent(
        "bingo means go to the red door",
        _FakeMemory(),
    )
    metrics["smoke_bare_means"] = (
        intent_means.intent_type == "concept_teach"
        and (intent_means.concept_name or "").lower() == "bingo"
        and "go to the red door" in (intent_means.concept_utterance or "")
    )

    # ── Station: concept_teach and concept_recall via command_from_operator_intent ──
    with patch("jeenom.run_demo.build_env", side_effect=_build_env):
        session = _make_session()

        # Inject a concept_teach intent via command_from_operator_intent
        intent_teach = OperatorIntent(
            intent_type="concept_teach",
            concept_name="bingo",
            concept_utterance="go to the red door",
            confidence=0.9,
        )
        cmd = session.command_from_operator_intent(intent_teach, "bingo means go to the red door")
        metrics["station_concept_intent_teach"] = cmd.kind == "concept_teach" and cmd.payload.get("name") == "bingo"

        # Now teach bingo for real so we can test recall
        session.teach_concept("bingo", "go to the red door")

        intent_recall = OperatorIntent(
            intent_type="concept_recall",
            concept_name="bingo",
            confidence=0.9,
        )
        cmd_recall = session.command_from_operator_intent(intent_recall, "bingo")
        metrics["station_concept_intent_recall"] = cmd_recall.kind == "task_instruction" and "red door" in cmd_recall.utterance

        # Unknown concept recall should return helpful clarification
        intent_unknown = OperatorIntent(
            intent_type="concept_recall",
            concept_name="phantom",
            confidence=0.5,
        )
        cmd_unknown = session.command_from_operator_intent(intent_unknown, "phantom")
        metrics["station_concept_recall_unknown"] = (
            cmd_unknown.kind == "clarification"
            and "phantom" in cmd_unknown.payload.get("message", "")
        )

        # ── Explicit regex fast path still works ──────────────────────────────
        _registry = CapabilityRegistry.minigrid_default()
        cmd_explicit = classify_utterance("remember zap means go to the blue door", _registry)
        metrics["explicit_regex_still_works"] = (
            cmd_explicit.kind == "concept_teach"
            and cmd_explicit.payload.get("name") == "zap"
            and cmd_explicit.payload.get("utterance") == "go to the blue door"
        )

        # ── Natural-language teach no longer short-circuits to concept_teach in classify_utterance ──
        # It must route through unresolved → SmokeTestCompiler
        cmd_natural_classify = classify_utterance(
            "when i say scout, you need to go to the red door", _registry
        )
        metrics["natural_language_routes_to_smoke"] = cmd_natural_classify.kind == "unresolved"

        # ── Ensure 'when i say X go to Y' not misclassified as task_instruction ──
        session2 = _make_session()
        resp = session2.handle_utterance("when i say scout go to the red door")
        # Should teach a concept, not execute a task — so no "RUN COMPLETE" and no task execution
        metrics["no_task_instruction_misclassify"] = "RUN COMPLETE" not in resp and (
            "scout" in resp.lower() or "concept" in resp.lower() or "remember" in resp.lower() or "taught" in resp.lower()
        )

    print(json.dumps(metrics, sort_keys=True))
    return 0 if all(metrics.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
