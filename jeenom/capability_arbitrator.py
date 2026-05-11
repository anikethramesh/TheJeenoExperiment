"""Phase 7.596 — Capability Arbitrator.

CapabilityArbitrator fires when the deterministic CapabilityMatcher detects a gap.
It reasons — using either a rule-based or LLM-backed backend — about what the station
should do when a required capability is missing or synthesizable.

No substrate imports. No runtime execution. Produces typed ArbitrationDecision only.

Architecture:
  CapabilityMatcher detects gap → CapabilityArbitrator.arbitrate(...)
  → ArbitrationDecision → station acts on decision

Decision types:
  refuse    — No path forward. Station returns a clear, honest operator message.
  clarify   — Operator can provide info that allows using existing capabilities.
  substitute — An existing capability is semantically equivalent (not a fallback).
  synthesize — Required capability is absent but marked synthesizable (Phase 7.7).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable

from .schemas import (
    ARBITRATION_DECISION_TYPES,
    ArbitrationDecision,
    SchemaValidationError,
)


def _condition_from_utterance(utterance: str) -> dict[str, Any] | None:
    """Best-effort fallback for non-LLM arbitration tests.

    The LLM arbitrator is expected to populate proposed_condition. This parser is
    only a guardrail for the deterministic SmokeTestArbitrator path.
    """
    text = " ".join(utterance.lower().strip().split())
    number_match = re.search(r"\b(\d+(?:\.\d+)?)\b", text)
    if not number_match:
        return None
    if any(term in text for term in ("above", "greater than", "more than", "over", "exceeds")):
        comparison = "above"
    elif any(term in text for term in ("at least", "minimum", "no less than")):
        comparison = "at_least"
    elif any(term in text for term in ("below", "less than", "under")):
        comparison = "below"
    elif any(term in text for term in ("at most", "within", "no more than")):
        comparison = "at_most"
    else:
        comparison = "above"
    metric = "euclidean" if "euclidean" in text else "manhattan" if "manhattan" in text else None
    return {
        "threshold": float(number_match.group(1)),
        "comparison": comparison,
        "metric": metric,
    }


def arbitration_decision_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "decision_type": {
                "type": "string",
                "enum": list(ARBITRATION_DECISION_TYPES),
            },
            "safe_to_execute": {"type": "boolean"},
            "reason": {"type": "string"},
            "suggested_handle": {"type": ["string", "null"]},
            "clarification_prompt": {"type": ["string", "null"]},
            "operator_message": {"type": "string"},
            "proposed_handle": {"type": ["string", "null"]},
            "proposed_description": {"type": ["string", "null"]},
            "proposed_condition": {
                "type": ["object", "null"],
                "properties": {
                    "threshold": {"type": ["number", "null"]},
                    "comparison": {
                        "type": ["string", "null"],
                        "enum": ["above", "below", "within", "at_least", "at_most", None],
                    },
                    "metric": {
                        "type": ["string", "null"],
                        "enum": ["manhattan", "euclidean", None],
                    },
                },
                "required": ["threshold", "comparison", "metric"],
                "additionalProperties": False,
            },
        },
        "required": [
            "decision_type",
            "safe_to_execute",
            "reason",
            "suggested_handle",
            "clarification_prompt",
            "operator_message",
            "proposed_handle",
            "proposed_description",
            "proposed_condition",
        ],
        "additionalProperties": False,
    }


def _parse_arbitration_decision(data: Any) -> ArbitrationDecision:
    if not isinstance(data, dict):
        raise SchemaValidationError("ArbitrationDecision response must be an object")
    decision_type = data.get("decision_type", "")
    if decision_type not in ARBITRATION_DECISION_TYPES:
        raise SchemaValidationError(
            "ArbitrationDecision.decision_type must be one of: "
            + ", ".join(ARBITRATION_DECISION_TYPES)
        )
    safe_to_execute = data.get("safe_to_execute", False)
    if not isinstance(safe_to_execute, bool):
        safe_to_execute = False
    # Enforce blueprint hard rule: refuse/synthesize must never be safe_to_execute
    if decision_type in {"refuse", "synthesize"} and safe_to_execute:
        safe_to_execute = False
    return ArbitrationDecision(
        decision_type=decision_type,
        safe_to_execute=safe_to_execute,
        reason=str(data.get("reason", "")),
        suggested_handle=data.get("suggested_handle"),
        clarification_prompt=data.get("clarification_prompt"),
        operator_message=str(data.get("operator_message", "")),
        proposed_handle=data.get("proposed_handle") or None,
        proposed_description=data.get("proposed_description") or None,
        proposed_condition=(
            dict(data["proposed_condition"])
            if isinstance(data.get("proposed_condition"), dict)
            else None
        ),
    )


class ArbitratorBackend(ABC):
    """Abstract arbitration backend. No substrate imports permitted."""

    @abstractmethod
    def arbitrate(
        self,
        utterance: str,
        intent_type: str,
        required_capabilities: list[str],
        missing_handles: list[str],
        synthesizable_handles: list[str],
        available_handles: list[str],
        scene_summary: dict[str, Any] | None = None,
        registry_synthesizable_handles: list[str] | None = None,
    ) -> ArbitrationDecision:
        raise NotImplementedError


class SmokeTestArbitrator(ArbitratorBackend):
    """Deterministic, rule-based arbitrator. No LLM calls.

    Always refuses with a clear gap description.
    Synthesizable capabilities produce a synthesize decision.
    """

    def arbitrate(
        self,
        utterance: str,
        intent_type: str,
        required_capabilities: list[str],
        missing_handles: list[str],
        synthesizable_handles: list[str],
        available_handles: list[str],
        scene_summary: dict[str, Any] | None = None,
        registry_synthesizable_handles: list[str] | None = None,
    ) -> ArbitrationDecision:
        if missing_handles:
            return ArbitrationDecision(
                decision_type="refuse",
                safe_to_execute=False,
                reason="Required capability handles are absent from the registry.",
                operator_message=(
                    "MISSING SKILLS: " + ", ".join(missing_handles) + "\n"
                    "I do not have these capabilities yet. "
                    "Use 'what can you do' to see what is available."
                ),
            )
        # Intent-specific synthesizable handles take priority; fall back to full
        # registry surface when the compiler emitted no specific required_capabilities.
        candidates = synthesizable_handles or registry_synthesizable_handles or []
        if candidates:
            return ArbitrationDecision(
                decision_type="synthesize",
                safe_to_execute=False,
                reason=(
                    "Required capability handles are synthesizable but "
                    "not yet implemented."
                ),
                operator_message=(
                    "That capability is synthesizable but not yet implemented: "
                    + ", ".join(candidates)
                    + ". Synthesis is not yet active (Phase 7.7)."
                ),
                proposed_handle=candidates[0],
                proposed_condition=_condition_from_utterance(utterance),
            )
        return ArbitrationDecision(
            decision_type="refuse",
            safe_to_execute=False,
            reason="Unsupported capability.",
            operator_message="I cannot safely execute that capability yet.",
        )


class LLMArbitrator(ArbitratorBackend):
    """LLM-backed arbitrator.

    Makes a compile-time reasoning call when a capability gap is detected.
    Falls back to SmokeTestArbitrator when the API key is unavailable or the
    call fails.

    Hard rules enforced after parsing (Blueprint Rule 9):
    - refuse/synthesize decisions are never safe_to_execute.
    - substitute is only allowed when the LLM provides a concrete suggested_handle.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        transport: Callable[[dict[str, Any]], Any] | None = None,
        fallback: SmokeTestArbitrator | None = None,
        timeout: int = 30,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.timeout = timeout
        self.transport = transport or self._chat_completions_transport
        self.fallback = fallback or SmokeTestArbitrator()
        self._fallback_reason: str | None = None
        if not self.api_key:
            self._fallback_reason = (
                "OPENROUTER_API_KEY not set; using smoke-test arbitrator."
            )

    def arbitrate(
        self,
        utterance: str,
        intent_type: str,
        required_capabilities: list[str],
        missing_handles: list[str],
        synthesizable_handles: list[str],
        available_handles: list[str],
        scene_summary: dict[str, Any] | None = None,
        registry_synthesizable_handles: list[str] | None = None,
    ) -> ArbitrationDecision:
        if self._fallback_reason:
            return self.fallback.arbitrate(
                utterance,
                intent_type,
                required_capabilities,
                missing_handles,
                synthesizable_handles,
                available_handles,
                scene_summary=scene_summary,
                registry_synthesizable_handles=registry_synthesizable_handles,
            )

        scene_api = (
            "SceneModel API available to synthesized functions:\n"
            "  scene.agent_x, scene.agent_y  — agent grid position (int)\n"
            "  scene.agent_dir               — agent facing direction (0=right,1=down,2=left,3=up)\n"
            "  scene.grid_width, scene.grid_height\n"
            "  scene.find(object_type, color, exclude_colors) → list[SceneObject]\n"
            "  scene.manhattan_distance_from_agent(obj) → int\n"
            "  SceneObject: .object_type, .color, .x, .y\n"
            "  math module is pre-injected (use math.sqrt etc. directly)\n"
            "  Standard Python builtins: abs, sorted, min, max, filter, enumerate, zip\n"
            "  NO imports allowed. NO environment, NO filesystem, NO randomness.\n"
            "  Signature: fn(scene, selector) → list[tuple[float, SceneObject]]\n"
            "  selector is a dict with keys: object_type, color, exclude_colors, relation, etc.\n"
            "\n"
            "Claims-filter API available when scene_summary.active_claims is present:\n"
            "  entries = active_claims.ranked_doors — typed GroundedDoorEntry list\n"
            "  GroundedDoorEntry: .color, .object_type, .x, .y, .distance, .metric\n"
            "  condition is a dict with keys: threshold, comparison, metric\n"
            "  Signature: fn(entries, condition) → list[GroundedDoorEntry]\n"
            "  Claims filters must preserve entry objects and order; they never access scene/env.\n"
        )
        system_prompt = (
            "You are the JEENOM capability arbitrator. The operator gave an instruction "
            "that requires capabilities not currently available in the registry. "
            "Your task is to reason about what the station should do next.\n\n"
            "The payload includes registry_synthesizable_handles — ALL primitives in the "
            "registry that are marked safe to synthesize, regardless of what the compiler "
            "declared. When intent_type=unsupported or synthesizable_handles is empty, "
            "check registry_synthesizable_handles first: if the utterance maps to any of "
            "those handles, choose synthesize and set proposed_handle to the matching handle. "
            "Do not refuse a request that a registry_synthesizable_handle could satisfy.\n\n"
            + scene_api + "\n"
            "Decision types:\n"
            "  'synthesize' — The operator's request can be expressed as a NEW pure Python\n"
            "                 grounding function using ONLY the SceneModel API above, OR as a\n"
            "                 pure claims-filter over active_claims.ranked_doors when those\n"
            "                 claims are available in scene_summary.\n"
            "                 Use this when:\n"
            "                   - The request involves a spatial/mathematical computation\n"
            "                     over visible objects (distance metrics, conditionals,\n"
            "                     relative direction, thresholds, inter-object distances)\n"
            "                   - The computation is deterministic and pure (no I/O)\n"
            "                   - It can be expressed as fn(scene, selector) → ranked list,\n"
            "                     or for claims filters as fn(entries, condition) → filtered list\n"
            "                 When synthesize:\n"
            "                   - proposed_handle: a dotted handle name like\n"
            "                     'grounding.closest_door.euclidean.agent' or\n"
            "                     'grounding.conditional_target.distance_threshold'\n"
            "                   - proposed_description: one sentence describing what the\n"
            "                     function computes (used as the synthesis prompt)\n"
            "                   - proposed_condition: for claims.filter.threshold.* handles,\n"
            "                     include {'threshold': number, 'comparison': one of above/below/"
            "within/at_least/at_most, 'metric': manhattan|euclidean}; otherwise null\n"
            "                   - safe_to_execute=false always\n"
            "                   - suggested_handle=null\n"
            "                 Examples:\n"
            "                   - 'euclidean distance to closest door'\n"
            "                     → proposed_handle='grounding.closest_door.euclidean.agent'\n"
            "                   - 'go to the door whose euclidean distance is above 6'\n"
            "                     with active ranked claims present\n"
            "                     → proposed_handle='claims.filter.threshold.euclidean'\n"
            "                   - 'door to my left'\n"
            "                     → proposed_handle='grounding.closest_door.relative_direction.agent'\n"
            "                   - 'door closest to the blue box'\n"
            "                     → proposed_handle='grounding.closest_door.object_reference.scene'\n"
            "  'clarify'    — An available capability could partially address the intent,\n"
            "                 OR the request is ambiguous and rephrasing would help.\n"
            "                 Emit a clarification_prompt naming what IS available.\n"
            "                 When clarifying with grounding.all_doors.ranked.manhattan.agent,\n"
            "                 set suggested_handle to that exact handle.\n"
            "                 Examples:\n"
            "                   - 'farthest door' (no farthest primitive, but ranked list exists)\n"
            "                   - 'second closest' (only rank-0 is implemented)\n"
            "  'substitute' — An available capability is semantically equivalent without loss.\n"
            "                 safe_to_execute=true only when truly equivalent.\n"
            "                 Emit suggested_handle from available_handles.\n"
            "  'refuse'     — The request cannot be expressed as a pure grounding function\n"
            "                 AND no available capability partially addresses it.\n"
            "                 Use for: pickup, toggle, unlock, navigation commands,\n"
            "                 anything requiring env interaction or side effects.\n\n"
            "HARD RULES — Blueprint Rule 9:\n"
            "  - NEVER substitute 'closest' for 'farthest' — intent inversion.\n"
            "  - NEVER substitute 'closest_door' for ranked/all — degradation.\n"
            "  - NEVER set safe_to_execute=true for refuse or synthesize.\n"
            "  - If the request is a pure spatial/logical computation over scene objects,\n"
            "    choose synthesize. Do NOT deflect to clarify for computable requests.\n"
            "  - proposed_handle and proposed_description must be set when synthesize.\n"
            "  - The station may actuate a robot — unintended motion has consequences.\n"
            "  - CRITICAL — Conditional/threshold filters over existing ranked claims use claims primitives:\n"
            "    If the operator's utterance contains ANY conditional constraint on distance\n"
            "    or position (e.g., 'above X', 'below Y', 'within N', 'more than X',\n"
            "    'less than Y', 'at least', 'at most', 'farther than', 'closer than',\n"
            "    'greater than', 'exceeds') AND scene_summary.active_claims.ranked_doors\n"
            "    is present, choose synthesize for a claims-filter primitive — even when\n"
            "    a related distance primitive is already implemented.\n"
            "    The threshold changes the semantics: euclidean-distance ≠ euclidean-above-10.\n"
            "    Example: 'door with euclidean distance above 10'\n"
            "      → proposed_handle='claims.filter.threshold.euclidean'\n"
            "      → proposed_description='Filter active ranked door claims by a parametric\n"
            "        Euclidean distance threshold carried in condition[threshold].'\n"
            "      → proposed_condition={'threshold': 10, 'comparison': 'above', 'metric': 'euclidean'}\n"
            "    Example: 'door with manhattan distance below 5'\n"
            "      → proposed_handle='claims.filter.threshold.manhattan'\n"
            "      → proposed_condition={'threshold': 5, 'comparison': 'below', 'metric': 'manhattan'}\n"
            "  - CRITICAL — Relative direction requires synthesize:\n"
            "    'door to my left/right/behind' → new primitive, not substitute.\n"
        )
        user_payload: dict[str, Any] = {
            "utterance": utterance,
            "intent_type": intent_type,
            "required_capabilities": required_capabilities,
            "missing_handles": missing_handles,
            "synthesizable_handles": synthesizable_handles,
            "available_handles": available_handles[:30],
            "registry_synthesizable_handles": registry_synthesizable_handles or [],
        }
        if scene_summary:
            user_payload["scene_summary"] = scene_summary
        request_payload = {
            "method_name": "compile_arbitration",
            "system_prompt": system_prompt,
            "user_payload": user_payload,
            "schema_name": "jeenom_arbitration_decision",
            "schema": arbitration_decision_json_schema(),
            "max_tokens": 512,
        }

        try:
            raw_data = self.transport(request_payload)
            return _parse_arbitration_decision(raw_data)
        except Exception:  # noqa: BLE001
            return self.fallback.arbitrate(
                utterance,
                intent_type,
                required_capabilities,
                missing_handles,
                synthesizable_handles,
                available_handles,
                scene_summary=scene_summary,
                registry_synthesizable_handles=registry_synthesizable_handles,
            )

    def _chat_completions_transport(self, request_payload: dict[str, Any]) -> Any:
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request_payload["system_prompt"]},
                {
                    "role": "user",
                    "content": json.dumps(request_payload["user_payload"], indent=2),
                },
            ],
            "max_tokens": request_payload["max_tokens"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request_payload["schema_name"],
                    "schema": request_payload["schema"],
                    "strict": True,
                },
            },
        }

        req = urllib.request.Request(
            url="https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"OpenRouter HTTP error {exc.code}: {details}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"OpenRouter network error: {exc}") from exc

        message = raw_response["choices"][0]["message"]
        refusal = message.get("refusal")
        if refusal:
            raise RuntimeError(f"Model refusal: {refusal}")

        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError(
                "Expected string JSON content from OpenRouter response"
            )

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"Model did not return valid JSON: {exc}"
            ) from exc


def build_arbitrator(
    compiler_name: str,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> ArbitratorBackend:
    if compiler_name == "llm":
        return LLMArbitrator(api_key=api_key, model=model)
    return SmokeTestArbitrator()


default_arbitrator: ArbitratorBackend = SmokeTestArbitrator()
