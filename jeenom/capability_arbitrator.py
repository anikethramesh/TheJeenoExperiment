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
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable

from .schemas import (
    ARBITRATION_DECISION_TYPES,
    ArbitrationDecision,
    SchemaValidationError,
)


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
        },
        "required": [
            "decision_type",
            "safe_to_execute",
            "reason",
            "suggested_handle",
            "clarification_prompt",
            "operator_message",
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
        if synthesizable_handles:
            return ArbitrationDecision(
                decision_type="synthesize",
                safe_to_execute=False,
                reason=(
                    "Required capability handles are synthesizable but "
                    "not yet implemented."
                ),
                operator_message=(
                    "That capability is synthesizable but not yet implemented: "
                    + ", ".join(synthesizable_handles)
                    + ". Synthesis is not yet active (Phase 7.7)."
                ),
            )
        return ArbitrationDecision(
            decision_type="refuse",
            safe_to_execute=False,
            reason="Unsupported capability.",
            operator_message="That capability is not supported.",
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
            )

        system_prompt = (
            "You are the JEENOM capability arbitrator. The operator gave an instruction "
            "that requires capabilities not currently available in the registry. "
            "Your task is to reason about what the station should do next.\n\n"
            "Decision types:\n"
            "  'clarify'    — USE THIS when an available capability could partially or\n"
            "                 indirectly address the intent, or when rephrasing would\n"
            "                 let the station help. Emit a clarification_prompt that\n"
            "                 honestly names what IS available and offers it.\n"
            "                 When clarifying with grounding.all_doors.ranked.manhattan.agent,\n"
            "                 set suggested_handle to that exact handle.\n"
            "                 Examples where clarify is correct:\n"
            "                   - 'farthest door': clarify with 'I can list all visible\n"
            "                     doors ranked by distance — would that help?'\n"
            "                     Set suggested_handle='grounding.all_doors.ranked.manhattan.agent'\n"
            "                   - 'distance of all doors': clarify with 'I can show you\n"
            "                     all visible doors ranked by Manhattan distance.'\n"
            "                     Set suggested_handle='grounding.all_doors.ranked.manhattan.agent'\n"
            "                   - 'second closest door': clarify with 'I can go to the\n"
            "                     closest door. Do you want that instead?'\n"
            "  'refuse'     — Use ONLY when no available capability could partially\n"
            "                 address the intent (e.g. pickup, toggle, unlock).\n"
            "                 Emit an honest operator_message explaining the gap.\n"
            "  'substitute' — An available capability is semantically equivalent\n"
            "                 (not a degraded fallback). safe_to_execute=true ONLY when\n"
            "                 the substitute truly satisfies the intent without loss.\n"
            "                 Emit suggested_handle from available_handles_sample.\n"
            "  'synthesize' — Required capability is absent but synthesizable.\n"
            "                 Must have safe_to_execute=false.\n\n"
            "HARD RULES — Blueprint Rule 9:\n"
            "  - NEVER substitute 'closest' for 'farthest' — that is intent inversion.\n"
            "  - NEVER substitute 'closest_door' for ranked/all — that is degradation.\n"
            "  - NEVER set safe_to_execute=true for refuse or synthesize.\n"
            "  - Prefer clarify over refuse whenever scene data or an available\n"
            "    primitive could give the operator useful partial information.\n"
            "  - The station may actuate a robot — unintended motion has consequences.\n"
        )
        user_payload: dict[str, Any] = {
            "utterance": utterance,
            "intent_type": intent_type,
            "required_capabilities": required_capabilities,
            "missing_handles": missing_handles,
            "synthesizable_handles": synthesizable_handles,
            "available_handles": available_handles[:30],
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
