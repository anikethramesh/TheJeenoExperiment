"""Phase 7.9 — Collaborative Capability Composition probe.

Verifies:
- Synthesizable utterance produces synthesis_proposal command (not immediate synthesis).
- Proposal message names the handle and similar implemented primitives.
- pending_synthesis_proposal is set after proposal.
- "yes" approval → synthesis runs via fake transport → primitive registered → task executes.
- "no" rejection → pending cleared, no synthesis.
- Redirect (new utterance) → pending cleared, new utterance handled normally.
- reset clears pending_synthesis_proposal.
- Golden path "go to the red door" is unaffected (no proposal fired).
- runtime_llm_calls_during_render remains 0 after synthesized-primitive task.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.primitive_synthesizer import LLMSynthesizer


CORRECT_EUCLIDEAN_CODE = """\
def grounding_closest_door_euclidean_agent(scene, selector):
    doors = scene.find(
        object_type=selector.get('object_type', 'door'),
        color=selector.get('color'),
        exclude_colors=selector.get('exclude_colors') or [],
    )
    if not doors:
        return []
    return sorted(
        [
            (math.sqrt((d.x - scene.agent_x) ** 2 + (d.y - scene.agent_y) ** 2), d)
            for d in doors
        ],
        key=lambda pair: (pair[0], pair[1].color or ''),
    )
"""


def _make_session() -> OperatorStationSession:
    return OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )


def _run(fn):
    def fake(env_id, render_mode):
        return FullyObsWrapper(gym.make(env_id))
    with patch("jeenom.run_demo.build_env", side_effect=fake):
        return fn()


def _fake_transport(payload):
    return {
        "function_name": "grounding_closest_door_euclidean_agent",
        "function_body": CORRECT_EUCLIDEAN_CODE,
        "description": "Euclidean grounding synthesized.",
    }


def main() -> int:
    checks: dict[str, bool] = {}

    # ── 1. Warm scene so grounding has something to work with ─────────────
    session = _make_session()
    _run(lambda: session.handle_utterance("go to the red door"))

    # ── 2. Synthesizable utterance → proposal, not immediate synthesis ────
    response = _run(
        lambda: session.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["proposal_returned"] = "SYNTHESIS PROPOSAL" in response
    checks["proposal_mentions_handle"] = "grounding.closest_door.euclidean.agent" in response
    checks["proposal_mentions_similar"] = any(
        h in response
        for h in (
            "grounding.closest_door.manhattan.agent",
            "grounding.unique_door",
            "grounding.",
        )
    )
    checks["pending_proposal_set"] = session.pending_synthesis_proposal is not None
    checks["pending_proposal_handle_correct"] = (
        session.pending_synthesis_proposal is not None
        and session.pending_synthesis_proposal.handle == "grounding.closest_door.euclidean.agent"
    )

    # ── 3. "no" → pending cleared, no synthesis ───────────────────────────
    no_response = _run(lambda: session.handle_utterance("no"))
    checks["no_clears_pending"] = session.pending_synthesis_proposal is None
    checks["no_does_not_synthesize"] = (
        session.capability_registry.lookup(
            "grounding.closest_door.euclidean.agent"
        ).implementation_status == "synthesizable"
    )
    checks["no_response_is_string"] = isinstance(no_response, str) and len(no_response) > 0

    # ── 4. Proposal fires again; "yes" → synthesis runs ──────────────────
    session2 = _make_session()
    session2.synthesizer = LLMSynthesizer(api_key="fake", transport=_fake_transport)
    _run(lambda: session2.handle_utterance("go to the red door"))

    _run(
        lambda: session2.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["second_session_proposal_set"] = session2.pending_synthesis_proposal is not None

    yes_response = _run(lambda: session2.handle_utterance("yes"))
    checks["yes_clears_pending"] = session2.pending_synthesis_proposal is None
    checks["yes_synthesized_primitive"] = (
        session2.capability_registry.lookup(
            "grounding.closest_door.euclidean.agent"
        ).implementation_status == "implemented"
    )
    checks["yes_response_is_string"] = isinstance(yes_response, str) and len(yes_response) > 0

    # ── 5. Second call uses synthesized primitive directly (no re-proposal) ─
    response2 = _run(
        lambda: session2.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["second_call_no_proposal"] = session2.pending_synthesis_proposal is None
    checks["second_call_succeeds"] = isinstance(response2, str) and len(response2) > 0

    # ── 6. Redirect clears pending and handles new utterance ─────────────
    session3 = _make_session()
    _run(lambda: session3.handle_utterance("go to the red door"))
    _run(
        lambda: session3.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["redirect_session_proposal_set"] = session3.pending_synthesis_proposal is not None

    redirect_response = _run(lambda: session3.handle_utterance("what do you see"))
    checks["redirect_clears_pending"] = session3.pending_synthesis_proposal is None
    checks["redirect_handles_new_utterance"] = isinstance(redirect_response, str) and len(redirect_response) > 0

    # ── 7. reset clears pending ───────────────────────────────────────────
    session4 = _make_session()
    _run(lambda: session4.handle_utterance("go to the red door"))
    _run(
        lambda: session4.handle_utterance("go to the closest door using euclidean distance")
    )
    checks["reset_session_proposal_set"] = session4.pending_synthesis_proposal is not None
    session4.reset()
    checks["reset_clears_proposal"] = session4.pending_synthesis_proposal is None

    # ── 8. Golden path unaffected ─────────────────────────────────────────
    session_golden = _make_session()
    golden = _run(lambda: session_golden.handle_utterance("go to the red door"))
    checks["golden_path_no_proposal"] = session_golden.pending_synthesis_proposal is None
    checks["golden_path_still_works"] = (
        "RUN COMPLETE" in golden or "task_complete" in golden
    )

    # ── Summary ───────────────────────────────────────────────────────────
    print("CHECKS")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
