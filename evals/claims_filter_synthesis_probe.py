#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from pprint import pprint
from unittest.mock import patch

import gymnasium as gym
import minigrid  # noqa: F401
from minigrid.wrappers import FullyObsWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jeenom.llm_compiler import SmokeTestCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.primitive_synthesizer import SynthesisResult
from jeenom.schemas import OperatorIntent


class ThresholdSynthesizer:
    def synthesize(
        self,
        handle,
        description,
        consumes,
        produces,
        existing_example=None,
        previous_code=None,
        validation_error=None,
    ):
        function_name = handle.replace(".", "_")
        code = (
            f"def {function_name}(entries, condition):\n"
            "    threshold = float(condition.get('threshold', 0))\n"
            "    comparison = condition.get('comparison', 'above')\n"
            "    if comparison == 'above':\n"
            "        return [e for e in entries if e.distance > threshold]\n"
            "    if comparison == 'at_least':\n"
            "        return [e for e in entries if e.distance >= threshold]\n"
            "    if comparison == 'below':\n"
            "        return [e for e in entries if e.distance < threshold]\n"
            "    if comparison in ('at_most', 'within'):\n"
            "        return [e for e in entries if e.distance <= threshold]\n"
            "    return []\n"
        )
        return SynthesisResult(
            handle=handle,
            function_name=function_name,
            code=code,
            status="success",
        )


def fake_build_env(env_id, render_mode):
    return FullyObsWrapper(gym.make(env_id))


def threshold_intent() -> OperatorIntent:
    return OperatorIntent(
        intent_type="unsupported",
        capability_status="synthesizable",
        confidence=1.0,
        reason="Probe arbitration gap without a prebuilt query plan.",
        required_capabilities=["claims.filter.threshold.manhattan"],
    )


def main() -> int:
    session = OperatorStationSession(
        compiler=SmokeTestCompiler(),
        compiler_name="smoke",
        env_id="MiniGrid-GoToDoor-16x16-v0",
        seed=8,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
        max_loops=512,
    )
    session.synthesizer = ThresholdSynthesizer()

    checks: dict[str, bool] = {}
    with patch("jeenom.run_demo.build_env", side_effect=fake_build_env):
        ranked = session.handle_utterance("rank all the doors by manhattan distance")
        checks["ranked_claims_written"] = session.active_claims is not None

        proposal = session.command_from_operator_intent(
            threshold_intent(),
            "go to the door where manhattan distance is above 7",
        )
        proposal_text = session.execute_command(proposal)
        checks["proposal_created"] = proposal.kind == "synthesis_proposal"
        checks["proposal_mentions_claims_handle"] = (
            "claims.filter.threshold.manhattan" in proposal_text
        )
        checks["proposal_preserves_condition"] = (
            session.pending_synthesis_proposal is not None
            and session.pending_synthesis_proposal.proposed_condition
            == {"threshold": 7.0, "comparison": "above", "metric": "manhattan"}
        )

        clarified = session.handle_utterance("yes")
        checks["claims_filter_registered"] = (
            session.capability_registry.lookup(
                "claims.filter.threshold.manhattan"
            ).implementation_status
            == "implemented"
        )
        checks["multiple_matches_clarified"] = (
            "DOORS WITH MANHATTAN DISTANCE ABOVE 7.0" in clarified
            and "blue door@(12,3)" in clarified
            and "red door@(10,7)" in clarified
        )
        checks["pending_candidate_created"] = session.pending_clarification is not None

        result = session.handle_utterance("red")
        checks["selected_target_ran"] = "RUN COMPLETE" in result
        checks["runtime_llm_zero"] = (
            session.last_result is not None
            and session.last_result["runtime_llm_calls_during_render"] == 0
        )
        checks["cache_miss_zero"] = (
            session.last_result is not None
            and session.last_result["cache_miss_during_render"] == 0
        )
        checks["final_skill_done"] = "final_skill_plan=['done']" in result

    print("CLAIMS FILTER SYNTHESIS PROBE\n")
    print("RANKED CLAIMS")
    print(ranked)
    print("\nSYNTHESIS PROPOSAL")
    print(proposal_text)
    print("\nFILTER RESPONSE")
    print(clarified)
    print("\nFINAL RUN")
    print(result)
    print("\nCHECKS")
    pprint(checks)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
