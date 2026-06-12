"""Phase 8.27 probe: LLM Compiler advertises semantic normalizer vocabulary in its prompt."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jeenom.llm_compiler import LLMCompiler
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import OperatorIntent

def main() -> int:
    metrics: dict[str, bool] = {}
    
    captured_payloads = []
    def fake_transport(payload):
        captured_payloads.append(payload)
        return {
            "intent_type": "status_query",
            "status_query": "scene",
            "required_capabilities": []
        }
    
    compiler = LLMCompiler(api_key="fake", transport=fake_transport)
    session = OperatorStationSession(
        compiler=compiler,
        compiler_name="llm",
        env_id="MiniGrid-GoToDoor-8x8-v0",
        seed=42,
        render_mode="none",
        memory_root=Path(tempfile.mkdtemp()),
    )
    
    # We want to intercept compile_operator_intent payload.
    session.handle_utterance("go to the third farthest door")
    
    if not captured_payloads:
        print("No payloads captured from LLMCompiler", file=sys.stderr)
        return 1

    prompt = captured_payloads[0]["system_prompt"]
    
    # The normalizer vocabulary should be advertised in the prompt
    # Check if we exported the constraints into the prompt
    metrics["advertises_ordinals"] = "first" in prompt and "any numeric ordinal like 11th" in prompt and "second" in prompt
    metrics["advertises_descending_terms"] = "highest" in prompt and "largest" in prompt
    metrics["advertises_ascending_terms"] = "lowest" in prompt and "smallest" in prompt
    
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if all(metrics.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
