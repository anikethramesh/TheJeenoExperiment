# JEENOM (Just-In-Time Embodied Execution Networked Orchestration Model)

JEENOM is an advanced, agentic capability architecture currently being built and tested within the [MiniGrid](https://minigrid.farama.org/) gridworld environment. It translates natural language instructions into validated, executable intents through an LLM compiler, grounding requests against a persistent scene model and a rigid capability registry. 

## Features

- **Natural Language Operator Station:** A CLI interface to interact with the agent ("go to the red door", "which door is closest by manhattan distance?", "go to the delivery target").
- **LLM Intent Compiler:** Parses fuzzy user utterances into typed, validated intents without directly executing unsafe actions.
- **Capability Arbitration & Verification:** Evaluates what the LLM *wants* to do against a registered manifest of *actual* capabilities, automatically asking the user for clarification (e.g., "Which distance metric?") or refusing unsupported requests.
- **Persistent Scene Model & Memory:** Maintains episodic memory of prior grounded targets ("go to the next closest one") and durable knowledge ("the red door is the delivery target").
- **Deterministic Execution:** Uses a prewarmed caching system to execute tasks quickly without making any runtime LLM calls during the rendering loop.

## Environment Compatibility

JEENOM is currently configured to run with **MiniGrid**, which is itself a **Gymnasium** environment. 
- **Using other Gymnasium environments:** Because JEENOM interacts through standard Gymnasium spaces and a modular `CapabilityRegistry`, it can be adapted to work with any Gymnasium environment. To do so, you simply need to implement the corresponding observation/action primitives in `primitive_library.py` and register them in your capability manifest.

## Planned Scope of Work

The architecture is being built in phases according to the Capability Ladder (`PlanOfAction/task_plan.md`). We are currently at **Phase 7.95**.

Future phases include:
- **Phase 8 (General Object Handling):** Expanding beyond door navigation to general objects, enabling tasks like "pick up the red key" or "unlock the blue door".
- **Phase 9 (Operator Correction):** Supporting mid-run execution interruptions and corrections.
- **Phase 10 (Multi-Step Task Planning):** Allowing the compiler to compose multi-step action plans from sequential natural language requests.
- **Phase 11 (Continuous World Execution):** Moving away from independent, isolated episodes to a continuous world model where the agent accumulates state without environment resets.

## Installation and Usage

### Requirements
Ensure you have Python installed, along with the required dependencies:
```bash
pip install gymnasium minigrid
```

If you plan to use the live LLM compiler instead of the deterministic smoke-test fallback, you will need to set an OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```
*(Note: JEENOM currently uses OpenRouter to multiplex LLM models. If you want to use a direct Anthropic (`ClaudeCode`) or OpenAI (`CodexKey`) API key instead of OpenRouter, you must either route those requests through your OpenRouter account, or manually update the endpoint URL in `jeenom/llm_compiler.py` to point to the respective provider's API).*

### Running the Operator Station
To interact with JEENOM, run the interactive operator station:
```bash
python run_operator_station.py
```

### Running Evaluations
To verify that the system is functioning correctly, you can run the master evaluation suite:
```bash
python evals/eval_master.py
```
*(Note: If `OPENROUTER_API_KEY` is not set, the eval suite will automatically fall back to the deterministic smoke-test compiler for tests that support it).*
