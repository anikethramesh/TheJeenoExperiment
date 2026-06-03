# JEENO

Just-In-Time Embodied Execution Networked Operational Model

JEENO is an experiment in reducing deployment friction for embodied agents. The
core idea is that a robot or simulator should expose its primitives, while a
separate understanding layer handles operator intent, grounding, memory,
readiness, and safe execution.

The broader JEENO vision is an externalized, queryable, and auditable cognition
layer for robots. This repository is the JEENO prototype of that idea, currently
tested in MiniGrid.

## Current Status

JEENO is an architecture prototype in MiniGrid and is entering **Phase 9:
Cleanup**. The system is no longer just a door-navigation demo, but it is also
not production-ready. The next priority is to clean up architectural leaks and
make the eval suite a trustworthy gate before adding new capability.

Implemented so far:

- Natural-language operator station for task, memory, status, concept, sequence,
  motor, and mission requests.
- Typed LLM compiler boundary: LLMs emit schema objects; runtime validates and
  executes deterministic primitives.
- Capability registry and command registry over task, grounding, sensing, action,
  and claims primitives.
- Intent verification, capability matching, and arbitration to avoid silent
  capability degradation.
- Scene model, grounding claims, durable operator claims, episodic memory, and
  named concepts.
- Typed `RequestPlan` and `ReadinessGraph` for decomposing requests before
  clarification, synthesis, answer generation, memory updates, or execution.
- Plan reuse, environment assumptions, stale-claim detection, mismatch detection,
  and early repair-loop scaffolding.
- JIT procedure/sense/skill template caching with the invariant that rendered
  runtime execution should not make LLM calls.
- Safe synthesis scaffolding for pure grounding/query primitives, with validation
  before registration.
- Explicit 5-level abstraction hierarchy: primitive, command, procedure, task,
  and goal/mission.

Known cleanup work:

- Unsupported task requests can currently leak through direct motor commands in
  some paths, for example pickup-like utterances.
- The repair loop records some repairs but does not yet reliably re-dispatch the
  repaired request.
- Repeat/reference behavior has drifted toward continuous-world adapter reuse,
  while the documented current semantics still assume fresh task episodes.
- `eval_master.py` does not catch all regressions that the project-local tests
  catch.
- Harder MiniGrid domains and robot ports remain future work.

Until Phase 9 is complete, current evals should be treated as useful probes, not
full architectural proof.

## Architecture Invariants

- LLM compiler outputs are typed schema objects only.
- The runtime validates and executes; unknown primitives are rejected or routed
  through explicit repair/synthesis paths.
- No LLM calls are allowed inside the rendered control loop.
- `RequestPlan` and `ReadinessGraph` are the execution-control plane, not raw
  operator text.
- Grounding claims are session-scoped and scene-fingerprinted.
- Operator claims are durable and invalidated only by explicit operator action.
- Invalid or stale plan/claim reuse must never execute silently.

## Environment

JEENO currently runs on MiniGrid through Gymnasium. A future robot or simulator
port should start by registering the substrate's actual primitives and bindings,
then reusing the same intent, readiness, claims, and execution-control layers.

## Running

Install the basic dependencies:

```bash
pip install gymnasium minigrid
```

Run the operator station:

```bash
python run_operator_station.py
```

For live LLM compilation, set an OpenRouter API key:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Without an API key, many probes use the deterministic smoke-test compiler.

## Verification

Run the eval suite:

```bash
python evals/eval_master.py
```

Run the project-local tests:

```bash
python -m pytest -q tests
```

Avoid treating whole-repo `pytest` as the primary project signal right now,
because the local `Minigrid/` tree can introduce unrelated dependency noise.

At the start of Phase 9 Cleanup, the known state is:

- `eval_master.py`: 31/32 passing; `phase91_operational_repair_probe.py` failing.
- `python -m pytest -q tests`: 153 passed, 7 failed.
