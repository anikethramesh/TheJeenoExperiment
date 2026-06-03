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

JEENO is an architecture prototype in MiniGrid. **Phase 9A Cleanup** restored
the immediate safety/readiness gates; **Phase 9B: Substrate Contract
Foundation** now adds the first generic primitive-contract gates. The goal is to
make primitive contracts explicit before Phase 10 hardening, so JEENO can become
retrofit cognition for arbitrary robots and stacks rather than a cleaner
MiniGrid controller.

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

- Phase 9A issues around motor leakage, request/readiness recording, repair
  truthfulness, episode semantics, and eval coverage are fixed in the current
  baseline.
- Phase 9B now has initial substrate-contract gates for authority, safety class,
  claim confidence/frame matching, validation hooks, and
  substrate/config/tool/calibration reuse invalidation.
- Phase 9B still needs broader conformance/fuzz coverage and real non-MiniGrid
  adapter pressure before robot-port claims are credible.
- Harder MiniGrid domains and robot ports remain future work.

Until Phase 9B is complete, current evals should be treated as useful probes,
not full architectural proof for arbitrary robots.

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
- Substrate primitives are contractual objects, not string handles. Readiness
  must check preconditions, effects, required/produced claims, frames/units,
  safety class, authority, failure modes, validation hooks, and substrate
  fingerprint assumptions.

## Environment

JEENO currently runs on MiniGrid through Gymnasium. A future robot or simulator
port should start by registering the substrate's actual primitive contracts and
bindings, then reusing the same intent, readiness, claims, and execution-control
layers.

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

Run the Phase 9 cleanup suite:

```bash
python evals/eval_master.py --suite cleanup
```

List the manifest-selected probes without running them:

```bash
python evals/eval_master.py --list
```

Run the project-local tests:

```bash
python -m pytest -q tests
```

Avoid treating whole-repo `pytest` as the primary project signal right now,
because the local `Minigrid/` tree can introduce unrelated dependency noise.

Current baseline after the initial Phase 9B substrate-contract slice:

- `python evals/eval_master.py`: 37/37 passing.
- `python -m pytest -q tests`: 165 passed.

Phase 9B substrate-contract probes are now part of the cleanup/architecture gate
before Phase 10 operational hardening.
