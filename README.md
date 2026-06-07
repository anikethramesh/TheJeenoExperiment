# JEENOM

Just-In-Time Embodied Execution Networked Operational Model

JEENOM is an experiment in substrate-independent cognition. The core idea is
simple: a substrate exposes Sense, Cortex, and Spine primitives; JEENOM turns
operator steering into claims, plans, readiness decisions, and safe execution.

The broader JEENO vision is an externalized, queryable, and auditable cognition
layer for robots. This repository is the JEENOM prototype of that idea, currently
tested in MiniGrid.

## Current Status

JEENOM is an architecture prototype in MiniGrid. **Phase 9E: Block, Schema, And
Knowledge Enforcement** is complete. The project now has eval-backed enforcement
for three simple boundaries:

- canonical blocks: `OperatorStation`, `Cortex`, `ReadinessGraph`, `Sense`,
  `Spine`, `KnowledgeBase`, and `SubstrateAdapter`
- typed messages between those blocks
- one knowledge surface for claims, procedures, and provenance

The current architecture task is Phase 10 extraction: shrink
`OperatorStationSession` into a substrate-independent orchestration facade plus
session, substrate, and domain adapters.

The guiding split is:

- **WHY**: operator/mission steering, constraints, authority, risk, budget, and
  stopping rules.
- **WHAT**: JEENOM's architecture-level intent, evidence needs, claims, plans,
  procedures, readiness, and execution authority.
- **HOW**: substrate/tool bindings such as camera frames, lidar scans, MiniGrid
  observations, ARC game states, path planners, controllers, `env.step`, and
  policies.

Concrete HOW belongs behind substrate adapters.

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
- `CorticalEnvelope`, `ApprovedCommand`, `ExecutionTicket`, `RawMotorTicket`,
  `MemoryWriteTicket`, and `CommandResult` as the current execution-authority
  and trace objects.
- Plan reuse, environment assumptions, stale-claim detection, mismatch detection,
  and early repair-loop scaffolding.
- JIT procedure/sense/skill template caching with the invariant that rendered
  runtime execution should not make LLM calls.
- Safe synthesis scaffolding for pure grounding/query primitives, with validation
  before registration.
- Explicit 5-level abstraction hierarchy: primitive, command, procedure, task,
  and goal/mission.

Current architectural debt:

- `OperatorStationSession` still owns too many internals:
  conversation flow, deterministic fast paths, LLM intent routing, readiness
  dispatch, MiniGrid adapter ownership, synthesis/repair, memory writes, and
  runtime task execution.
- Phase 9E added a thin `RepresentationStore`, `ClaimRecord`, and
  `KnowledgeSnapshot`; deeper station extraction still needs to move more
  orchestration behind clean modules.
- Phase 10A moved command/result trace construction into `CommandAuthority`;
  side-effect ticket creation and substrate HOW still need extraction.
- `CapabilityRegistry.minigrid_default()` is the only real substrate manifest.
- Request planning, primitive validation fixtures, and many station formatters
  still contain MiniGrid/door/grid assumptions.
- Contract preflight is represented and gated, but not yet a general executable
  proof system for arbitrary robot stacks.
- Robotics-like and ARC-style substrate pressure remain future work.

Current evals prove the Phase 9 control-plane invariants. They should not be
treated as proof that JEENOM generalizes to arbitrary robots, stacks, or ARC-like
domains until the same cognition loop runs on MiniGrid and a robotics-like
substrate.

## Architecture Invariants

- WHY is steered; WHAT is JEENOM architecture; HOW is substrate/tool binding.
- Sense, Spine, and Cortex are architecture-native roles. Their concrete
  sensors, actions, controllers, planners, and tool APIs are substrate bindings.
- Blocks communicate through typed messages only. A direct field or dict is not
  authority.
- Claims, provenance, procedures, and memory are the representation. They must
  be accessed through an enforced representation surface, not ad hoc station or
  memory dictionaries. Add new schema families only when an eval proves the
  current representation is lossy.
- LLM compiler outputs are typed schema objects only.
- The runtime validates and executes; unknown primitives are rejected or routed
  through explicit repair/synthesis paths.
- No LLM calls are allowed inside the rendered control loop.
- `RequestPlan` and `ReadinessGraph` are the execution-control plane, not raw
  operator text.
- Side-effectful actions require approved tickets:
  - `ExecutionTicket` for task/runtime entry.
  - `RawMotorTicket` for explicit low-level motor action.
  - `MemoryWriteTicket` for durable operator-claim mutation.
- Grounding claims are session-scoped and scene-fingerprinted.
- Operator claims are durable and invalidated only by explicit operator action.
- Invalid or stale plan/claim reuse must never execute silently.
- JEENOM must distinguish known, visible, inferred, stale, unknown, searchable,
  and impossible-to-know. It should not claim global certainty from local
  visibility.
- Substrate primitives are contractual objects, not string handles. Readiness
  must check preconditions, effects, required/produced claims, frames/units,
  safety class, authority, failure modes, validation hooks, and substrate
  fingerprint assumptions.

## Environment

JEENOM currently runs on MiniGrid through Gymnasium. A future robot or simulator
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

Current baseline after Phase 10A:

- `python evals/eval_master.py --suite cleanup`: 20/20 passing.
- `python evals/eval_master.py`: 49/49 passing.
- `python -m pytest -q tests`: 203 passed.

Roadmap:

- Phase 10: extract the operator station and substrate adapter boundary.
- Phase 11: add the minimal evidence-planning loop.
- Phase 12: demonstrate the same cognition loop on MiniGrid and a robotics-like
  substrate.
- Phase 13: pressure the same architecture with an ARC-style steerable prototype.
