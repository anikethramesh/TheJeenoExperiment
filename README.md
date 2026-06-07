# JEENOM

Just-In-Time Embodied Execution Networked Operational Model

JEENOM is an experiment in substrate-independent cognition. The core idea is
simple: a substrate exposes Sense, Cortex, and Spine primitives; JEENOM turns
operator steering into claims, plans, readiness decisions, and safe execution.

The broader JEENO vision is an externalized, queryable, and auditable cognition
layer for robots. This repository is the JEENOM prototype of that idea, currently
tested in MiniGrid.

## Current Status

JEENOM is an architecture prototype in MiniGrid. **Phase 10: Operator Station
Extraction** is complete. The project now has eval-backed enforcement for the
core architecture boundaries:

- canonical blocks: `OperatorStation`, `Cortex`, `ReadinessGraph`, `Sense`,
  `Spine`, `KnowledgeBase`, and `SubstrateAdapter`
- typed messages between those blocks
- one knowledge surface for claims, procedures, and provenance
- station-owned runtime context, substrate package, and context-driven
  planner/verifier semantics

Phase 10 extraction is nearly complete. `OperatorStationSession` is still large,
but the blocking boundaries are now explicit: runtime package injection,
operational context, substrate adapter, domain helper, turn orchestrator,
side-effect authority, command authority, and context-driven planning semantics.
The baseline now includes a live operator regression probe because structural
boundary checks alone were not enough to prove user-visible progress.
The current architecture task is Phase 10I: operator-defined primitive assembly,
so commands like defining `convenientDistance = min(manhattan, euclidean)` can
become typed, validated, reusable query primitives instead of unsupported text.
Phase 10I is currently in red-bar mode: the evals/tests for this behavior have
been added and are expected to fail until the implementation lands.

The guiding split is:

- **WHY**: operator/mission steering, constraints, authority, risk, budget, and
  stopping rules.
- **WHAT**: JEENOM's architecture-level intent, evidence needs, claims, plans,
  procedures, readiness, and execution authority.
- **WHERE/MEANING**: `OperationalContext`, the typed situation frame describing
  objects, task families, references, grounding semantics, claim rules, and
  display rules for the current domain.
- **HOW**: substrate/tool bindings such as camera frames, lidar scans, MiniGrid
  observations, ARC game states, path planners, controllers, `env.step`, and
  policies.

Concrete HOW belongs behind substrate adapters.
Domain meaning belongs behind `OperationalContext`.

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
  dispatch, synthesis/repair, memory writes, and domain-specific formatting.
- Phase 9E added a thin `RepresentationStore`, `ClaimRecord`, and
  `KnowledgeSnapshot`; deeper station extraction still needs to move more
  orchestration behind clean modules.
- Phase 10A moved command/result trace construction into `CommandAuthority`;
  Phase 10B moved execution, raw-motor, and memory-write ticket minting into
  `SideEffectAuthority`; Phase 10C introduced `SubstrateAdapter` and moved the
  first MiniGrid env/runtime HOW paths into `MiniGridSubstrateAdapter`; Phase
  10D added the typed `OperationalContext` and `MiniGridOperationalContext`;
  Phase 10E added a context-bound `MiniGridDomainHelper`; Phase 10F added
  `TurnOrchestrator` for top-level turn routing; Phase 10G added
  `RuntimePackage` injection; Phase 10H added `PlanningSemantics` so planner
  and verifier handles derive from `OperationalContext`.
- `CapabilityRegistry.minigrid_default()` is the only real substrate manifest.
- Deeper station branches, primitive validation fixtures, and the LLM compiler
  profile still carry MiniGrid-shaped assumptions.
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

Last green baseline after Phase 10H live regression repair:

- `python evals/eval_master.py --suite cleanup`: 28/28 passing.
- `python evals/eval_master.py`: 57/57 passing.
- `python -m pytest -q tests`: 229 passed.

Current Phase 10I red-bar signal:

- `python evals/eval_master.py --suite cleanup`: 28/29 passing;
  `phase10i_user_defined_metric_probe.py` fails as expected.
- `python -m pytest -q tests/test_phase10i_user_defined_metrics.py`: 6 failed
  as expected until operator-defined primitive assembly is implemented.

Roadmap:

- Phase 10I: add operator-defined primitive assembly for pure query/grounding
  primitives.
- Phase 11: add the minimal evidence-planning loop.
- Phase 12: demonstrate the same cognition loop on MiniGrid and a robotics-like
  substrate.
- Phase 13: pressure the same architecture with an ARC-style steerable prototype.
