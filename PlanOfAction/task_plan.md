# JEENOM Implementation Plan

This is the current roadmap. Keep it simple. JEENOM is not a MiniGrid solver;
MiniGrid is the first stress substrate.

## Objective

Build a steerable cognition layer where:

- Sense, Cortex, and Spine expose architecture-level primitive roles.
- Substrates provide concrete HOW bindings for those roles.
- The operator steers WHY: goal, constraints, risk, budget, authority, and
  stopping rule.
- JEENOM owns WHAT: intent, evidence needs, claims, plans, procedures,
  readiness, execution authority, and reusable macro structure.
- The same cognition loop can run on MiniGrid and a robotics-like substrate, and
  later pressure an ARC-style interactive reasoning substrate.

Elon-algorithm rule for this repo:

- Delete before adding.
- Simplify before generalizing.
- Add an architecture object only when an eval exposes a real failure mode.
- Do not introduce ontology because it sounds right.

## Current Known State

- Current phase: **Phase 10 - Operator Station Extraction**.
- Phase 9D is complete. Operator turns now route through typed envelopes,
  request plans, readiness graphs, approved commands, and tickets.
- Phase 9E is complete. Architecture blocks, message schemas, and the knowledge
  surface now have eval-backed enforcement before Operator Station extraction.
- Current verification signal:
  - `python evals/eval_master.py --suite cleanup`: 19/19 passing
  - `python evals/eval_master.py`: 48/48 passing
  - `python -m pytest -q tests`: 201 passed
- Whole-repo `pytest` is not the main project signal right now because the local
  `Minigrid/` tree can introduce unrelated dependency noise.

## Core Invariants

- WHY is steered.
- WHAT is architectural.
- HOW is substrate/tool-specific.
- JEENOM must not hardcode HOW inside the orchestration layer.
- Sense, Cortex, and Spine are architecture-native roles; concrete cameras,
  grid observations, path planners, controllers, game actions, and policies are
  substrate bindings.
- Canonical blocks communicate through typed messages only:
  `OperatorStation`, `Cortex`, `Sense`, `Spine`, `ReadinessGraph`,
  `KnowledgeBase`, and `SubstrateAdapter`.
- LLM compiler outputs are typed schema objects only.
- No LLM calls are allowed inside the rendered control loop.
- `OperatorIntent` is not the execution plan.
- `RequestPlan` and `ReadinessGraph` are the execution-control plane.
- Claims are the universal representation unit. A claim must carry enough
  provenance, scope, authority, freshness, and confidence to decide whether it
  can be used.
- Operator claims are durable authority. Observation/world claims are evidence;
  they are not operator truth unless the operator promotes them.
- Substrate primitives are contractual objects, not string handles.
- Side effects require typed authority:
  - `ExecutionTicket` for task/runtime entry
  - `RawMotorTicket` for explicit low-level motor action
  - `MemoryWriteTicket` for durable operator-claim mutation
- JEENOM must distinguish known, visible, inferred, stale, unknown, searchable,
  and not-knowable. It must not report global certainty from local visibility.
- Reusable macro actions are earned: only promote a solved decomposition after
  it has claims, provenance, preconditions, postconditions, and failure modes.

## Completed Phases

### Phase 0 - MiniGrid Smoke Test

Status: done.

Proved the basic MiniGrid wrapper, observation, action, render, and simple task
execution path.

### Phase 1 - Minimal JEENOM Vertical Slice

Status: done.

Implemented the first Cortex/Sense/Spine split with typed runtime contracts:
world samples, operational evidence, percepts, execution contracts, execution
reports, and operational memory.

### Phase 2 - Typed LLM Compiler Boundary

Status: done.

Established that LLMs compile typed schema objects and never directly execute.
Runtime code validates compiler outputs and rejects unknown primitives.

### Phase 3 - JIT Template Cache And Prewarm

Status: done.

Added cached procedure, sense, and skill templates. Known task families can be
prewarmed before render so runtime executes from validated cached templates.

Guardrail:

- runtime LLM calls during render: 0
- cache misses during render: 0

### Phase 4 - Larger Same-Task Stress Test

Status: done.

Ran the same `go_to_object` structure in a larger GoToDoor environment. This
proved same-task transfer, not broad competence.

### Phase 5 - CLI Operator Station

Status: done.

Added `OperatorStationSession` and `run_operator_station.py` as the first public
operator interface.

### Phase 6 - Memory-Grounded References

Status: done.

Added durable delivery-target knowledge and episodic references such as
`last_target`, `last_task`, and `last_successful_instruction`.

Current semantic decision:

- each task starts from fresh task-episode semantics for now
- continuous-world task chaining is future work
- reset clears episodic context but keeps durable operator claims by default

### Phase 7 - Understanding, Readiness, And Synthesis

Status: done.

Built the first typed control plane:

- `OperatorIntent`
- `TargetSelector` and `GroundingQueryPlan`
- `SceneModel` and `StationActiveClaims`
- `CapabilityRegistry`, `CapabilityMatcher`, and `CapabilityArbitrator`
- `IntentVerifier`
- safe synthesis scaffolding for pure grounding/query primitives
- `RequestPlan`, `RequestPlanStep`, `ReadinessGraph`, and `ReadinessNode`

### Phase 8 - Adaptation And Abstraction Hierarchy

Status: done.

Added environment identity, plan reuse, stale-claim detection, named concepts,
mismatch detection, command registry, typed claims, mission contracts, and the
5-level hierarchy: primitive, command, procedure, task, goal/mission.

This exposed the problem: newer station paths were not all governed by the same
readiness and safety boundaries.

### Phase 9 - Cleanup And Structural Enforcement

Status: done.

Goal: make the architecture non-optional before adding new phases.

Phase 9A fixed immediate safety/readiness leaks and restored green local
verification.

Phase 9B made primitive contracts visible to readiness: preconditions,
postconditions, required/produced claims, frames/units, safety class, authority,
failure modes, validation hooks, and substrate fingerprints.

Phase 9C added `SelectionObjective` so semantic preservation can use structured
intent rather than scattered vocabulary checks.

Phase 9D made typed cortical objects the execution currency:

- `CorticalEnvelope`
- `ApprovedCommand`
- `ExecutionTicket`
- `RawMotorTicket`
- `MemoryWriteTicket`
- `MissionExecutionPlan`
- `CommandResult`

Phase 9D exit state:

- `python evals/eval_master.py --suite cleanup`: 15/15 passing
- `python evals/eval_master.py`: 44/44 passing
- `python -m pytest -q tests`: 196 passed

#### Phase 9E - Block, Schema, And Knowledge Enforcement

Status: done.

Goal:

Make the simple architecture enforce itself before adding Phase 10 extraction or
new substrates.

Canonical architecture:

- `OperatorStation`: operator I/O, session state, pending clarification, result
  display. It does not own planning, sensing, execution, or durable knowledge.
- `Cortex`: intent preservation, RequestPlan construction, ReadinessGraph
  arbitration, repair/synthesis decisions, and steering questions.
- `ReadinessGraph`: the execution gate. It consumes schemas, primitive
  contracts, authority, and knowledge snapshots. It is not a memory store.
- `Sense`: satisfies evidence requests through substrate HOW and returns
  observation claims.
- `Spine`: satisfies execution contracts through substrate HOW and returns
  execution claims.
- `KnowledgeBase`: the representation boundary for claims, procedures, and
  provenance.
- `SubstrateAdapter`: concrete HOW for sensors, controllers, planners,
  MiniGrid actions, ARC actions, and validation hooks.

Where we have not drifted:

- The core blocks exist in code.
- `RequestPlan`, `ReadinessGraph`, `ApprovedCommand`, tickets, and
  `CommandResult` exist.
- Sense and Spine have typed runtime contracts.
- Primitive contracts and readiness checks exist.
- Phase 9D evals are green.

Drift found before 9E:

- Knowledge has drifted into multiple pockets:
  - `OperationalMemory.knowledge`
  - `OperationalMemory.episodic_memory`
  - `OperationalMemory.scene_model`
  - `KnowledgeBase`
  - `Cortex._claims`
  - `OperatorStationSession.active_claims`
- `OperatorStationSession` still does too much and can touch memory, active
  claims, scene models, formatting, MiniGrid details, repair, synthesis, and
  runtime execution.
- Sense and Cortex can still read memory pockets directly instead of receiving
  typed knowledge snapshots or writing claims through a representation boundary.
- ReadinessGraph still accepts `active_claims` and scene/memory context as
  direct arguments rather than querying a representation surface.
- Procedures, provenance, facts, beliefs, hypotheses, observation claims, and
  operator claims are not yet accessed through one enforced surface.
- Schema objects exist, but loose dict payloads still move between some paths.
- `MemoryWriteTicket` protected some durable writes, but older station paths
  could still mutate memory pockets directly.

What was implemented:

- Added `ClaimRecord` as the minimal common claim wrapper for facts, beliefs,
  hypotheses, operator assertions, observations, execution results, and
  procedures.
- Added `KnowledgeSnapshot` so readiness can consume typed knowledge state
  instead of station fields.
- Added `RepresentationStore` as the thin boundary over existing
  `OperationalMemory` and `KnowledgeBase`.
- Routed station active grounding claims through a representation-backed
  property.
- Routed station request-plan/readiness provenance through the representation
  store.
- Routed durable knowledge writes through `MemoryWriteTicket` into the
  representation store.
- Removed direct station writes to `memory.knowledge`,
  `memory.episodic_memory`, and `memory.scene_model`.

Non-bloat rule:

- Do not build a giant ontology.
- Do not add a world-model subsystem yet.
- Do not add a hypothesis engine yet.
- Do not add a macro-promotion framework yet.
- Keep existing working objects unless an eval proves they are lossy.

Implementation order:

1. **9E.1 - Block boundary enforcement**
   - Define allowed responsibilities for each block.
   - Add static evals for forbidden dependencies and direct boundary bypasses.
   - Move obvious cross-block calls behind narrow interfaces only when required
     by failing evals.
   - OperatorStation may orchestrate a turn, but it must not become the owner of
     Sense, Spine, KnowledgeBase, or substrate HOW.

2. **9E.2 - Schema/message enforcement**
   - Every block boundary uses typed messages.
   - No raw dict is allowed as authority or control flow.
   - Existing schemas are preferred: `OperatorIntent`, `RequestPlan`,
     `ReadinessGraph`, `ApprovedCommand`, tickets, `ObservationClaim`,
     `ExecutionClaim`, `ProcedureRecipe`, and `CommandResult`.
   - Add only the smallest missing schema if needed, likely one generic
     `ClaimRecord`/`KnowledgeClaim` wrapper for fact, belief, hypothesis,
     operator assertion, observation evidence, and execution result.

3. **9E.3 - KnowledgeBase enforcement**
   - Add a thin representation facade over existing `OperationalMemory` and
     `KnowledgeBase`.
   - The facade owns claims, procedures, provenance, invalidation, and snapshots.
   - Durable writes require `MemoryWriteTicket`.
   - Observation and execution reports enter as claims/provenance.
   - Scene/world snapshots are evidence sources, not durable truth unless the
     operator promotes them.
   - ReadinessGraph consumes claim/procedure snapshots from this surface, not
     random station fields.

Minimum knowledge API:

- `put_claim`
- `get_claim`
- `query_claims`
- `invalidate_claims`
- `put_procedure`
- `get_procedure`
- `record_provenance`

Required enforcement:

- Durable writes require `MemoryWriteTicket` or an equivalent typed authority.
- Observation claims carry source, confidence, scope, freshness, and provenance.
- Operator claims carry operator authority and explicit invalidation semantics.
- Hypotheses and beliefs must be distinguishable from confirmed facts.
- Procedures carry provenance, preconditions, postconditions, and failure modes
  when promoted for reuse.
- ReadinessGraph consumes the representation surface, not random station fields.
- Runtime execution produces claims/provenance through the same surface.
- Direct dictionary mutation outside the representation module is treated as an
  architectural violation.

Evals/tests to add first:

- Block-boundary probe: station/cortex/sense/spine cannot import or mutate each
  other's internals except through approved interfaces.
- Direct-memory probe: no path outside the representation module writes
  `memory.knowledge[...]`, `memory.episodic_memory[...]`, `scene_model`, or
  `active_claims` directly.
- Schema-boundary probe: block methods reject loose dict authority and accept
  only typed message objects.
- Claim-shape probe: fact, belief, hypothesis, operator claim, observation
  claim, and execution claim round-trip through the same representation API.
- Provenance probe: every stored claim has source, authority, confidence or
  certainty state, freshness/invalidation, and provenance.
- Procedure probe: named concepts and procedures are accessible through the
  representation API, not only `KnowledgeBase`.
- Readiness probe: readiness reads claim/procedure state through representation
  snapshots, not station fields.

Phase 9E evals/tests:

- `phase9e_block_boundary_probe.py` - passing
- `phase9e_schema_boundary_probe.py` - passing
- `phase9e_knowledge_surface_probe.py` - passing
- `phase9e_readiness_snapshot_probe.py` - passing
- `tests/test_phase9e_representation.py` - passing

Acceptance criteria:

- The canonical blocks are enforced.
- The schema/message boundary is enforced.
- The knowledge base is no longer an informal collection of dicts.
- Existing Phase 9D behavior remains green.
- Every architectural block has a controlled read/write/query path into
  knowledge.
- Old memory pockets may remain internally, but they are behind the facade.
- Phase 10 extraction starts only after Phase 9E is green.

Phase 9E exit state:

- `python evals/eval_master.py --suite cleanup`: 19/19 passing
- `python evals/eval_master.py`: 48/48 passing
- `python -m pytest -q tests`: 201 passed

## Phase 10 - Operator Station Extraction

Status: current.

Goal: shrink `OperatorStationSession` into a thin facade and move substrate HOW
out of the orchestration path.

Why this comes before substrate generalization:

- The station currently owns conversation flow, deterministic fast paths, LLM
  intent routing, readiness dispatch, clarification, synthesis, repair, memory
  writes, MiniGrid adapter ownership, query formatting, and runtime execution.
- If we add robotics or ARC pressure now, we will just discover substrate leaks
  inside a 5k-line station.

Keep:

- public CLI behavior
- `CommandResult` compatibility
- Phase 9 gates and ticket authority
- MiniGrid golden path

Extract only what we need:

- `OperatorStationSession`: public facade and session state.
- `TurnOrchestrator`: one turn through intent, plan, readiness, command, ticket,
  and result.
- `SubstrateAdapter`: HOW boundary for manifest, sensing, action, task runtime,
  reset, prewarm, and validation hooks.
- `MiniGridSubstrateAdapter`: the current MiniGrid HOW.
- `CommandAuthority`: command/result tracing and ticket lookup; ticket creation
  moves in the side-effect authority slice.
- MiniGrid domain helpers: door/color/grid formatting and grounding display.

Do not create:

- a full world-model subsystem
- a hypothesis engine
- a macro-promotion framework
- a general cognitive-operation registry

Those come only if Phase 11 evals force them.

Progress:

- 10A command/result authority extraction: done.
  - Added `jeenom/command_authority.py` as a schema-only authority surface.
  - Added `evals/phase10_command_authority_probe.py`.
  - Added `tests/test_phase10_command_authority.py`.
  - `OperatorStationSession._record_command_result()` and
    `_pending_clarification_trace()` now delegate trace construction instead of
    manufacturing `CorticalEnvelope`, `ApprovedCommand`, and `CommandResult`
    inline.

Current Phase 10 baseline:

- `python evals/eval_master.py --suite cleanup`: 20/20 passing
- `python evals/eval_master.py`: 49/49 passing
- `python -m pytest -q tests`: 203 passed

Implementation slices:

1. Add station-modularity evals.
   - status: started with command-authority probe
   - current behavior snapshot
   - direct MiniGrid imports forbidden in the extracted orchestration kernel
   - side effects must pass through authority/tickets

2. Extract command/result tracing.
   - status: done
   - move envelope/result construction out of the station
   - keep `handle_utterance()` string-compatible

3. Extract side-effect authority.
   - status: pending
   - task, motor, and memory paths use one authority/ticket module
   - tests prove request id, plan, graph, and next action are bound

4. Extract the substrate adapter boundary.
   - status: pending
   - MiniGrid env ownership leaves the station facade
   - scene construction, reset, raw motor, prewarm, and task runtime become
     adapter responsibilities

5. Extract MiniGrid domain helpers.
   - status: pending
   - door/color/grid display and grounding helpers leave orchestration code
   - orchestration stops knowing MiniGrid vocabulary

6. Add a minimal mock substrate smoke eval.
   - status: pending
   - prove the kernel can run against a non-MiniGrid manifest/adapter
   - do not build a full robot or ARC port in this phase

Acceptance criteria:

- `OperatorStationSession` is a thin facade.
- Orchestration has no direct MiniGrid imports.
- Substrate HOW is behind `SubstrateAdapter`.
- MiniGrid-specific formatting lives outside the kernel.
- All side effects still require typed tickets.
- All public turns still return `CommandResult`.
- `python evals/eval_master.py --suite cleanup` passes.
- `python evals/eval_master.py` passes.
- `python -m pytest -q tests` passes.
- `python -m pytest -q tests` passes.

## Phase 11 - Minimal Representation And Evidence Planning

Status: planned.

Goal: prove JEENOM can collaborate about uncertainty without building a giant
ontology.

Minimum loop:

1. Operator gives WHY.
2. JEENOM turns it into WHAT.
3. JEENOM checks claims it already has.
4. If evidence is missing, JEENOM says what is unknown and how it could find out.
5. Operator steers budget/scope/risk.
6. JEENOM executes sensing/action through substrate HOW.
7. JEENOM updates claims and tries the plan again.

Minimal additions:

- Extend existing claims rather than adding many new classes.
- Add `RequestPlan` evidence steps.
- Add `ReadinessGraph` status/action for `needs_evidence`.
- Add simple typed steering constraints: budget, scope, risk, stopping rule.
- Keep operator claims separate from observation/world claims.
- Let procedures reuse solved decompositions only when the evidence supports it.

Pressure tests:

- MiniGrid visible-only query:
  - "closest/farthest door" must answer with scope or ask to search
  - no omniscient full-grid answer unless the substrate explicitly provides that
    as an authorized HOW
- Robotics-like mock:
  - path-planner primitive returns reachable/unreachable/path
  - JEENOM converts that into claims and plan decisions

Acceptance criteria:

- JEENOM can say: known, visible, inferred, stale, unknown, searchable, or
  not-knowable.
- Evidence gathering is represented as a plan, not a phrase branch.
- Operator steering changes typed plan constraints.
- MiniGrid and robotics-like mock both use the same cognition loop.

## Phase 12 - Cross-Substrate Demonstration

Status: planned.

Goal: prove the same architecture works on MiniGrid and a robotics-like
substrate.

Demo requirements:

- Same `OperatorIntent`, `RequestPlan`, `ReadinessGraph`, claims, tickets, and
  orchestration kernel.
- Different substrate HOW:
  - MiniGrid: grid observation, grid actions, grid pathing.
  - Robotics-like mock or robot stack: sensors, pose/map, path planner,
    controller/action binding.
- Same operator collaboration pattern:
  - ask for missing evidence
  - accept steering constraints
  - compose a plan from available primitives
  - execute through tickets
  - update claims/provenance

Acceptance criteria:

- One MiniGrid task and one robotics-like task follow the same cognitive flow.
- Differences are confined to substrate adapter and domain helpers.
- No MiniGrid vocabulary leaks into the generic kernel.

## Phase 13 - ARC-Style Steerable Prototype

Status: later.

Goal: pressure the same architecture with an ARC-like interactive reasoning
substrate.

Do not start by trying to solve ARC. Start with a tiny ARC-like substrate:

- observation/state API
- legal action API
- transition feedback
- score/end feedback
- substrate-specific state parser

JEENOM should own:

- representing observations as claims
- comparing state transitions
- asking for steering
- planning the next experiment/action
- updating claims/procedures from feedback

Acceptance criteria:

- The ARC-like prototype uses the same WHY/WHAT/HOW split.
- The LLM does not directly solve by free-form answer.
- The operator can steer experiments and strategy.

## Phase 14 - Operational Hardening

Status: later.

Goal: make the architecture reliable after the extraction and cross-substrate
proofs exist.

Planned work:

- repair metrics
- synthesis provenance
- intervention counts
- transfer evals
- missing primitive / ambiguity / no-path handling
- render-time guarantees preserved across substrates

## Phase 15 - Capability Stress Tests

Status: later.

Use harder MiniGrid, real robotics, or ARC-like tasks only as architecture
pressure tests. The goal is not benchmark chasing; the goal is to expose missing
primitive contracts, missing claims, missing evidence, bad decomposition, and
bad steering.
