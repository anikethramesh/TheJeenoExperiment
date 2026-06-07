## Codex Operating Rules

1. Do not redesign the architecture.
   Use the canonical architecture diagram.

2. Work capability by capability.
   Never implement "full architecture" as a single task.

3. For every implementation task:
   - state the phase
   - state the capability being added
   - state the files likely touched
   - state the success criteria
   - add or update a regression test

4. Preserve the working golden path:
   instruction: "go to the red door"
   compiler: llm
   render/prewarm enabled
   expected:
     task_complete=True
     runtime_llm_calls_during_render=0
     cache_miss_during_render=0
     final skill_plan=['done']

5. LLM compiler outputs are schema objects only.
   The runtime validates and executes.
   Unknown primitives must be rejected or corrected by fallback.

6. Do not put LLM calls in the rendered control loop.
   Compile/prewarm before render.
   Runtime loop must use cached templates.

7. If a change breaks the golden path, stop and fix that before adding new features.

8. Prefer explicit small patches over large refactors.

9. The station must verify that compiled intent preserves the semantic content of the
   operator's utterance. A valid schema object with contradictory or degraded semantics
   is a compilation failure, not an execution input.

   Superlatives, cardinality, direction, and negation signals in the utterance must
   correspond to declared capability requirements. The LLM may silently substitute a
   nearby executable capability for one it cannot fulfill — this is not acceptable.
   Intent inversion (farthest → closest) must be a hard stop. Silent degradation
   (ranked listing → single closest) must be a hard stop.

   The IntentVerifier sits between the LLM compiler output and the Readiness gate.
   It is deterministic, substrate-independent, and proactive: it extracts semantic
   signals from the utterance regardless of what the LLM declared, injects the
   correct required_capabilities, and lets the CapabilityMatcher fire on the
   enriched intent. No LLM calls. No substrate-specific logic.

   This rule applies to all substrate ports. A robot receiving "go to the farthest
   object" must not silently navigate to the closest one.

10. OperatorIntent is not the execution plan.
    Natural language must be converted into a typed RequestPlan before the station
    clarifies, synthesizes, answers, updates memory, or executes.

    The RequestPlan decomposes the operator request into dependency-aware steps:
    grounding, claims filtering, selection, task execution, memory update, answer
    generation, or control. Each step declares required handles, inputs, outputs,
    constraints, tie policy, memory reads/writes, and expected side effects.

    The ReadinessGraph arbitrates the RequestPlan, not the raw utterance. It checks each
    step against the CapabilityRegistry, primitive library, OperationalMemory,
    ActiveClaims, SceneModel, synthesis policy, and runtime cache/prewarm guarantees.
    Its verdicts are explicit: executable, needs_clarification, synthesizable,
    missing_skills, unsupported, stale_claims, or blocked_by_dependency.

    Clarification questions, synthesis proposals, query answers, and task execution must
    come from ReadinessGraph verdicts. This prevents phrase patches and silent capability
    degradation. If the request cannot be represented as a valid RequestPlan, the station
    must fail safely and explain the blocking node.

    Claims are the unified abstraction for any fact, belief, hypothesis,
    observation, operator assertion, or execution result the system holds.
    Every claim has kind/status, authority, scope, provenance, freshness, and an
    invalidation policy:

    - Grounding claims (StationActiveClaims) — derived from scene observation.
      Session-scoped. Tied to a scene fingerprint (agent pose + step count).
      Invalidated when the scene changes.
    - Operator claims (KnowledgeBase, OperationalMemory.knowledge) — asserted directly
      by the operator. Durable across session restarts. Invalidated only by explicit
      operator retraction (forget, clear memory). Named concepts and delivery-target
      facts are both operator claims.

    Both claim types flow into the ReadinessGraph through a representation
    surface. Grounding claims are never promoted to durable storage unless the
    operator explicitly asserts them. Computed analysis is not an operator claim
    unless the operator promotes it. Episodic memory stores the last plan,
    target, task, and result and is distinct from both claim types.

    Phase 9E enforcement has three gates:

    - block gate: canonical blocks do not mutate each other's internals
    - schema gate: block boundaries pass typed messages, not loose authority dicts
    - knowledge gate: claims, procedures, and provenance go through one
      representation surface

    Existing storage pockets may remain internally, but direct external mutation
    is architectural drift.

11. Substrate primitives are contractual objects, not string handles.
    Any robot, simulator, or tool stack must expose primitive metadata before JEENOM can
    compose it. The metadata must describe:

    - required claims and produced claims
    - preconditions and postconditions/effects
    - units and reference frames
    - safety class and authority level
    - known failure modes
    - validation hooks for shadow/simulation/preflight checks
    - substrate/config/tool/calibration fingerprints that affect reuse

    The ReadinessGraph gates execution on this contract metadata. A primitive that is
    present and implemented can still be blocked if authority is missing, a required
    validation hook is absent, a claim is stale or low-confidence, or a frame/unit
    assumption does not match. This is the Phase 9B foundation for JEENOM as retrofit
    cognition rather than a MiniGrid-specific controller.

12. ApprovedCommand and tickets are executable authority.
    A plan or primitive name is not enough to act. The station must wrap each
    operator turn in a `CorticalEnvelope`, route it through `RequestPlan` and
    `ReadinessGraph`, then issue an `ApprovedCommand`. Any side-effectful path
    needs the matching ticket:

    - `ExecutionTicket` before task/runtime entry
    - `RawMotorTicket` before explicit low-level motor action
    - `MemoryWriteTicket` before durable operator-claim mutation

    User-visible results are returned as `CommandResult`, preserving the envelope,
    approved command, ticket, and result trace. Query answers and refusals may not
    need a ticket, but they still need a current plan, graph, and approved command.

13. The current repo is a transitional implementation, not the final module split.
    `OperatorStationSession` currently hosts the conversation/session facade,
    deterministic fast paths, LLM intent compilation, request-plan recording,
    readiness dispatch, clarification/synthesis resume, memory writes, MiniGrid
    adapter ownership, and runtime task execution. Phase 9D made those paths
    ticket-gated. Phase 9E must enforce block, schema, and knowledge boundaries
    before Phase 10 extracts the station into a substrate-independent
    orchestration kernel plus substrate/domain adapters.

14. Separate WHY, WHAT, and HOW.
    The operator, mission, and safety policy steer WHY. JEENOM owns WHAT.
    The substrate provides HOW.

    - WHY: goal, constraints, authority, risk, budget, stopping rule.
    - WHAT: intent, evidence need, claim, plan step, procedure, readiness
      verdict, execution contract.
    - HOW: camera, lidar, MiniGrid observation, ARC state, SLAM map, path
      planner, controller, `env.step`, tool call, or policy.

    The orchestration layer must not hardcode HOW.

15. Sense, Cortex, and Spine are architecture-native primitive roles.
    Substrates bind concrete implementations into those roles.

    - Sense WHAT: request evidence and produce observation claims.
    - Cortex WHAT: preserve intent, decide missing evidence, build plans,
      arbitrate readiness, ask for steering, repair, synthesize, and explain.
    - Spine WHAT: satisfy execution contracts and produce execution claims.

16. Keep representation minimal.
    Claims, provenance, procedures, and memory are the representation. Do not add
    new schema families until an eval proves the current representation is too
    lossy. Operator claims are durable authority. Observation/world claims are
    evidence; they are not operator truth unless promoted by the operator.

17. No omniscience.
    JEENOM must distinguish known, visible, inferred, stale, unknown, searchable,
    and not-knowable. If evidence is incomplete, it must answer with scope,
    ask for steering, or plan evidence gathering.

18. Macro actions are earned.
    A solved decomposition may become a reusable procedure only after it has
    claims, provenance, preconditions, postconditions, and failure modes. Do not
    build a macro system before repeated working decompositions exist.

## Canonical Blocks

Phase 9E freezes the simple block map. Do not add a new block unless an eval
shows one of these cannot carry the responsibility.

| Block | Owns | Must not own |
|-------|------|--------------|
| `OperatorStation` | operator I/O, session state, pending clarification, result display | planning internals, sensing internals, execution internals, durable knowledge |
| `Cortex` | intent preservation, RequestPlan, readiness arbitration, repair/synthesis decisions, steering questions | substrate HOW, durable storage mutation |
| `ReadinessGraph` | executable/blocking verdicts from plans, contracts, authority, and knowledge snapshots | memory storage, execution |
| `Sense` | evidence requests and observation claims | task planning, durable operator truth |
| `Spine` | execution contracts and execution claims | intent planning, durable operator truth |
| `KnowledgeBase` | claims, procedures, provenance, invalidation, snapshots | substrate sensing/action HOW |
| `SubstrateAdapter` | concrete sensors, actions, planners, controllers, env/game calls, validation hooks | JEENOM WHAT decisions |

Every boundary between these blocks is a schema boundary. A block may carry local
working state, but cross-block control and authority must be typed messages.

## Minimal Message Surface

Use existing messages first:

- operator/control: `CorticalEnvelope`, `OperatorIntent`, `RequestPlan`,
  `ReadinessGraph`, `ApprovedCommand`, `CommandResult`
- authority: `ExecutionTicket`, `RawMotorTicket`, `MemoryWriteTicket`
- sense/execution: `EvidenceFrame`, `OperationalEvidence`, `ObservationClaim`,
  `ExecutionContract`, `ExecutionReport`, `ExecutionClaim`
- procedure: `ProcedureRecipe`, cached sense/skill templates

Only add one generic claim wrapper if the existing claim types cannot represent
fact, belief, hypothesis, operator assertion, observation evidence, and execution
result without losing authority/provenance/freshness.

## 5-Level Abstraction Hierarchy

JEENOM's execution stack is organised into five named levels, each with a Motor and Sensory
track. Claims are the universal I/O at every level boundary.

| Level | Name      | Motor track                         | Sensory track                      | Schema type               |
|-------|-----------|-------------------------------------|------------------------------------|---------------------------|
| L0    | primitive | `move_forward`, `turn_right`, …     | `parse_grid_objects`, …            | `ACTION_PRIMITIVES` dict  |
| L1    | command   | `navigate_to_object` (A* + execute) | `locate_object`, `verify_adjacent` | `MotorCommandTemplate` /  |
|       |           |                                     |                                    | `SensoryCommandTemplate`  |
| L2    | procedure | `go_to_object` step sequence        | (same procedure, sense track)      | `ProcedureContract`       |
| L3    | task      | `go_to_object` with params+readiness| (same task, grounding applied)     | `TaskContract`            |
| L4    | goal      | multi-task mission contract         | (same goal, abort-on-failure)      | `MissionContract`         |

**Motor track**: Spine executes `MotorCommandTemplate` primitives. Cortex issues `MotorSkillRequest`
(alias: `ExecutionContract`) to Spine after choosing the active skill.

**Sensory track**: Sense executes `SensoryCommandTemplate` primitives. Cortex reads evidence
needs from `SENSORY_COMMANDS` registry to build `EvidenceFrame` requests to Sense.

**Claims**: Every level produces and consumes typed Claim objects (`ObservationClaim` for sensory
outputs, `ExecutionClaim` for motor outputs). Phase 9E must make these available through one
representation surface. The current repo still has historical pockets: Cortex-local claims,
station active claims, durable operator memory, named concepts/procedures, and scene snapshots.
Those may remain internally, but architecture blocks should use the representation API rather
than mutating those pockets directly.

**Substrate contract**: Every L0 primitive and L1 command is backed by a
`PrimitiveSpec`/manifest contract. MiniGrid fills this from its grid primitives today; a
robot port must fill the same contract from its real controllers, sensors, frames, and
safety preflight checks.

## WHY / WHAT / HOW Examples

Robotics:

- WHY: "deliver to the target, stay within approved risk."
- WHAT: ground target, assess reachability, request a path, execute movement,
  monitor progress, verify completion, update claims.
- HOW: object detector, SLAM map, Nav2/MoveIt/path planner, controller, gripper,
  safety monitor.

MiniGrid:

- WHY: "go to the closest door globally."
- WHAT: determine whether the candidate set is complete, gather evidence if it
  is not, rank candidates, select target, navigate, verify adjacency.
- HOW: MiniGrid observation, turn/forward/toggle primitives, grid path planner,
  rendered or non-rendered env stepping.

ARC-AGI3:

- WHY: "solve the environment under a steering preference."
- WHAT: represent observations as claims, compare transitions, ask for steering,
  plan the next experiment/action, update claims/procedures from feedback.
- HOW: ARC game-state API, legal action API, state parser, replay/simulation
  tools, scoring/end-state feedback.

## Current Repo Shape After Phase 9E

The implementation now enforces the cortical control-plane objects and the first
block/schema/knowledge boundary gates.

Current enforced gateways:

- Operator turns return `CommandResult`.
- Each recorded command result carries a `CorticalEnvelope`.
- Task execution requires `ExecutionTicket`.
- Raw motor execution requires `RawMotorTicket`.
- Durable knowledge writes require `MemoryWriteTicket`.
- `ClaimRecord` is the minimal common claim wrapper.
- `KnowledgeSnapshot` lets readiness consume typed knowledge state.
- `RepresentationStore` wraps existing `OperationalMemory` and `KnowledgeBase`.
- Station active grounding claims are representation-backed.
- Station request-plan/readiness provenance is recorded through the
  representation store.
- Direct station writes to `memory.knowledge`, `memory.episodic_memory`, and
  `memory.scene_model` are blocked by Phase 9E probes.
- Clarification resumes re-enter request planning and readiness before action or
  memory mutation.
- Mission children build child execution tickets instead of calling raw task
  strings as authority.

Current architectural debt:

- Remaining block debt:
  - `OperatorStationSession` is still a large facade over orchestration,
    conversation, MiniGrid substrate bindings, query formatting,
    repair/synthesis, memory writes, and task runtime.
  - Sense, Cortex, ReadinessGraph, and station paths can still see each other's
    concrete fields in places.
- Remaining schema/knowledge debt:
  - Existing memory pockets remain internally while the facade stabilizes.
  - More Sense/Cortex/Spine paths should consume representation snapshots during
    Phase 10 extraction.
- Substrate drift:
  - `CapabilityRegistry.minigrid_default()` is the only real substrate manifest.
  - Request planning, primitive validation fixtures, and many station formatters
    still contain MiniGrid/door/grid assumptions.
  - Contract preflight is represented and gated, but not yet a general executable
    proof system for arbitrary robot stacks.

The next architecture step is Phase 10 station extraction. Do not add harder
domains first.
