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

    Claims are the unified abstraction for any fact the station holds about the world.
    Every claim has a type, authority, scope, provenance, and an invalidation policy:

    - Grounding claims (StationActiveClaims) — derived from scene observation.
      Session-scoped. Tied to a scene fingerprint (agent pose + step count).
      Invalidated when the scene changes.
    - Operator claims (KnowledgeBase, OperationalMemory.knowledge) — asserted directly
      by the operator. Durable across session restarts. Invalidated only by explicit
      operator retraction (forget, clear memory). Named concepts and delivery-target
      facts are both operator claims.

    Both claim types flow into the ReadinessGraph. Grounding claims are never promoted
    to durable storage unless the operator explicitly asserts them. Computed analysis is
    not an operator claim unless the operator promotes it. Episodic memory stores the
    last plan, target, task, and result and is distinct from both claim types.

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
outputs, `ExecutionClaim` for motor outputs). The Cortex internal store (`self.claims`) holds
`ObservationClaim` values keyed by evidence name. `StationActiveClaims` are task-level claims
held by the operator station (scene-fingerprinted, session-scoped). `KnowledgeBase` holds
operator-asserted durable claims.
