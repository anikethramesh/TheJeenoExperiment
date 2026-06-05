# JEENO Implementation Plan

This file is the current roadmap. It replaces the earlier phase diary with a
compact record of what exists, what invariants must hold, and what comes next.

## Current Known State

- Current phase: **Phase 9D - Strict Cortical Schema Enforcement** is complete;
  the project is ready to start Phase 10.
- Phase 9A cleanup, the initial Phase 9B substrate-contract slice, and Phase 9C
  objective distillation are implemented.
- Current cleanup/audit signal:
  - `python evals/eval_master.py --suite cleanup`: 15/15 passing
  - `python evals/eval_master.py`: 44/44 passing
  - `python -m pytest -q tests`: 196 passed
- Whole-repo `pytest` is not the main project signal right now because the local
  `Minigrid/` tree can introduce unrelated dependency noise.

## Core Invariants

- LLM compiler outputs are typed schema objects only.
- Runtime validates and executes deterministic primitives.
- Unknown primitives must be rejected, clarified, or routed through explicit
  synthesis/repair paths.
- No LLM calls are allowed inside the rendered control loop.
- Procedure, sense, and skill templates must be compiled/prewarmed before render.
- `OperatorIntent` is not the execution plan.
- `RequestPlan` and `ReadinessGraph` are the execution-control plane.
- Claims are the universal fact abstraction:
  - Grounding claims are derived from scene observation, session-scoped, and
    scene-fingerprinted.
  - Operator claims are asserted by the operator, durable, and invalidated only
    by explicit retraction.
- Invalid or stale plan/claim reuse must never execute silently.
- Substrate primitives are not just names. Every robot or simulator port must
  expose typed primitive contracts: preconditions, postconditions/effects,
  required claims, produced claims, safety class, authority level, frames/units,
  failure modes, validation hooks, and substrate fingerprint assumptions.
- A valid LLM schema or valid primitive name is not sufficient for execution.
  The `ReadinessGraph` must also prove that the primitive contract is satisfied.
- Schemas are executable authority boundaries, not logging artifacts. No station
  path may execute motion, mutate memory, register synthesis, answer from claims,
  resume clarification, or run a mission step unless the action is represented by
  a current typed plan, readiness verdict, and approved command/ticket.

## Completed Phases

### Phase 0 - MiniGrid Smoke Test

Status: done.

Proved the basic MiniGrid wrapper, observation, action, render, and simple task
execution path. This established MiniGrid as the first test substrate, not the
target architecture.

### Phase 1 - Minimal JEENO Vertical Slice

Status: done.

Implemented the first Cortex/Sense/Spine split with typed runtime contracts:
world samples, operational evidence, percepts, execution contracts, execution
reports, and operational memory. This created the basic separation between
understanding, sensing, and actuation.

### Phase 2 - Typed LLM Compiler Boundary

Status: done.

Established that LLMs compile typed schema objects and never directly execute.
Runtime code validates compiler outputs and rejects unknown primitives. A
deterministic smoke-test compiler exists for offline evals and regression tests.

### Phase 3 - JIT Template Cache And Prewarm

Status: done.

Added cached `ProcedureRecipe`, `SensePlanTemplate`, and `SkillPlanTemplate`
objects. Known task families can be prewarmed before render so the rendered
runtime loop runs from validated cached templates.

The key guardrail from this phase remains active:

- runtime LLM calls during render: 0
- cache misses during render: 0

### Phase 4 - Larger Same-Task Stress Test

Status: done.

Ran the same `go_to_object` structure in a larger GoToDoor environment. This
proved that the cached runtime pattern can transfer across a larger instance of
the same task family, but it did not prove broad task competence.

### Phase 5 - CLI Operator Station

Status: done.

Added `OperatorStationSession` and `run_operator_station.py`. The station owns
environment configuration, compiler choice, memory, plan cache, render mode, and
last result. It supports task instructions, knowledge updates, status/cache
queries, reset, and quit from natural operator turns.

### Phase 6 - Memory-Grounded References And Session Semantics

Status: done.

Added durable delivery-target knowledge and episodic references such as
`last_target`, `last_task`, and `last_successful_instruction`. The operator can
ask for the delivery target, repeat the last task, or refer to the previous
target.

The intended current semantic decision is still:

- each task starts from fresh task-episode semantics for now
- continuous-world task chaining is future work
- reset clears episodic context but keeps durable operator claims by default

### Phase 7 - Operator Understanding, Readiness, And Synthesis

Status: done.

Built the main understanding and readiness stack:

- `OperatorIntent` schema for typed operator intent.
- LLM compiler plus deterministic fallback.
- `TargetSelector` and `GroundingQueryPlan` for scene/query grounding.
- `SceneModel` projected from sense output.
- `StationActiveClaims` for scene-scoped grounding claims.
- `CapabilityRegistry` derived from the primitive substrate.
- `CapabilityMatcher` for exact required-handle matching.
- `IntentVerifier` to inject missed semantic requirements such as farthest,
  ordinal, ranked, metric, and cardinality signals.
- `CapabilityArbitrator` for typed gap decisions.
- Safe synthesis scaffolding for pure grounding/query primitives.
- Collaborative synthesis proposal flow with operator approval.
- `RequestPlan`, `RequestPlanStep`, `ReadinessGraph`, and `ReadinessNode`.

This phase moved the project from phrase handling toward an inspectable control
plane: language produces typed intent, intent produces a request plan, and the
readiness graph decides whether to clarify, synthesize, answer, update memory,
execute, or refuse.

### Phase 8 - Cross-Environment Adaptation And Abstraction Hierarchy

Status: done, with cleanup required before further capability expansion.

Added the pieces needed to test whether the architecture can scale beyond one
MiniGrid happy path:

- Environment identity and explicit environment assumptions.
- Conservative `RequestPlan` reuse with semantic plan keys.
- Stale claim and invalid reuse detection.
- Named concept knowledge base with durable operator claims.
- Mismatch detection for stale claims, absent entities, invalidated grounding,
  unsupported grounding, missing primitives, and constraint weakening.
- Unified claim vocabulary for grounding and operator claims.
- Procedure concepts and raw multi-step utterance sequences.
- Direct motor-command composition for simple primitive repetitions.
- Explicit 5-level hierarchy:
  - L0 primitive
  - L1 command
  - L2 procedure
  - L3 task
  - L4 goal/mission
- `command_registry.py` for motor and sensory command definitions.
- Typed observation and execution claims inside Cortex.
- `MissionContract` and mission execution with abort-on-failure semantics.
- Initial operational repair loop scaffolding.

Phase 8 also exposed the main cleanup problem: some newer execution paths are
not yet fully governed by the same readiness and safety boundaries as the older
task path.

## Phase 9 - Cleanup And Structural Enforcement

Status: current umbrella phase.

Goal: close the architecture leaks exposed by Phase 8 before moving into
operational hardening, harder MiniGrid domains, or robot ports. Phase 9 is not a
feature phase; it is the phase that makes the control plane non-optional.

Phase 9 is organized into four subphases:

- **9A - Cleanup**: restore immediate safety/readiness gates and green local
  verification.
- **9B - Substrate Contract Foundation**: make primitive contracts visible to
  readiness.
- **9C - Objective Distillation**: remove lossy vocabulary-driven intent checks.
- **9D - Strict Cortical Schema Enforcement**: make typed plans/tickets the only
  executable authority.

### Phase 9A - Cleanup

Status: done.

Goal: restore the architectural gates before adding new capability. This phase
is mandatory before harder MiniGrid domains, richer repair, or robot ports.

Tasks:

- Restore the project-local test suite to green.
- Fix eval coverage so `eval_master.py` catches unit-level architectural
  regressions.
- Keep eval execution manifest-driven:
  - `python evals/eval_master.py --list` shows the selected probes
  - `python evals/eval_master.py --suite cleanup` runs the Phase 9 red-bar suite
  - utility modules such as harnesses are not treated as evals
- Repair the motor-command safety boundary:
  - direct primitives are useful for explicit motor commands
  - task requests like "pick up the red key" must not bypass task/readiness gates
  - unsupported task capability must refuse or enter a typed repair path
- Restore or explicitly redesign episode semantics:
  - repeat/reference tasks currently drift toward continuous-world adapter reuse
  - current docs and tests still expect fresh task-episode semantics
  - choose one behavior and update code, docs, and evals together
- Fix the repair loop:
  - repair must re-evaluate and re-dispatch when it claims success
  - otherwise it must honestly report that it only cleared state
- Resolve arbitrator interface drift in tests and test doubles.
- Restore or replace deleted eval coverage where still architecturally relevant:
  - query plan probes
  - grounding composition probes
  - primitive synthesis probes
  - operator clarification probes
- Update golden and stress evals so they test architecture invariants, not only
  a single successful MiniGrid seed.
- Keep the runtime-render guarantees intact:
  - runtime LLM calls during render: 0
  - cache misses during render: 0

Exit criteria:

- `python evals/eval_master.py` passes.
- `python -m pytest -q tests` passes.
- Unsupported object/task requests cannot execute through motor-command leakage.
- Repeat/reference behavior is consistent with the documented episode policy.
- Repair-loop evals prove either successful re-dispatch or honest non-execution.
- README and this task plan match the actual project state.

### Phase 9B - Substrate Contract Foundation

Status: done; initial foundation slice implemented.

Goal: make the substrate boundary explicit before operational hardening. JEENOM
must not merely solve MiniGrid through a better station; it must know what a
robot or simulator primitive is allowed to do, what facts it requires, and when
reuse or execution is unsafe.

Tasks:

- Extend primitive manifests with contract metadata:
  - preconditions and postconditions/effects
  - required claims and produced claims
  - units and coordinate/reference frames
  - safety class and authority level
  - failure modes
  - validation hooks for shadow/simulation/preflight checks
  - substrate/config/tool/calibration fingerprint metadata
- Extend `ReadinessGraph` so it gates on primitive contracts, not only primitive
  names and implementation status.
- Extend claim metadata with freshness, confidence, frame/source provenance, and
  authority. Low-confidence, expired, or frame-mismatched claims must block.
- Extend plan reuse invalidation so substrate/config/tool/calibration changes
  reject reuse even when the semantic task shape is identical.
- Add Phase 9B evals/tests before broad implementation:
  - malformed primitive contracts are rejected
  - authority-sensitive primitives block without explicit authority
  - safety-classed actuation requires a validation hook
  - stale/low-confidence/frame-mismatched claims block
  - substrate fingerprint changes invalidate reuse
  - MiniGrid-specific nouns do not appear in generic contract gates
- Keep current MiniGrid behavior green while proving MiniGrid is only one
  substrate adapter.

Exit criteria:

- Phase 9B substrate-contract evals pass.
- `python evals/eval_master.py` passes.
- `python -m pytest -q tests` passes.
- Existing MiniGrid evals still pass without weakening the new generic gates.

### Phase 9C - Objective Distillation

Status: done.

Goal: eliminate scattered vocabulary-based semantic normalization by having the
LLM explicitly distill the operator's selection intent into a structured field.

Problem this solved: `OperatorIntent` carried a plan (how to execute) but no
explicit structured statement of what the operator wanted. Every validation layer
had to re-derive intent from utterance text using hardcoded vocabulary lists.
Adding a new primitive domain (temperature, size) required updating keyword lists
in `semantic_normalizer.py`, `intent_verifier.py`, and `operator_station.py`.

What was built:

- `SelectionObjective` dataclass in `schemas.py` with two fields that matter:
  - `direction: "minimum" | "maximum"` — closed enum, validated strictly
  - `attribute: str` — open string; the LLM uses controlled vocabulary but the
    code does not hardcode it
- `selection_objective: SelectionObjective | None` field added to `OperatorIntent`
- `OperatorIntent.from_dict()` parses `selection_objective` (optional,
  backwards-compatible — existing LLM output without it continues to work)
- `selection_objective` added to the JSON schema so structured LLM output can
  include it, and to the LLM prompt with explicit vocabulary instruction:
  `direction="maximum"` for farthest/hottest/largest/greatest, `direction="minimum"`
  for closest/coldest/smallest
- `IntentVerifier.enrich()` now has two paths:
  - **Objective-based (primary)**: when `selection_objective` is set, inversion
    check is `direction=="maximum" → expect order=="descending"`. Pure enum logic.
    No vocabulary scanning. Attribute-agnostic.
  - **Vocabulary-based (fallback)**: when `selection_objective` is absent
    (SmokeTestCompiler, legacy LLM output), existing `infer_direction_from_utterance`
    path runs unchanged.
- `_validate_grounding_query_plan_preserves_utterance` in `operator_station.py`
  updated with same two-path structure, gated on `intent.selection_objective`
- `SmokeTestCompiler` updated to set `selection_objective` on intents it builds
  for farthest, ordinal-farthest, and ordinal-closest cases
- `test_objective_distillation.py` — 20 tests: parsing, inversion detection,
  handle injection, vocabulary fallback, dual-direction plans, and the core
  scalability proof (`test_objective_path_does_not_need_utterance_text`: a
  completely opaque utterance still triggers inversion detection because the
  check reads the objective, not the text)
- `test_expose_compiler_boundary.py` transport A updated to include
  `selection_objective: {direction: "maximum"}` with an inverted plan so the
  expose test exercises the objective-based path end-to-end

Scalability proof: adding a temperature ranking primitive requires one change —
add `"temperature"` as an attribute example in the LLM prompt. No changes to
`semantic_normalizer.py`, `intent_verifier.py`, or `operator_station.py`.

Exit state:

- `python evals/eval_master.py`: 37/37 passing (no regressions).
- `python -m pytest -q tests`: 193 passed, 3 intentional failures (repair loop
  expose tests from Phase 9A, unchanged, document Phase 10 work).

### Phase 9D - Strict Cortical Schema Enforcement

Status: done.

Goal: make the typed cortical control plane mandatory. The architecture already
has the right schema objects, but `OperatorStationSession` still has loose
string/dict authority paths. Phase 9D turns schemas from audit records into the
only executable currency.

Problem this fixes:

- `OperatorCommand(kind: str, payload: dict)` was the effective station router.
- Deterministic fast paths can return loose commands before the full control
  plane.
- `run_task(instruction: str)` accepts raw natural-language strings.
- Memory updates accept loose payload dicts.
- Raw motor commands were described and implemented as bypassing task planning;
  this is now ticket-gated for explicit low-level commands.
- Clarification resumes can directly run tasks or write memory.
- Missions and procedures decompose into raw task strings and call `run_task`
  step by step.
- Current evals can pass while these structural leaks remain.

Required schema/control-plane additions:

- `CorticalEnvelope`: one operator turn with utterance, `OperatorIntent`,
  `RequestPlan`, `ReadinessGraph`, provenance, and pending/resume context.
- `ApprovedCommand`: typed command variants replacing loose
  `OperatorCommand(kind, payload)`.
- `ExecutionTicket`: the only object allowed to start task execution.
- `MemoryWriteTicket`: the only object allowed to mutate durable memory.
- `RawMotorTicket`: explicit low-level motor authority backed by a plan and
  readiness verdict.
- `MissionExecutionPlan`: parent mission plan plus child task plans/tickets.
- `CommandResult`: user-visible response plus plan/graph/ticket trace.

Current Phase 9D implementation progress:

- Done:
  - Phase 9D red-bar evals are added and wired into `eval_master.py`.
  - `ExecutionTicket` is required for task runtime entry.
  - Raw string task execution is rejected and leaves a blocking audit plan.
  - Mission child execution now builds child execution tickets.
  - `MemoryWriteTicket` is required for durable knowledge writes.
  - Concept teaching no longer stores concepts when planning/compiler validation
    fails.
  - `RawMotorTicket` for explicit low-level motor commands.
  - Raw motor plans now use `objective_type=motor_control`,
    `expected_response=execute_motor`, and `ReadinessGraph.next_action=execute_motor`.
  - `CommandResult` now wraps every public `handle_utterance()` response while
    preserving string compatibility for the CLI and older evals.
  - `CorticalEnvelope`/`ApprovedCommand`/`CommandResult` traces are recorded for
    status, scene query, knowledge update, task, and raw motor paths.
  - Pending clarification now stores request-plan/readiness/envelope context.
  - Clarification answers re-enter `RequestPlan`/`ReadinessGraph` evaluation
    before task execution, query answer, or memory write continuation.
  - Loose `OperatorCommand(kind, payload)` routing was removed from
    `operator_station.py`; station command objects now use `ApprovedCommand`.
  - Older evals that asserted `OperatorCommand` were updated to assert
    `ApprovedCommand`.
- Pending:
  - None for Phase 9D.

Implementation tasks completed:

- Replace station dispatch on string `kind` values with typed approved command
  variants.
- Rename or restrict `run_task` so it accepts an `ExecutionTicket`, not a raw
  instruction string.
- Convert deterministic fast paths into deterministic `OperatorIntent` producers
  or typed envelope producers, not direct executors.
- Route every operator turn through:
  `OperatorIntent -> RequestPlan -> ReadinessGraph -> ApprovedCommand -> CommandResult`.
- Keep explicit raw motor controls, but require a `RawMotorTicket` and primitive
  contract approval.
- Require `MemoryWriteTicket` for delivery-target updates, concept writes, and
  any future durable operator claim writes.
- Store pending clarification as a pending envelope/plan, then re-evaluate
  readiness when the operator answers.
- Represent mission/procedure children as child request plans and tickets, not
  raw task strings.
- Make repair update the active envelope/plan/graph and either redispatch through
  a valid ticket or report honest non-execution.
- Update the LLM prompt and schema docs so raw motor is a typed, authorized
  low-level command, not an unplanned bypass.

Phase 9D eval status:

- `phase9d_schema_enforcement_probe.py` - passing
  - every user-visible command path records envelope, plan, graph, and result
    trace
  - no loose station command can mutate state
- `phase9d_execution_ticket_probe.py` - passing
  - direct raw-string task execution is rejected
  - successful task execution requires `ExecutionTicket`
  - ticket request id matches the active `RequestPlan` and `ReadinessGraph`
- `phase9d_memory_write_gate_probe.py` - passing
  - durable knowledge writes require `MemoryWriteTicket`
  - concept teach cannot silently swallow planning failures
  - unsupported or malformed memory updates do not write durable state
- `phase9d_motor_ticket_probe.py` - passing
  - explicit low-level motor commands still work
  - motor execution produces a `RawMotorTicket`
  - object/task requests cannot become raw motor tickets
  - hazardous or restricted motor primitives require contract approval
- `phase9d_clarification_resume_gate_probe.py` - passing
  - clarification answers re-enter readiness
  - clarification resume cannot directly call task execution or memory mutation
- `phase9d_mission_child_ticket_probe.py` - passing
  - mission parent plans carry child task plans/tickets
  - unsupported child steps abort before motion
  - no mission path loops over raw task strings as executable authority
- `phase9d_static_architecture_probe.py` - passing
  - fail if task execution accepts raw strings
  - fail on station-side direct calls that bypass approved tickets
  - fail if memory mutation APIs accept loose dict payloads as authority

Phase 9D unit tests to add:

- schema round-trip and validation tests for envelope, ticket, and result types
- station tests proving every branch returns a typed `CommandResult`
- monkeypatch tests proving task execution cannot happen without a ticket
- memory tests proving writes require typed ticket authority
- motor tests proving raw motor is planned/gated
- mission tests proving child plans/tickets exist
- clarification tests proving resume goes back through readiness

Exit criteria:

- New Phase 9D evals fail on the current loose-router implementation for the
  right reasons.
- After implementation, `python evals/eval_master.py --suite cleanup` passes.
- After implementation, `python evals/eval_master.py` passes.
- After implementation, `python -m pytest -q tests` passes.
- No station path can execute, mutate memory, register synthesis, answer from
  claims, resume clarification, or run mission children from raw string/dict
  authority.
- `RequestPlan + ReadinessGraph + approved typed ticket` is the mandatory
  execution currency.

Exit state:

- `python evals/eval_master.py --suite cleanup`: 15/15 passing.
- `python evals/eval_master.py`: 44/44 passing.
- `python -m pytest -q tests`: 196 passed.

## Phase 10 - Operational Hardening

Status: planned.

Goal: close the loop on cross-environment robustness after Phase 9A cleanup and
Phase 9B substrate contracts are complete.

Planned work:

- Strengthen repair actions: `REFRESH_CLAIMS`, `REGROUND`, `CLARIFY`,
  `SYNTHESIZE`, and `ABORT`.
- Track synthesized primitive provenance, validation fixtures, reuse history,
  and failure history.
- Add intervention metrics: operator asks, repairs, reuse decisions, synthesis
  attempts, validation failures, and successful transfers.
- Build a full transfer eval that proves valid reuse is accepted, invalid reuse
  is rejected, repairs are logged, interventions are counted, and render-time
  guarantees are preserved.
- Handle missing primitive, ambiguity, no-path, and blocked-task cases through
  explicit operator ask, fallback, repair, or mission abort.

## Phase 11 - Harder MiniGrid Integration

Status: later.

Only begin after Phase 9 cleanup and Phase 10 hardening are stable.

Target domains include MultiRoom, DoorKey, and KeyCorridor. These require new
capabilities before they are meaningful architecture tests:

- exploration
- pickup
- drop
- toggle/open
- unlock
- inventory state
- blocked-path detection
- replan
- richer object grounding

The goal is not to solve MiniGrid for its own sake. The goal is to pressure the
architecture with task families that require new primitives, new claims, and
repairable failures.

## Phase 12 - Port

Status: later.

Porting should happen after the architecture gates are stable.

Order of work:

- Register the new substrate's primitive set and capability metadata.
- Port Sense bindings.
- Port Spine/action bindings.
- Reuse the same OperatorIntent, RequestPlan, ReadinessGraph, claims, memory,
  arbitration, and repair layers.
- Run transfer and hardening evals against the new substrate.

The port succeeds only if the understanding layer remains largely unchanged and
the substrate-specific work is concentrated in primitive registration and
runtime bindings.
