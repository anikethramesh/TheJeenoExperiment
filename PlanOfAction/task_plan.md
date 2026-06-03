# JEENO Implementation Plan

This file is the current roadmap. It replaces the earlier phase diary with a
compact record of what exists, what invariants must hold, and what comes next.

## Current Known State

- Current phase: **Phase 9B - Substrate Contract Foundation**.
- Initial Phase 9B substrate-contract slice is implemented.
- `python evals/eval_master.py`: 37/37 passing.
- `python -m pytest -q tests`: 165 passed.
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

## Phase 9A - Cleanup

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

## Phase 9B - Substrate Contract Foundation

Status: current; initial foundation slice implemented.

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
