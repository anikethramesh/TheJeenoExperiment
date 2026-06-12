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

- Current phase: **Phase 11 - Architecture Fix - Mission Flow** continues.
  Phase 11A and 11C are complete. Phase 11B hostile evals exist (red) and need
  production-code fixes before Phase 12.
- Phase 9D is complete. Operator turns now route through typed envelopes,
  request plans, readiness graphs, approved commands, and tickets.
- Phase 9E is complete. Architecture blocks, message schemas, and the knowledge
  surface now have eval-backed enforcement before Operator Station extraction.
- Phase 10D has a first-cut `OperationalContext`: MiniGrid domain meaning is now
  represented as a typed situation frame with a stable fingerprint and compact
  prompt slice.
- Phase 10E has a first-cut `MiniGridDomainHelper`: obvious MiniGrid
  door/color/metric display and parsing helpers moved out of the station and now
  consume `OperationalContext`.
- Phase 10F has a first-cut `TurnOrchestrator`: top-level turn routing and
  pending-clarification routing moved out of the station facade.
- Phase 10G has a first-cut `RuntimePackage`: the station can be initialized
  with an injected substrate/context/helper/registry package instead of birthing
  MiniGrid pieces inline.
- Phase 10H is complete. Planner and verifier control-plane handles now derive
  from `PlanningSemantics` over `OperationalContext`, and the station binds its
  verifier to the same runtime context it uses for planning.
- A live operator regression probe now checks that answers, intents,
  RequestPlans, ReadinessGraphs, and plan reuse agree on the same outcome.
- Phase 10I is complete. Operator-defined query metrics now become typed
  primitive-definition requests, gated proposals, validated/registered query
  primitives, operational-context metrics, ticketed knowledge records, and
  inline derived metrics embedded inside task requests.
- Phase 11A is complete. Inline compound missions now route through
  `MissionCortex`, typed `MissionExecutionPlan`, mission-linked
  `ExecutionTicket` provenance, and hostile mission-flow eval coverage.
- Phase 11B hostile evals exist and remain red. They expose paraphrase
  brittleness, false success, unsafe conditional motor leakage, lossy
  motor/procedure lineage, and mission flattening. Production-code fixes are
  the next required step.
- Phase 11C is complete. Seven compounding architectural violations resolved
  (import partition, domain purge, TurnOrchestrator dispatch extraction,
  knowledge-type rerouting, IntentCache, Readiness deletion, eval naming
  contract, hardware schema fields). See Phase 11C section below.
- Current verification signal:
  - `python evals/eval_master.py`: 60/60 passing
  - `python evals/eval_master.py --suite cleanup`: all cleanup probes passing
  - `python -m pytest -q tests`: 244 passed (last recorded)
- Eval naming contract is now enforced: all registered eval files use
  capability-based prefixes. The naming contract probe fails on violation.
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

Status: done.

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
- `CommandAuthority`: command/result tracing and ticket lookup.
- `SideEffectAuthority`: execution, raw-motor, and memory-write ticket minting.
- MiniGrid domain helpers: door/color/grid formatting and grounding display.

Do not create:

- a full world-model subsystem
- a hypothesis engine
- a macro-promotion framework
- a general cognitive-operation registry

Those come only if later evals force them.

Numbering rule:

- Phase 10 uses lettered implementation slices only: 10A, 10B, 10C, ...
- Every slice is eval/test first. There is no separate "eval slice"; the eval is
  part of the slice it protects.

Phase 10 exit baseline after live regression coverage:

- `python evals/eval_master.py --suite cleanup`: 28/28 passing
- `python evals/eval_master.py`: 57/57 passing
- `python -m pytest -q tests`: 229 passed

### Phase 10A - Command/Result Authority

Status: done.

Purpose: move "what happened this turn?" trace construction out of the station.

Implemented:

- Added `jeenom/command_authority.py` as a schema-only authority surface.
- Added `evals/phase10_command_authority_probe.py`.
- Added `tests/test_phase10_command_authority.py`.
- `OperatorStationSession._record_command_result()` and
  `_pending_clarification_trace()` now delegate trace construction instead of
  manufacturing `CorticalEnvelope`, `ApprovedCommand`, and `CommandResult`
  inline.

Measured outcome:

- Station no longer constructs command/result trace objects in result paths.
- Public `CommandResult` compatibility is preserved.

### Phase 10B - Side-Effect Authority

Status: done.

Purpose: move "who is allowed to change world or memory state?" ticket minting
out of the station.

Implemented:

- Added `jeenom/side_effect_authority.py` as the schema-only ticket minting
  surface.
- Added `evals/phase10_side_effect_authority_probe.py`.
- Added `tests/test_phase10_side_effect_authority.py`.
- `OperatorStationSession` no longer directly constructs `ExecutionTicket`,
  `RawMotorTicket`, or `MemoryWriteTicket`.

Measured outcome:

- Side-effect tickets have one named authority surface.
- The station still plans and dispatches, but it does not mint authority tokens
  inline.

### Phase 10C - Substrate Adapter Boundary

Status: done, first cut.

Purpose: move MiniGrid HOW out of the station facade.

Implemented:

- Added `jeenom/substrate_adapter.py` as the architecture-level HOW protocol.
- Added `jeenom/minigrid_substrate_adapter.py` as the MiniGrid HOW binding.
- Added `evals/phase10_substrate_adapter_probe.py`.
- Added `tests/test_phase10_substrate_adapter.py`.
- `OperatorStationSession` now gets sense, spine, capability registry, preview,
  scene observation, JIT prewarm, task runtime, and raw motor execution through
  `self.substrate`.
- Compatibility properties expose `preview_adapter` and `task_adapter` for
  older tests without moving ownership back into the station.

Measured outcome:

- `OperatorStationSession` no longer imports `MiniGridAdapter`, `MiniGridSense`,
  `MiniGridSpine`, or `ensure_custom_minigrid_envs_registered`.
- `OperatorStationSession` no longer calls `run_demo.build_env`,
  `run_demo.run_episode`, `run_demo.run_motor_sequence`, or
  `run_demo.prewarm_jit_cache` directly.
- `operator_station.py` dropped from roughly 5665 lines to 5606 lines.

Remaining 10C debt:

- The station still constructs the default MiniGrid substrate/context bundle
  instead of accepting an injected non-MiniGrid runtime package.

### Phase 10D - OperationalContext Schema

Status: done, first cut.

Purpose: standardize the situation frame that adapts JEENOM to a domain without
hardcoding that domain into the station.

Core distinction:

- `SubstrateAdapter`: HOW to sense, act, render, run, reset, and bind tools.
- `OperationalContext`: WHAT the current domain/mission means.

Schema:

- `context_id`
- `substrate_id`
- object vocabulary and attribute vocabulary
- task families and canonical task mappings
- reference semantics such as same, other, closest, farthest, delivery target
- grounding semantics such as distance metrics, visibility model, ranking, and
  tie policy
- claim mapping rules and required provenance fields
- operator-facing display rules
- environment identity fields
- known procedure/composition hints

Implemented:

- Add `OperationalContext` as a typed schema/message.
- Add a MiniGrid operational context implementation/manifest.
- Add `evals/phase10_operational_context_probe.py`.
- Add `tests/test_phase10_operational_context.py`.
- Register the Phase 10D eval in `evals/manifest.py`.
- `OperatorStationSession` now owns `operational_context` and
  `context_fingerprint` beside the substrate adapter.

Measured outcome:

- JEENOM can say what domain it is operating in without embedding that meaning
  directly in the station.
- MiniGrid door/grid/color meaning now has a named typed home:
  `MiniGridOperationalContext`.
- The context has a stable content fingerprint.
- The context can provide a compact slice for prompt/planning use without
  dumping full display/procedure metadata each turn.

Remaining 10D debt:

- Station paths that currently assume doors/colors/grid coordinates still need
  to read those assumptions from the context. That is the job of 10E.

### Phase 10E - Domain Helper Extraction

Status: done, first cut.

Purpose: move door/color/grid-specific parsing, presentation, and grounding
helpers out of orchestration using the `OperationalContext` from 10D.

Implemented:

- Added `jeenom/minigrid_domain_helper.py`.
- Added `evals/phase10_domain_helper_probe.py`.
- Added `tests/test_phase10_domain_helper.py`.
- Registered the Phase 10E eval in `evals/manifest.py`.
- Added MiniGrid color values and aliases to `MiniGridOperationalContext`.
- `OperatorStationSession` now owns a context-bound `domain_helper`.
- Obvious station-owned helpers were extracted:
  color normalization, color/object parsing, metric-answer parsing,
  entry labels, task utterance labels, ranked-door display, and color answer
  formatting.

Measured outcome:

- `OperatorStationSession` no longer defines `SUPPORTED_COLORS`,
  `_normalize_color()`, `_color_reference_in_utterance()`, `_entry_label()`,
  `_format_ranked_doors_from_claims()`, `_color_answer()`,
  `_is_manhattan_answer()`, `_is_euclidean_answer()`, or
  `_metric_from_grounding_handle()`.
- The new Phase 10E eval fails if those obvious helpers drift back into the
  station.
- `operator_station.py` dropped from 5609 lines after 10D to 5483 lines.

Remaining 10E debt:

- Some deeper MiniGrid semantics still live in `OperatorStationSession`,
  `llm_compiler.py`, `request_planner.py`, and `intent_verifier.py`.
- 10F extracts top-level turn orchestration; the remaining station domain
  branches should either move into helper modules or become context-driven
  planning logic.

### Phase 10F - Turn Orchestrator Extraction

Status: done, first cut.

Purpose: isolate one operator turn through intent, plan, readiness, command,
ticket, execution/answer, and result.

Implemented:

- Added `jeenom/turn_orchestrator.py`.
- Added `evals/phase10_turn_orchestrator_probe.py`.
- Added `tests/test_phase10_turn_orchestrator.py`.
- Registered the Phase 10F eval in `evals/manifest.py`.
- `OperatorStationSession` now owns a `turn_orchestrator`.
- `handle_utterance()` delegates top-level turn text routing to
  `TurnOrchestrator.handle_utterance_text()`.
- `execute_command()` remains as a compatibility shim, but delegates execution
  routing to `TurnOrchestrator.execute_command()`.
- Pending clarification routing moved to
  `TurnOrchestrator.handle_pending_clarification()`.

Measured outcome:

- `OperatorStationSession` no longer defines `_handle_utterance_text()` or
  `handle_pending_clarification()`.
- The new Phase 10F eval fails if top-level turn methods drift back into the
  station.
- `operator_station.py` dropped from 5483 lines after 10E to 5242 lines.
- `turn_orchestrator.py` is 214 lines and owns the top-level turn router.

Remaining 10F debt:

- `TurnOrchestrator` still delegates to many station-private implementation
  methods. The station is cleaner, but not yet a pure substrate-independent
  shell.
- `command_from_operator_intent()` and deeper request/intent planning branches
  still live in the station. Those are candidates for later extraction after
  10G proves the facade can host a non-MiniGrid substrate/context pair.

### Phase 10G - Runtime Package Injection

Status: done, first cut.

Purpose: make the station substrate-neutral at construction time.

This is a definitive architecture fix, not a mock-only smoke. The station must
be able to receive a runtime package instead of constructing MiniGrid pieces
internally.

Implemented:

- Added `jeenom/runtime_package.py`.
- Added `jeenom/minigrid_runtime_package.py`.
- Added `evals/phase10_runtime_package_probe.py`.
- Added `tests/test_phase10_runtime_package.py`.
- Registered the Phase 10G eval in `evals/manifest.py`.
- Added a typed runtime package/bundle containing:
  - `SubstrateAdapter`
  - `OperationalContext`
  - domain helper
  - capability registry
- Added a MiniGrid runtime package factory as the default path.
- `OperatorStationSession` accepts an injected `runtime_package`.
- `OperatorStationSession` no longer directly births MiniGrid substrate/context
  pieces except through the default MiniGrid runtime factory.
- Add a tiny injected non-MiniGrid runtime fixture only as the eval vehicle, not
  as the architectural achievement.

Measured outcome:

- Station can be initialized with an injected runtime package.
- The default MiniGrid CLI path still works.
- Substrate HOW is behind `SubstrateAdapter`.
- Domain meaning is behind `OperationalContext` and domain helper.
- `OperatorStationSession` construction no longer hardcodes
  `MiniGridSubstrateAdapter`, `MiniGridOperationalContext`, and
  `MiniGridDomainHelper` inline.
- The new Phase 10G eval fails if inline MiniGrid runtime construction drifts
  back into the station.
- All side effects still require typed tickets.
- All public turns still return `CommandResult`.
- `operator_station.py` is 5246 lines after 10G. This slice prioritizes
  substrate-neutral construction over line-count reduction.

10G handoff to 10H:

- `classify_utterance()` still uses a default MiniGrid domain helper for legacy
  deterministic parsing. That is not construction coupling, but 10H moved the
  planning/verifier boundary toward context-driven domain semantics.

### Phase 10H - Context-Driven Planning Boundary

Status: done.

Purpose: remove the blocking MiniGrid/domain assumptions from the planning and
verification boundary before moving to Phase 11.

This is the final Phase 10 cleanup step. It should not try to make the codebase
beautiful or small. It should only remove domain coupling that blocks
substrate-independent cognition.

Target:

- `request_planner.py` stops hardcoding the core meanings of door/color/distance
  where those can be read from `OperationalContext` or the domain helper.
- `intent_verifier.py` stops hardcoding the core MiniGrid signal vocabulary where
  those can be read from `OperationalContext` or the domain helper.
- Remaining station selector/grounding branches use context/domain helper for
  core object, attribute, metric, and display semantics.
- MiniGrid-specific LLM prompt examples may remain in the MiniGrid compiler
  profile; that is not Phase 10 cleanup debt unless it leaks into the
  architecture-neutral path.

Implemented:

- Added `jeenom/planning_semantics.py`.
- Added context capability-handle templates to `MiniGridOperationalContext`.
- `request_planner.build_request_plan()` accepts `planning_semantics`.
- `IntentVerifier` accepts `planning_semantics` and injects context-derived
  handles instead of hardcoded MiniGrid ranking handles.
- `OperatorStationSession` owns `planning_semantics` and a verifier bound to
  that same object.
- All station-local `build_request_plan()` calls pass the session semantics.
- Added `evals/phase10_context_planning_probe.py`.
- Added `evals/phase10_live_operator_regression_probe.py`.
- Added `tests/test_phase10_context_planning.py`.
- Added `tests/test_phase10_live_operator_regressions.py`.
- Registered the Phase 10H eval in `evals/manifest.py`.

Measured outcome:

- A non-MiniGrid token-ranking context can produce
  `grounding.all_tokens.ranked.score.agent` through both the planner and the
  verifier.
- The 10H probe fails if the worst hardcoded MiniGrid handles drift back into
  `request_planner.py` or `intent_verifier.py`.
- The live regression probe fails if a ranked-distance answer is backed by an
  unsupported/refuse plan, if metric follow-up loses prior grounding context, or
  if unresolved/refuse plans are cached as reusable progress.
- `operator_station.py` is 5488 lines afte
r the 10H live-regression fix. Phase
  10 deliberately did not chase file slimming, and this regression fix added
  station code that should be extracted later.
- Verification after 10H:
  - `python evals/eval_master.py --suite cleanup`: 28/28 passing
  - `python evals/eval_master.py`: 57/57 passing
  - `python -m pytest -q tests`: 229 passed

Remaining debt after Phase 10H:

- `OperatorStationSession` is still large and still owns deeper MiniGrid-shaped
  branches. That is acceptable for the prototype while the boundary objects are
  enforced.
- `classify_utterance()` still uses the default MiniGrid domain helper for
  legacy deterministic parsing.
- The LLM compiler prompt/profile still contains MiniGrid examples. That is a
  substrate profile issue, not a Phase 10 control-plane boundary leak.
- Repo/file-size minimization should be a later cleanup phase after capability
  pressure proves what should stay.

### Phase 10I - Operator-Defined Primitive Assembly

Status: complete.

Purpose: make collaborative primitive construction real. The operator must be
able to define a new pure query/grounding primitive by composing existing
primitives or formulas, then use it in later turns.

Why this belongs in Phase 10:

- This is not new MiniGrid capability; it is a missing architecture outcome.
- The project goal is just-in-time primitive assembly under operator steering.
- A command like "synthesize a new distance metric which is the minimum between
  euclidean and manhattan distance; call it convenientDistance" should not fall
  through as unsupported.
- Phase 12 evidence planning should build on this capability, not work around
  its absence.

Non-goals:

- Do not create a broad ontology.
- Do not add arbitrary unsafe code execution.
- Do not make actuation primitives synthesizable.
- Do not chase operator-station line-count reduction in this slice.

Eval-first requirements:

- Added `evals/phase10i_user_defined_metric_probe.py`, registered in
  `evals/manifest.py`; it now passes.
- Added `tests/test_phase10i_user_defined_metrics.py`; it now passes.
- The Phase 10I live eval covers:
  - `synthesize a new distance metric which is the minimum between euclidean and
    manhattan distance. call it convenientDistance`
  - `rank all doors by convenientDistance`
  - `what is the convenientDistance to all the doors`
- It also throws wacky operator-defined metrics at the station:
  - `ramesian = euclidean mod 5`
  - `convenientDistance = min(euclidean, manhattan)`
  - `nopeDistance = manhattan + 99` rejected by operator
  - `rammer = move forward then euclidean` refused as unsafe actuation leakage
- The eval must assert the first command becomes typed primitive-definition work,
  not plain unsupported text.
- The eval must assert JEENOM checks dependencies:
  - Manhattan ranked distance already exists.
  - Euclidean ranked distance is missing/synthesizable unless already built.
- The eval must assert operator approval gates registration.
- The eval must assert the new handle is registered only after validation.
- The eval must assert future planning can use the new metric without reverting
  to `manhattan` or `euclidean`.

Required schema/message additions:

- Add a typed primitive-definition intent/request. Possible shape:
  `PrimitiveDefinitionRequest`.
- Minimum fields:
  - `definition_type`: e.g. `distance_metric`
  - `name`: operator-facing metric/primitive name such as `convenientDistance`
  - `normalized_name`: registry-safe name such as `convenient_distance`
  - `expression`: structured formula such as `min(euclidean, manhattan)`
  - `dependencies`: handles or metric names used by the expression
  - `proposed_handle`: e.g.
    `grounding.all_doors.ranked.convenient_distance.agent`
  - `safety_class`: must be query-only for this slice
  - `authority_level`: `operator`
  - `provenance`: operator utterance, approval turn, dependency handles

Required control-flow behavior:

1. Compiler/semantic parser detects operator-defined primitive requests.
2. Cortex/station builds a typed definition request and RequestPlan.
3. Readiness checks dependency availability and synthesis requirements.
4. If dependencies are missing but safe, JEENOM materializes the query-only
   dependency during approval.
5. Once dependencies exist, JEENOM registers the composed primitive.
6. Operator approves or rejects.
7. The primitive assembler generates a pure query callable from the structured
   expression; arbitrary code is not accepted.
8. Validator runs deterministic fixtures.
9. CapabilityRegistry registers the primitive only after validation.
10. OperationalContext/PlanningSemantics can resolve the new metric name.
11. The knowledge surface records the primitive definition, dependencies,
    provenance, validation result, and handle via a memory-write ticket.

Acceptance criteria:

- The example `convenientDistance = min(euclidean, manhattan)` can be proposed,
  approved, validated, registered, and used in a later ranked-door query.
- Inline composition such as `go to the third farthest door based on the sum of
  both distance metrics` proposes the derived metric, registers it on approval,
  then resumes the original task through RequestPlan/ReadinessGraph.
- A rejected proposal registers nothing.
- A validation failure registers nothing and says so honestly.
- The new primitive is query-only and cannot authorize motion by itself.
- The new metric/handle appears in the capability registry and planning
  semantics after registration.
- Re-running `python evals/eval_master.py --suite cleanup`,
  `python evals/eval_master.py`, and `python -m pytest -q tests` stays green.

Measured outcome:

- `python evals/phase10i_user_defined_metric_probe.py`: passing.
- `python -m pytest -q tests/test_phase10i_user_defined_metrics.py`: 9 passed.
- `python evals/eval_master.py --suite cleanup`: 25/25 passing.
- `python evals/eval_master.py`: 54/54 passing.
- `python -m pytest -q tests`: 244 passed.

Phase 10 stop rule after 10I:

- After 10I, close Phase 10 and move to Phase 11.
- Do not add another Phase 10 slice unless a live operator outcome is missing
  from the core just-in-time primitive assembly story.
- `operator_station.py` may remain large if the blocking architecture boundaries
  are enforced.
- Repo/file-size minimization becomes a later cleanup phase after the prototype
  proves more capability.

## Phase 11 - Architecture Fix - Mission Flow

Status: in progress.

Goal: make the implemented control flow match the architecture before adding
new capability. Compound missions must be owned by Cortex, decomposed into typed
mission/procedure steps, satisfied through Sense and Spine contracts, and only
rendered by the Operator Station.

Problem to fix:

- `OperatorStationSession` still owns too much mission behavior.
- Inline derived metrics currently work, but the station can parse, propose,
  register, resume, and dispatch them itself.
- That makes the station act like Cortex, Sense coordinator, memory writer, and
  dispatcher.
- Compound requests can collapse into a flattened task such as "go to the
  yellow door" instead of preserving the full mission reason, evidence path,
  selection rule, and execution authority.
- This is architecture debt, not just file-size debt.

Required flow:

1. Operator Station receives the utterance and hands it to the cognition kernel.
2. Cortex produces typed intent and builds a `RequestPlan` / `MissionContract`.
3. Cortex decomposes compound work into architecture-level steps:
   derive/query primitive, gather evidence, bind claims, select target, execute.
4. Sense executes evidence requests and writes observation/grounding claims.
5. Knowledge Base records claims, primitive definitions, procedures, provenance,
   and approval state through typed APIs/tickets.
6. ReadinessGraph gates every step from claims and primitive availability.
7. Cortex issues `ApprovedCommand` / `ExecutionTicket` only after readiness.
8. Spine executes actuation through an `ExecutionContract`.
9. Runtime execution/render still makes zero LLM calls.

Eval-first requirements:

- Add a mission-flow eval for a compound request:
  "go to the third farthest door based on the sum of euclidean and manhattan
  distance".
- The eval must prove the resulting plan keeps the full mission structure:
  metric definition, evidence/ranking, ordinal selection, target claim, and
  actuation ticket.
- The eval must fail if Operator Station owns the decomposition or resumes the
  mission with station-local payloads.
- The eval must fail if execution starts without `RequestPlan`,
  `ReadinessGraph`, `ApprovedCommand`, and `ExecutionTicket`.
- The eval must fail if Sense/evidence work is represented only as a station
  phrase branch.
- The eval must fail if Spine receives a flattened task without mission
  provenance.
- Add focused unit tests around Cortex mission decomposition and ticket lineage.

Implementation requirements:

- Move compound mission decomposition out of `OperatorStationSession`.
- Keep station-local parsing only for REPL/facade concerns.
- Add or finish a typed mission-flow object only if existing `RequestPlan` and
  `MissionContract` cannot represent the flow without lossy flattening.
- Route primitive-definition approval through Cortex/command authority, not
  station-local continuation logic.
- Route evidence/ranking through Sense/claims, not display helpers.
- Route motion through Spine/ExecutionContract, not direct task dispatch.
- Preserve Phase 10I user-visible behavior while changing ownership.

Non-goals:

- Do not add a giant world model.
- Do not solve partial observability yet.
- Do not start repo liposuction inside this phase.
- Do not add new capability unless an eval proves it is required to enforce the
  mission flow.

Acceptance criteria:

- The station becomes a facade for mission turns, not the owner of mission
  decomposition.
- Compound user-defined metric tasks execute through Cortex -> Sense/claims ->
  ReadinessGraph -> ApprovedCommand/Ticket -> Spine.
- Ticket/result lineage can explain why the selected target was chosen.
- Runtime/render path remains LLM-free.
- Existing Phase 10I behavior remains green.
- `python evals/eval_master.py --suite cleanup`, `python evals/eval_master.py`,
  and `python -m pytest -q tests` remain green after the implementation.

### Phase 11A - Cortex-Owned Inline Mission Flow

Status: done.

Implemented in Phase 11A:

- Added `jeenom/mission_cortex.py` as the Cortex-owned mission-flow helper.
- Moved inline metric mission parsing, primitive-definition planning, and
  continuation intent construction out of `OperatorStationSession`.
- Extended `MissionExecutionPlan` with mission contract, primitive definition,
  continuation intent/plan/graph, provenance, and child ticket lineage.
- Extended `ExecutionTicket` with `mission_id`, `parent_request_id`, and
  provenance.
- Added `evals/phase11_mission_flow_probe.py` and
  `tests/test_phase11_mission_flow.py`.

Measured outcome:

- Inline compound metric requests no longer rely on station-local resume
  payloads.
- Approval preserves mission provenance into the final execution ticket.
- Existing Phase 10I operator-defined metric behavior remains green.

### Phase 11B - Hostile Primitive And Mission Eval Ladder

Status: current.

Goal: stop the eval suite from giving false confidence. The suite must test the
architecture ladder from low-level primitive invocation through procedure
assembly and compound missions, with paraphrase sweeps at every level.

This is eval-first work. Do not fix production code in 11B until the red bars
are visible.

What 11B must expose:

- Equivalent operator phrasing must route to equivalent typed intent, plan,
  readiness, claims, and tickets.
- Query/sense phrases must not be unsupported just because they use different
  surface language.
- Conditional actuation must not execute raw motor primitives before Sense
  produces evidence and Cortex evaluates the condition.
- Multi-action motor/procedure requests must preserve parent/child lineage, not
  only the final child action.
- Named procedural primitives must be stored through the representation surface
  and invoked through RequestPlan/ReadinessGraph, not through a chat-memory
  shortcut.
- Compound missions must keep MissionContract, evidence/ranking, selection,
  execution ticket, and provenance intact.

Hostile eval files:

- Added `evals/phase11b_primitive_ladder_probe.py`.
- Added `tests/test_phase11b_primitive_ladder.py`.
- Registered the probe in `evals/manifest.py`.

Low-level Sense paraphrase sweep:

- "what is in front of me"
- "what am I facing"
- "what object is ahead"
- "sense the cell in front"
- "look forward"

Required invariant:

- routes to a typed query/evidence plan
- writes or reads observation claims through Sense/representation
- does not issue `ExecutionTicket` or `RawMotorTicket`
- does not return unsupported/refuse for known senseable relations

Low-level Spine paraphrase sweep:

- "take one step forward"
- "move forward once"
- "advance one cell"
- "step ahead"
- "go forward one"

Required invariant:

- explicit low-level actuation creates a `RawMotorTicket`
- RequestPlan objective is `motor_control`
- ReadinessGraph next action is `execute_motor`
- no task-like object request leaks into raw motor authority

Named procedural primitive sweep:

- Teach:
  - "when I say fing fam foom, give me the distance to all doors"
  - "remember fing fam foom means give me the distance to all doors"
  - "define fing fam foom as give me the distance to all doors"
- Invoke:
  - "fing fam foom"
  - "do fing fam foom"
  - "run fing fam foom"

Required invariant:

- teaching creates a typed knowledge/procedure update plan
- representation snapshot records the procedure/macro, plan, and provenance
- invocation expands into the same ranked-distance RequestPlan
- no side-effect ticket is issued for query-only procedure invocation

Low-level action procedure sweep:

- "go straight two steps and turn left"
- "move forward twice then turn left"
- "advance two cells, then face left"

Required invariant:

- decomposes into a parent procedure/sequence plan
- preserves all child motor actions and counts
- each child action has ticketed authority or an explicit child-ticket record
- final result/provenance does not collapse to only the final action

Conditional Sense + Spine sweep:

- "if there is a red door in front of me, go forward, otherwise stay"
- "if I am facing a red door, step forward, else do nothing"
- "move only if the object ahead is a red door"

Required invariant:

- Sense runs first and records the front-cell/object claim
- Cortex evaluates the condition from claims
- Spine executes only if the condition is true
- if false, JEENOM reports a no-op/non-execution honestly
- no raw motor ticket is issued before the evidence condition is satisfied

Distance/query paraphrase sweep:

- "how far are all the doors from you"
- "how far are the doors from you"
- "how far away are the doors"
- "distance to the doors"
- "show me the door distances"
- "how far is each door"
- "what are the distances to the doors"

Required invariant:

- all route to the same ranked-door distance query
- required capability includes `grounding.all_doors.ranked.manhattan.agent`
- RequestPlan objective is `query`
- ReadinessGraph next action is `answer_query`
- response is not unsupported/refuse

Compound mission sweep:

- "find euclidean distance to all doors, find manhattan distance to all doors,
  then go to the third farthest by their sum"
- "go to the second highest door by max(euclidean, manhattan)"
- "create ramesian = euclidean mod 5 and go to the farthest ramesian door"

Required invariant:

- Cortex owns decomposition
- `MissionExecutionPlan` exists for compound execution
- generated primitive is query-only unless explicit actuation safety is proven
- continuation plan preserves rank/select/execute
- final ticket carries mission provenance
- runtime/render still reports zero LLM calls

Negative controls:

- "go to the door"
- "how far is the door from you"
- "pick up the red key"
- "move to the farthest door by walking randomly"

Required invariant:

- clarify, refuse, or block honestly
- no silent default target
- no unsupported raw motor or task execution
- no answer-only degradation for an actuation request

Acceptance criteria for 11B:

- The new evals fail on the current repo for named, architecture-level reasons.
- Failure output is metric-shaped JSON and names the violated invariant.
- Focused pytest covers the same hostile ladder with clear assertions.
- No production code is changed in 11B.

Current 11B red-bar findings:

- Low-level Sense paraphrases are unsupported.
- Some low-level Spine paraphrases are unsupported.
- Distance-query paraphrases only work when trigger words like "all" or "each"
  are present.
- Stateful references like "the doors" and "their distances" are not reliably
  bound to the visible set after a scene query.
- Named procedure teaching can store a concept without a typed memory-update
  RequestPlan, and "when I say ..." can produce a false query answer instead of
  storing the procedure.
- Multi-action motor/procedure prompts can execute while preserving only the
  final child action in the current plan/ticket state.
- Conditional actuation can execute raw motor motion before evidence proves the
  condition.
- Compound mission paraphrases beyond the exact 11A inline sum phrase do not
  consistently create `MissionExecutionPlan` lineage.
- Actuation requests with unsupported policy language can degrade into
  answer-only grounding responses.

After Phase 11B:

- Implement the smallest architecture fixes needed to make the hostile ladder
  green.
- Do not start Phase 12 evidence planning or repo liposuction until this ladder
  is green.

### Phase 11C - Architecture Surgery

Status: **complete**.

Goal: resolve seven compounding architectural violations in strict sequence,
each behind a red-bar eval and green suite checkpoint. Two absolute
prohibitions: no rewrite, no new features until all operations are done.

Operations completed (all 7):

**Op 1+2 (atomic) — Import partition + domain purge:**
- Deleted `OPERATOR_OBJECT_TYPES = ("door",)` from `schemas.py`.
- Added `register_domain_vocabulary()` / `clear_registered_vocabulary()` /
  `_validate_object_type()` to `schemas.py`. All `"door"` literal comparisons
  replaced with the registered vocabulary check.
- `GroundedDoorEntry` → `GroundedObjectEntry` across helpers and station.
- `minigrid_domain_helper.py` calls `register_domain_vocabulary(("door",))` at
  init; fail-open window is provably never live in production.
- `sense.py` reads color/object indices from the domain manifest, not a MiniGrid
  import. `PlanningSemantics` accepts `operational_context: Any` at construction.
- Probe: `substrate_static_architecture_probe.py`

**Op 3a — Mechanical extraction (branch-preserving):**
- `command_from_operator_intent` and all call-site machinery moved from
  `OperatorStationSession` into `TurnOrchestrator.dispatch`.
- `CortexSession` and `MissionCortex` references moved to TurnOrchestrator.
- Station is a dialogue shell; no planning logic.
- Probe: `pipeline_turn_orchestrator_probe.py`

**Op 3b — Knowledge-type rerouting:**
- 20-branch `intent_type` if/elif chain in `TurnOrchestrator.dispatch` collapsed
  to 5 paths keyed on `OperatorIntent.knowledge_type` computed property.
- Paths: `_handle_claim` / `_handle_procedure` / `_handle_provenance` /
  `_handle_action` / `_handle_control`.
- Probe: `pipeline_dispatch_probe.py`

**Op 4 — Demote shadow NLU into IntentCache:**
- New module `jeenom/intent_cache.py`:
  - `IntentCache` dataclass with `register()`, `_register_compiled()`,
    `lookup()`.
  - Exported: `SEQUENCE_STEP_PREFIX`, `SEQUENCE_STEP_SUFFIX`,
    `parse_metric_query()`, `seed_intent_cache()`.
- `classify_utterance` has zero `re.compile` calls. Fast-path intents (prim_def,
  concept_teach, concept_forget) run IntentVerifier + dispatch identically to
  LLM-compiled intents.
- `metric_query` and `delivery_target` kept in `classify_utterance` fast path
  (bypass dispatch — no registered primitives for these types).
- Probe: `intent_fidelity_cache_probe.py`

**Op 5 — One readiness, one cortex:**
- `class Readiness` deleted from `jeenom/cortex.py`.
- `self.readiness = Readiness(memory)` and `self.readiness.check()` removed from
  `Cortex.__init__` and `Cortex.onboard_task`.
- `Cortex.onboard_task` returns a minimal always-executable `ReadinessReport`
  (`status="executable"`); the real gate is `ReadinessGraph` upstream in
  `TurnOrchestrator.dispatch`.
- `Readiness` removed from `jeenom/__init__.py` and `__all__`.
- `MissionCortex` confirmed one-directional (no imports of Cortex, CortexSession,
  or TurnOrchestrator).
- Probe: `substrate_cortex_invariant_probe.py`

**Op 6 — Re-key evals from archaeology to contract:**
- 55 eval files renamed via `git mv` to capability-based names.
- `evals/manifest.py` rewritten with files grouped by prefix.
- `eval_naming_contract_probe.py` enforces the naming contract going forward.
- Approved prefixes: `authority_`, `claim_custody_`, `intent_fidelity_`,
  `pipeline_`, `regression_`, `repair_`, `substrate_`, `synthesis_`.
- `eval_phase_3_5.py` → `regression_jit_prewarm_probe.py` (kept — live JIT test).
- `eval_phase_9e.py` → `claim_custody_representation_probe.py` (kept — tests
  ClaimRecord/ReadinessGraph/representation store).
- Probe: `eval_naming_contract_probe.py`

**Op 7 — Land hardware schema fields:**
- `ClaimRecord.valid_until: float | None = None`
- `PrimitiveSpec.postcondition_primitive: str | None = None`
- `MissionContract.risk_tier: str = "low"` and `cadence: str | None = None`
- `FailureOutcome` dataclass (`category`, `detail`, `blocking_claim_handle`)
- `CommandResult.failure_outcome: FailureOutcome | None = None` (via `__new__`)
- Probe: `substrate_hardware_schema_probe.py`

Measured outcome:

- `python evals/eval_master.py`: 60/60 passing.
- All 7 operations landed with zero regressions.

## Phase 12 - Minimal Representation And Evidence Planning

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

## Phase 13 - Cross-Substrate Demonstration

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

## Phase 14 - ARC-Style Steerable Prototype

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

## Phase 15 - Operational Hardening

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

## Phase 16 - Capability Stress Tests

Status: later.

Use harder MiniGrid, real robotics, or ARC-like tasks only as architecture
pressure tests. The goal is not benchmark chasing; the goal is to expose missing
primitive contracts, missing claims, missing evidence, bad decomposition, and
bad steering.
