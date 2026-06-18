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

- Current phase: **Phase 13 - Steerable Cognition Layer** (**13A complete**, incl. **13A.1**
  coordinate-system abstraction and the **13A.2** cleanup spike — all 4 slices landed
  (fingerprint, capability-handle grammar, metric-name detection, typed TurnState).
  **13B in progress**: 13B.1 claim freshness, 13B.2 MiniGrid FOV, 13B.3 `needs_evidence`,
  and 13B.4 eval-pipeline + tool-call discipline (deterministic gate + opt-in `live_llm`
  suite; root-caused and fixed a silent LLM→regex fallback from a `max_tokens` truncation)
  are complete; the remaining 13B work is the meta-primitive / `search_allowed` decision).
  13A delivered the typed, constraint-first steering layer
  (`SteeringDirective`: budget/scope/risk/stopping-rule) that demonstrably reshapes plan
  assembly — risk withdraws actuation authority via `needs_authorization`, budget caps
  the Spine stepping loop as `FailureOutcome(category="budget_exhausted")`, and the active
  directive is recorded in `LabelledEpisode.steering`. Proven by the hostile
  `pipeline_steering_directive_probe.py`; 71/71 evals green.
  Phase 12D is complete (12D.1–12D.4): ORPI v0.1 label unified, `LabelledEpisode`
  attribution + verification wired (ORPI taxonomy, named checker path), coupling
  audit produced, curriculum-touching leaks removed, two cheap non-curriculum-touching
  leaks removed early (see 12D.4).
  Phase order from here: **13 Steerable Cognition (proves steering on MiniGrid) -> 14
  Cheap Leak Removal + AI2-THOR spike (proves substrate-independence) -> 15
  Cross-Substrate & v1 Freeze -> 16 Operational Hardening (absorbs the deferred
  station de-bloat) -> 17 Capability Stress**.
  Steering and substrate-independence are validated on separate substrates so
  neither signal masks the other. `operator_station.py` bloat is orthogonal to
  both proofs and is parked for Phase 16; any station extraction is gated by a
  reviewed decomposition design first.
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
- Phase 11B is complete. Hostile primitive/mission evals now prove paraphrase
  stability, typed procedure teaching, conditional Sense-before-Spine gating,
  multi-action lineage, and compound mission provenance.
- Phase 11C is complete. Seven compounding architectural violations resolved
  (import partition, domain purge, TurnOrchestrator dispatch extraction,
  knowledge-type rerouting, IntentCache, Readiness deletion, eval naming
  contract, hardware schema fields). See Phase 11C section below.
- Phase 12 is complete for MiniGrid ORPI v0.1: `OrpiContract`,
  `OrpiProcedure`, `OrpiManifest`, `LabelledEpisode`, compatibility mapping from
  legacy primitive layers to `{sense, actuation, meta}`, bundled-procedure
  validation, MiniGrid manifest metadata, knowledge scoping, JSON-serializable
  labelled episodes, and the `orpi` eval suite are green.
- Current verification signal:
  - `python evals/eval_master.py`: 78/78 passing (deterministic gate; runs with the
    live-LLM key stripped, so it is reproducible and offline)
  - `python evals/eval_master.py --suite orpi`: 10/10 passing
  - `python evals/eval_master.py --suite cleanup`: 30/30 passing
  - `python evals/eval_master.py --suite llm_path`: 5/5 passing
  - `python evals/eval_master.py --suite live_llm`: 1/1 passing (opt-in; REAL model calls,
    skips when `OPENROUTER_API_KEY` is unset — NOT part of the gate)
  - `python -m pytest -q tests`: 298 passed, 1 warning, 9 subtests passed
- Eval naming contract is now enforced: all registered eval files use
  capability-based prefixes. The naming contract probe fails on violation.
- Whole-repo `pytest` is not the main project signal right now because the local
  `Minigrid/` tree can introduce unrelated dependency noise.
- **Threat model — KNOWN LIMITATION (good-faith operator assumed).** JEENOM currently
  assumes a non-adversarial operator and a non-adversarial LLM backend. The architecture
  *contains* a misbehaving LLM for the dangerous cases (typed tool-call outputs only,
  enum-validated decision fields, unknown primitives rejected, side effects gated by
  station-minted tickets + `IntentVerifier` + `ReadinessGraph`, no LLM in the render loop),
  but this is **not yet hardened or proven against hostile prompts / prompt injection /
  jailbreaks**, and one dispatch field (`grounding_query_plan.answer_fields`) is still an
  open string list (it fails *safe* today — silent dead-end, no side effect). Adversarial
  hardening is deferred to **Phase 17** ("Adversarial robustness & hostile-input hardening").
  Do not deploy to untrusted operators or feed it untrusted text until that lands.

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
- Phase 13 evidence planning should build on this capability, not work around
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

Status: complete.

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

- The hostile evals fail before fixes for named, architecture-level reasons.
- Failure output is metric-shaped JSON and names the violated invariant.
- Focused pytest covers the same hostile ladder with clear assertions.
- Production fixes are scoped to the violated invariants and keep the existing
  eval/test suite green.

Resolved 11B red-bar findings:

- Low-level Sense paraphrases route to query/evidence plans without motion.
- Low-level Spine paraphrases issue `RawMotorTicket` authority.
- Distance-query paraphrases route to ranked-door distance queries, including
  stateful references after scene queries.
- Named procedure teaching records typed memory-update plans and representation
  procedures.
- Multi-action motor/procedure prompts preserve child action/count lineage.
- Conditional actuation blocks raw motor movement until evidence/condition flow
  is represented.
- Compound mission paraphrases create or reuse `MissionExecutionPlan` lineage.
- Unsupported actuation policy language blocks instead of degrading to
  answer-only grounding responses.

After Phase 11B:

- The hostile ladder is green and Phase 12 proceeded.
- Phase 12 ORPI contract/manifest/trace boundary is stable for MiniGrid; Phase
  13 evidence planning may proceed from the ORPI-v0.1 surface.

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

## Phase 12 - ORPI v0.1 - Open Robotics Primitive Interface

Status: done for MiniGrid ORPI v0.1. Authoritative spec:
`PlanOfAction/orpi_spec.md`.

Goal: extract the typed interface standard between the cognition layer (JEENOM)
and an embodiment - "MCP for robot cognition." This is deliberately a v0: it is
extracted from n=1 substrate (MiniGrid) and exists to be broken by the second
substrate during the Phase 15 cross-substrate port. Phase 12 started after the
Phase 11B hostile ladder went green.

ORPI has two halves:

- Inbound - what a substrate exposes to cognition. Every capability is published
  as a primitive with a machine-readable contract. Cognition sees contracts, not
  hardware/drivers/policies.
- Outbound - what a deployment emits to learning. Every executed turn produces a
  `LabelledEpisode` (intent, grounding, plan, authority, execution, verification,
  attribution, steering). This is the supervision artifact for self-improving
  deployment loops; no other part of the robotics stack produces it by
  construction.

The grounding obligation is discharged at the primitive boundary. The cognition
layer above provides grounding accounting (custody, validity, arbitration,
composition), never grounding itself.

### Primitive classes and the contract

- Every primitive declares exactly one class: `sense` (reality -> claims),
  `actuation` (approved command -> physical effect), `meta` (claims -> claims).
- `meta` primitives declare `mode: deterministic | deliberative`. Compiled plans
  may never reference a `deliberative` meta-primitive - this is the enforcement
  point for the no-LLM-in-the-loop invariant.
- The contract is the existing `schemas.PrimitiveSpec`, serialized. It already
  carries `preconditions`, object-centric `postconditions` (Δg), the
  `postcondition_primitive` checker, `required_claims`/`produced_claims`,
  `units`/`frame_id`/`required_frames`, `safety_class`, `authority_level`,
  typed `failure_modes`, `validation_hooks`, and `substrate_fingerprint`.
  See `orpi_spec.md` §4 for the full field table.

### Taxonomy migration (compat bridge first)

- ORPI contract serialization remaps `schemas.PrimitiveSpec.primitive_type` from
  `{task, grounding, sensing, action, claims}` to `{sense, actuation, meta}`.
  5 -> 3 mapping: `sensing -> sense`, `action -> actuation`, and
  `task | grounding | claims -> meta`.
- The first implementation keeps legacy layer names valid for registry grouping
  and prompt summaries while exposing the ORPI taxonomy through `OrpiContract`.
- Central implementation risk: two `PrimitiveSpec` classes exist - the contract
  type in `schemas.py` and the frozen runtime type in `primitive_library.py`
  (`runtime_kind`/`runtime_value`, used by the
  `TASK/GROUNDING/SENSING/ACTION/CLAIMS_FILTER_PRIMITIVES` registries). "Every
  capability registered through a contract" (conformance item 1) requires
  bridging these two, not just renaming a string set.

### Implemented Phase 12 surface

1. `jeenom/orpi.py`: serializable `OrpiContract`, `OrpiManifest`,
   `OrpiProcedure`, JSON-serializable `LabelledEpisode`, plus a
   no-deliberative-meta plan-reference helper.
2. `schemas.PrimitiveSpec`: `mode`, `cadence`, and `invariant_level` fields with
   MiniGrid-compatible defaults.
3. MiniGrid ORPI manifest metadata: `symbol_mappings`, `frames`, `units`, and
   `risk_policy` are published from `MiniGridOperationalContext`.
4. `CapabilityRegistry.minigrid_default()` remains the legacy registry source,
   while `MiniGridSubstrateAdapter.orpi_manifest()` publishes the ORPI contract
   view over every registered capability.
5. `LabelledEpisode` emission is attached to every `CommandResult` through
   `CommandAuthority.record_result`.
6. `OrpiManifest.bundled_procedures` carries OEM-vouched procedure recipes.
   Manifest validation rejects synthesized/operator procedures in bundled OEM
   slots and fails if a bundled step references a primitive missing from the
   same manifest.
7. `CapabilityRegistry` can surface primitive and procedure candidates by
   declared postcondition, and `LabelledEpisode.plan.candidates` records whether
   a selected candidate was a primitive or procedure plus its provenance.
8. `NamedConcept.scope` and `derive_scope()` give durable knowledge records a
   falsifiable transfer scope: `site`, `embodiment`, or `universal`;
   `episodic` remains outside the KB.
9. `KnowledgeChannel` is the gated KB access surface for station/orchestrator
   paths. It enforces writer-by-scope policy, emits KB writes into
   `LabelledEpisode.steering.knowledge`, and records per-scope reuse counters.
10. `PlanOfAction/orpi_spec.md` remains the versioned ORPI-v0.1 reference.

### Conformance

A substrate is ORPI-v0.1 conformant when (1) every capability is registered through
a contract - no side channels; (2) the manifest is registered at init; (3) all
postconditions are object-centric deltas and every actuation primitive with
`safety_class != query` names a `postcondition_primitive`; (4) cadence
declarations are honoured (no deliberation-cadence call on a Spine path); (5)
every executed turn emits a `LabelledEpisode`; (6) no compiled plan references a
`deliberative` meta-primitive.

### ORPI conformance probes

These probes are registered in `evals/manifest.py` under the `orpi` suite and
pass against the completed MiniGrid ORPI-v0.1 boundary.

- `substrate_orpi_contract_coverage_probe.py` - conformance 1: every capability
  is registered through a contract; no side-channel capabilities.
- `substrate_orpi_manifest_registration_probe.py` - conformance 2: manifest is
  registered at adapter init; the permissive validation window is provably never
  live; bundled procedures and manifest risk policy are validated.
- `substrate_orpi_postcondition_probe.py` - conformance 3: all postconditions are
  object-centric Δg; every actuation with `safety_class != query` names a
  `postcondition_primitive`.
- `substrate_orpi_cadence_probe.py` - conformance 4: data check over manifest
  contracts - no actuation/sense contract has wrong cadence; no control-cadence
  contract is deliberative.
- `substrate_orpi_no_llm_in_loop_probe.py` - conformance 6: exercises the
  runtime enforcement gate in `CortexSession.plan` - a synthetic deliberative
  plan raises `SchemaValidationError`.
- `pipeline_orpi_labelled_episode_probe.py` - conformance 5: full success,
  refusal, and task turns round-trip to valid, JSON-serializable
  `LabelledEpisode` artifacts with verification evidence.
- `pipeline_procedure_selection_probe.py` - procedure parity: primitives and OEM
  procedures are selectable by postcondition, selected procedures expand to
  primitive handles, and readiness is not bypassed.
- `claim_custody_knowledge_scope_probe.py` - scope custody: named concepts
  carry transfer scope, persisted legacy records migrate to `site`, and
  `derive_scope()` distinguishes universal/effect-only recipes from
  embodiment-bound recipes.
- `regression_orpi_primitive_type_migration_probe.py` - the taxonomy remap leaves
  the existing eval suite green (migration safety net).

### Acceptance criteria

- MiniGrid is ORPI-v0.1 conformant against all eight conformance items.
- The three NEW fields exist with degenerate MiniGrid defaults; no other
  substrate semantics leak into the schema.
- Every executed turn round-trips to a valid `LabelledEpisode`, failed episodes
  included with attribution, and serializes as JSON.
- Known v0 proxy (closed in Phase 12D, not 13): `LabelledEpisode.verification`
  reflects `task_complete` / boolean claims, not `postcondition_primitive`
  invocation against predicted Δg. `LabelledEpisode.attribution` passes through
  `FailureOutcome.category`, not the ORPI attribution taxonomy. Phase 12D lands
  the attribution-taxonomy mapping and the `postcondition_primitive` verification
  wiring for the named-checker case; rich Δg verification that depends on the
  derived-claim layer completes in Phase 13.
- The taxonomy compatibility layer lands with zero regressions on the existing
  eval suite.
- OEM procedure bundling is schema-backed without adding a parallel recipe
  hierarchy.
- Knowledge transfer scope is explicit and write-gated before Phase 13 consumes
  per-scope reuse metrics.
- `universal` scope is a correct but degenerate case for MiniGrid: all compiled
  plans reference concrete primitive names (`action.turn_left`, not
  `"agent direction changes"`), so no real MiniGrid plan will ever derive
  `universal` scope. `derive_scope` returns `universal` only when every
  `required_handle` in the plan matches a postcondition/output string in the
  manifest's effect vocabulary — i.e., the plan is expressed by effect, not by
  primitive name. That representation requires the cross-substrate planner in
  Phase 15. The `claim_custody_knowledge_scope_probe.py` constructs a synthetic
  plan to demonstrate the mechanism exists; the transfer fixture in 12C.3 tests
  `site` vs `embodiment` invalidation only. Phase 15 provides the first real
  `universal`-scope concept.

## Phase 12D - Consolidation + Leak Audit

Status: complete.

Goal: close ORPI v0.1 cleanly and produce the audit that decides the order of the
phases that follow. No new capability, no architecture redesign. Three workstreams,
each behind the green suite.

### 12D.1 - Doc fixes

- Unify the version label: the interface is **v0.1** everywhere (code already
  shipped procedures and knowledge scopes beyond the original v0 spec). Reconcile
  `orpi_spec.md`, `blueprint.md`, and `task_plan.md`.
- Fix `workflow_diagram.mmd`: the `LabelledEpisode` emission node is wired to the
  KnowledgeBase, but emission happens in `CommandAuthority.record_result`. Repoint
  it.
- Parity-check the three docs: §4 field table, §7/§9 conformance list, probe
  names, and phase numbers must agree.

### 12D.2 - LabelledEpisode completion

- `attribution`: map `FailureOutcome.category` -> the ORPI attribution taxonomy
  (`stale_claim | miscompiled_intent | unmet_postcondition | missing_authority |
  substrate_fault`). Pure mapping, no dependency on the derived-claim layer.
- `verification`: invoke a contract's `postcondition_primitive` against its
  predicted Δg where one is named; keep the boolean only for MiniGrid's degenerate
  `postcondition_primitive=None` case and label it as such. Rich Δg verification
  that needs the derived-claim layer is explicitly left to Phase 13 - it is not
  faked here.
- Red bar: extend `pipeline_orpi_labelled_episode_probe.py` to assert the ORPI
  attribution taxonomy is used and that a named `postcondition_primitive` is
  actually invoked.

### 12D.3 - Leak audit (the ordering-relevant deliverable)

Catalog every place MiniGrid/domain vocabulary is baked into generically-named
code. The audit separates **two orthogonal problems** and treats them differently:

**Substrate coupling (the boundary - on the critical path).** Per coupling site,
record two axes:

- `{cheap | structural}` - is removal a surgical edit, or does it require real
  rework (e.g. replacing `llm_compiler.py`'s hardcoded `go to the <color> door`
  grammar with `OperationalContext`-derived patterns)?
- `{curriculum-touching | not}` - **does the Phase 13 steering/PO work build on
  top of this leak?**

The product table `{cheap|structural} x {curriculum-touching|not}` decides phase
order:

- **Curriculum-touching leaks are removed before Phase 13**, regardless of
  cheap/structural. The curriculum must not be built on top of a leak it would
  then entrench. This is the one ordering question not yet settled - the audit
  decides it, not an assumption that the curriculum can safely precede leak
  removal.
- **Non-curriculum-touching leaks stay in the Phase 14 cheap-removal pass** as
  planned.
- Any leak flagged `structural` is a go/no-go flag for its phase's scope.

Known concentrations to classify (from the pre-audit survey, to be confirmed):
`llm_compiler.py` (~116 lines, real fast-path grammar - the prime structural
suspect and the leak most likely to block the spike), `operator_station.py` (~67
lines, mostly live paths), `primitive_library.py` (~33, the MiniGrid primitive
set in a generic file), `sense.py`/`spine.py` (`MiniGridSense`/`MiniGridSpine`
role bindings), `request_planner.py` (`rank_scene_doors`).

**Structural bloat (orthogonal - parked).** The `operator_station.py` size
(~5,613 lines / 178 methods, of which only ~67 are substrate-coupled) is a
*different* problem. It blocks neither proof - steering or substrate-independence
- so it does **not** compete for position in the phase order. The audit still
catalogs it as the worklist for the eventual cut, but **de-bloating is deferred to
Operational Hardening (Phase 16) in full**. Wanting the file smaller is not a
reason to take on a 5,600-line refactor ahead of either proof.

**Hard prerequisite:** any `operator_station.py` extraction - whenever it happens,
in Phase 16 or pulled forward by a `structural` + `curriculum-touching` audit
verdict - is gated by a decomposition design (target modules, shared-state map,
ordered green-able extraction sequence) written and reviewed *first*. No station
code moves before that design exists.

### 12D.4 - Cheap non-curriculum-touching leaks (pulled forward)

Two `cheap × not curriculum-touching` items from the Phase 14 table removed early
since the boundary was already open and the edits were surgical:

- `primitive_library.py` door grounding primitives (`visible_doors`, `closest_door.*`,
  `unique_door.*`, `all_doors.ranked.*`) moved to `minigrid_primitive_library.py`.
  `capability_registry.py` merges them in `minigrid_manifest_dict()`. `capability_registry.py`
  removed from the static-architecture probe's substrate-neutral file list (it already owns
  `minigrid_default()` and was never neutral).
- `request_planner.py` `rank_scene_doors` step_id: was hardcoded; now
  `f"rank_scene_{semantics.pluralize(object_type)}"` — derived from the manifest's object
  vocabulary. For MiniGrid `object_type="door"` this yields `"rank_scene_doors"` unchanged.

### Acceptance criteria

- Version label is `v0.1` across all docs; the diagram emission node is correct.
- `LabelledEpisode` uses the ORPI attribution taxonomy and invokes named
  `postcondition_primitive` checkers; the deferred rich-Δg case is documented.
- Coupling table populated (see Phase 14); curriculum-touching items removed;
  bloat worklist parked in Phase 16.
- Full eval suite green; golden path preserved.

## Phase 13 - Steerable Cognition Layer

This phase proves **steering** on MiniGrid only - substrate-independence is a separate
proof, validated later on the AI2-THOR spike. The two are deliberately not conflated.
The make-or-break: the operator tells JEENOM *how* to approach a task and JEENOM
assembles a typed plan accordingly. It is split so value lands early (13A) before the
heavier substrate work (13B) and the longitudinal proof (13C).

Steering is **constraint-first**: a typed, separable `SteeringDirective`
(budget/scope/risk/stopping-rule), following the `SelectionObjective` pattern (typed
fields, enum validation, vocabulary only in prompts/parsers). Decomposition/method
guidance layers on top afterward.

Minimum loop (carried across the sub-phases):

1. Operator gives WHY.
2. JEENOM turns it into WHAT.
3. JEENOM checks claims it already has.
4. If evidence is missing, JEENOM says what is unknown and how it could find out. (13B)
5. Operator steers budget/scope/risk. (13A)
6. JEENOM executes sensing/action through substrate HOW.
7. JEENOM updates claims and tries the plan again.

### 13A - Steering core (constraint steering reshapes plan assembly)

Status: **complete**. Same WHAT + different typed steering => different typed
`RequestPlan`/`ReadinessGraph`; a conflicting directive clarifies/refuses instead of
executing. Proven by the hostile `pipeline_steering_directive_probe.py` (eval-first;
71/71 green).

Delivered:

- `SteeringDirective` schema (`budget`/`scope`/`risk`/`stopping_rule`), enum-validated,
  with `STEERING_SCOPES` / `STEERING_RISK_LEVELS` / `STEERING_STOPPING_RULES` and the
  `STEERING_RISK_ALLOWED_SAFETY` map; `OperatorIntent.steering_directive` +
  `RequestPlan.steering`. New `intent_type="steering_directive"`.
- `steering_parser.py` is the single home for steering regex (no inline shadow NLU): it
  splits the HOW off the WHAT at turn entry; the residual drives the normal compiler and
  `IntentVerifier.enrich` (the gate) validates coherence and attaches the directive only
  to action intents — a misparse cannot silently shape a plan. Standalone steering turns
  are acknowledged (mid-plan steering of a *pending* plan is deferred to 13B).
- `build_request_plan` folds the directive into actuating steps' constraints
  (`steering_risk`/`steering_budget`/`steering_stopping_rule`) and records plan-level
  `steering`.
- Readiness enforcement: a `risk` that does not authorize a step's `safety_class` is an
  authorization withdrawal — reuses the existing `needs_authorization` status (no new
  status), so `query_only` on a go-to task refuses with no ExecutionTicket.
- Budget enforcement lives in the stepping loop (`MiniGridSpine.execute_plan`), not the
  authorization gate: cumulative env actions are capped; exceeding `max_steps` halts and
  surfaces `FailureOutcome(category="budget_exhausted")`, mapped to `missing_authority`
  in the ORPI attribution taxonomy (not the default `substrate_fault`).
- `LabelledEpisode.steering` now carries the active typed directive (previously passive).

Deferred within 13A: decomposition/method guidance reuses the existing
`concept_teach` / `sequence_instruction` / `mission_contract` rails; the richer
"compose primitives from a described method" version depends on the 13B `meta` layer.

#### 13A.1 - Coordinate-system abstraction (bugfix, AI2-THOR unblock)

Status: **complete**. Surfaced by the AI2-THOR branch: a continuous 3D substrate
needs float, N-dimensional coordinates, but coordinates were hardcoded `int` 2D and
distance math was hand-rolled (`obj.x - agent_x`) across ~16 files. This is
*coordinate* debt, distinct from (and not addressed by) the parked `operator_station.py`
de-bloat — only ~9 of the station's lines were coordinate-coupled.

Delivered (every live distance computation now flows through one home — no hand-rolled
coordinate math remains, including in synthesized primitives):

- New `jeenom/geometry.py`: the single home for coordinate metrics. `manhattan`/
  `euclidean` are N-dimensional (`zip` over coord tuples → 2D or 3D, no special-casing);
  `as_coord()` keeps integral coords `int` (MiniGrid display/equality unchanged) while
  preserving genuine floats. Pure stdlib — importable by `schemas` with no cycle.
- `schemas.py`: `SceneObject`/`SceneModel`/`GroundedObjectEntry` coords `int → float`;
  optional `z`/`agent_z` (absent on 2D substrates) with `coord`/`agent_coord` properties;
  `manhattan_distance_from_agent` delegates to geometry; `from_world_model_sample` stops
  truncating via `int(...)` (uses `as_coord`, reads optional `z`); `distance_value` widened.
- `operator_station.py`: the inline euclidean now calls `geometry.euclidean`.
- Synthesized-primitive path made geometry-backed and 3D-ready: `primitive_validator`
  pre-injects `geometry` into the runtime exec namespace; the synthesizer's canonical
  example and both synth/arbitrator prompt API docs now use
  `geometry.manhattan(d.coord, scene.agent_coord)` and advertise float coords + `.coord`/
  `.agent_coord`, so a synthesized metric is no longer silently 2D-only on a 3D substrate.
- `tests/test_geometry_coordinates.py` (eval-first): 2D/3D metrics, float preservation,
  int-collapse, schema integration, and the MiniGrid integral-path-unchanged contract.

Two latent bugs from the 13A base commit fixed in passing (both surfaced by the full
suite, neither caused by this work): `OperatorStationSession.active_steering_directive`
is now initialized in `__init__` (was only set in `handle_utterance`, so direct-dispatch
paths hit `AttributeError`); `test_phase12_orpi` now asserts `orpi_version == "0.1"`
matching the 12D v0.1 unification (was a stale `"0"`).

Verification: full `pytest -q tests` 273 passed / 0 failed, `eval_master` all green
(`--suite cleanup`, `--suite orpi` included).

#### 13A.2 - Cleanup spike: centralize scattered logic + fragmented turn state

Status: **complete — all 4 slices landed.** 13A.2.1 ✅, 13A.2.2 ✅ (re-scoped: control-plane
part; `llm_compiler` left to Phase 14), 13A.2.3 ✅ (scoped: detection consolidated
behavior-preserving; behavior-laden defaults/supported-lists left untouched), 13A.2.4 ✅. A
spike, so spike discipline applied: per slice, a red-bar probe landed **before** the
consolidating edit; full `eval_master` + `pytest` green between slices (final: `pytest`
290 passed, all eval suites green). Ordered lowest-blast-radius first so value landed early.

Motivation: 13A.1 exposed a debt *class* — a single concept hand-rolled across many files,
each copy free to drift (the int-truncation / 2D-only bug). An audit found more of the same
shape, plus per-turn state managed as 25 loose session attributes (the `active_steering_directive`
`AttributeError` was one symptom). This spike pays down that class. Scope is **concept
consolidation** (one concept → one home) and **state consolidation** — explicitly **not** the
parked `operator_station.py` file-decomposition (Phase 16) and **not** substrate-vocabulary
deleaking (Phase 14/15).

##### 13A.2.1 - Stable-fingerprint helper (quick, contained) — ✅ done

Delivered `jeenom/fingerprint.py` (`canonical_json` with a `sort_keys` flag —
`plan_semantic_key` legitimately needs `sort_keys=False`; `stable_hash`; `fingerprint`).
All four sites delegate; `import hashlib` dropped from the three hashing sites.
Behavior-preserving (verified byte-for-byte). Probe: `tests/test_fingerprint_consolidation.py`
(red-bar AST guard: no site calls `hashlib.sha256` directly).

- Problem: `mission_cortex.py`, `plan_cache.py`, `plan_reuse.py`, `schemas.py` each roll their
  own `hashlib` + `json.dumps(sort_keys=...)` fingerprint.
- Fix: one `stable_fingerprint(obj) -> str` home; the four sites delegate. Behavior-preserving
  (identical digest inputs → identical digests).
- Red-bar: probe/test asserts the four sites produce digests equal to the shared helper for
  representative inputs; fails while any site still hand-rolls hashing.

##### 13A.2.2 - Capability-handle grammar (route through the home that already exists) — ✅ done (re-scoped)

Delivered, but **re-scoped on contact with the code**: the handle debt is more entangled
with the Phase 14 door-leak than the original plan implied. `llm_compiler.py` holds no
`PlanningSemantics` instance and is the MiniGrid compiler *profile* (Phase 14), so it stays
out. This slice routed the cleanly-separable part — `operator_station.py`'s 6 construction
sites (3 f-strings + 3 bare literals) — through `self.planning_semantics.capability_handle(...)`
(the non-validating composer; `ranked_handle` would have gated synthesized metrics). The
MiniGrid context templates are byte-identical to the old literals, so cache keys / lookups
don't shift. Probe: `tests/test_handle_grammar_consolidation.py` (byte-for-byte equivalence +
AST guard: no handle f-strings in `operator_station.py`). `llm_compiler`'s ~39 handle strings
remain Phase 14 work.

- Problem: ~13 files hand-build handle strings (`grounding.all_doors.ranked.manhattan.agent`
  hardcoded 8×, plus `f"...ranked.{metric}.agent"` variants), **bypassing**
  `PlanningSemantics.{ranked_handle,filter_handle,capability_handle}` — which already exists and
  is substrate-parameterized (`{object_type_plural}`). This is the 13A.1 pattern exactly.
- Fix: route every generic-core handle *construction* through `PlanningSemantics`; delete the
  hand-built literals/f-strings. Handle *comparison/parsing* is untouched (`metric_from_grounding_handle`
  stays the parse home).
- Probe (`substrate_*`): fails if a raw `"grounding.`/`"claims.`/`"task.` handle literal or
  f-string is constructed outside `planning_semantics.py`, the `*_operational_context`, or the
  capability registry.
- Blast radius: **medium** — handles are string-*compared* in many places; only construction is
  centralized. Per-site review, not blind replace.
- Interplay note: because the home takes `object_type`, this shrinks the Phase 14 door-leak
  surface as a side effect — but full vocabulary deleaking stays Phase 14; this slice only moves
  construction onto the existing parameterized home.

##### 13A.2.3 - Distance-metric-name handling (control plane only) — ✅ done (behavior-preserving, scoped)

Delivered, scoped to the one genuinely behavior-preserving consolidation: text **detection**.
`semantic_normalizer._detect_metric` is now the public `detect_metric` + `mentions_metric`,
with the known-metric SET sourced from the canonical `OPERATOR_DISTANCE_METRICS` and the
**euclidean-first** priority preserved exactly (ambiguous text mentioning both metrics still
resolves to euclidean — pinned by `tests/test_metric_vocabulary_consolidation.py` so a
manhattan-first regression is caught). `operator_station`'s two inline euclidean-first detection
sites route through these (`mentions_metric` for "any metric mentioned"; `detect_metric(...) or
fallback` for the pick).

Left untouched **on purpose** (behavior-laden, not duplication): the divergent per-site default
metrics (filter→`euclidean`, ranked→`manhattan`), the deliberately-narrow supported lists
(`["manhattan"]`), the `metric == "manhattan"` compute-branch, the plan-mismatch logic, and
`intent_verifier`'s own PlanningSemantics-driven detector (the substrate-neutral path). Bare
`"manhattan" in normalized` substring checks were *not* routed — they are not equivalent to
`detect_metric == "manhattan"` under euclidean-first. Full substrate-neutral metric names
(PlanningSemantics-driven per substrate) remain Phase 14/15.

- Problem: `OPERATOR_DISTANCE_METRICS` exists, but the control plane re-checks raw
  `"manhattan"`/`"euclidean"` (`operator_station.py` ~24, `turn_orchestrator.py`, `intent_cache.py`,
  `semantic_normalizer.py`).
- Fix: a small resolver/validator sourced from `OPERATOR_DISTANCE_METRICS`; control-plane checks
  call it. **Out of scope:** `llm_compiler.py`'s metric mentions are prompt text, which the
  blueprint explicitly permits — left alone.

##### 13A.2.4 - Typed turn state (the structural slice; doubles as the de-bloat gate artifact) — ✅ done

Delivered `jeenom/turn_state.py` (`TurnState` dataclass + `TURN_STATE_FIELDS`). The 19 bare
per-turn attributes (`last_*` / `current_environment_identity` / `active_steering_directive`)
moved onto one typed object, initialized once in `__init__`; the session surfaces each field
as a delegating property (`_turn_field`, attached in a loop over `TURN_STATE_FIELDS`), so every
internal write and the ~40 evals reading `session.last_*` keep working unchanged. Initialization
is now guaranteed — the `active_steering_directive` `AttributeError` bug class is structurally
gone. Scope notes from the code: `active_claims` and the `pending_*` continuation state were
already property-backed (own homes, left alone); `LabelledEpisode` is already projected from
`command_result`, not the loose attrs, so no projection rewrite was needed. Probe:
`tests/test_turn_state_consolidation.py` (fields complete; each surfaced as a property;
delegation both ways; AST guard that `__init__` no longer bare-assigns the 19).

- Problem: 25 loose per-turn session attributes (`last_*` / `active_*` / `current_*` / `pending_*`)
  are **one concept** — the turn in flight plus the pending-continuation machine — scattered as
  bare fields. `TurnOrchestrator` reaches into 58 session members, ~40 evals assert on them,
  un-initialized ones bug out, and the same trace is **re-assembled** as `LabelledEpisode`
  (duplication: live attributes during the turn, `LabelledEpisode` at the end).
- Deliverable A (the spike's review artifact): a **shared-state map** — every reader/writer of
  each of the 25 attributes across `operator_station.py`, `turn_orchestrator.py`,
  `mission_cortex.py`, and evals/tests.
- Deliverable B: a typed `TurnState`
  (`envelope → intent → plan → graph → command → tickets → result → repair/mismatch`) owned by the
  session, plus a `PendingState` for the continuation machine; `LabelledEpisode` is **projected
  from** `TurnState` (single source of the turn trace).
- Migration: preserve the public `session.last_*`/`pending_*` reads via thin properties delegating
  to `TurnState`, so the ~40 dependent evals stay green while internal writers move onto the object.
- Probe: fails if `TurnOrchestrator`/`MissionCortex` read or write raw `last_*`/`pending_*` instead
  of going through `TurnState`; fails if any per-turn field is read before initialization.
- Payoff: kills the un-initialized-attribute bug *class*, turns the station↔orchestrator
  god-object coupling into a typed hand-off, and **this map + object is exactly the "shared-state
  map" the Phase 16 decomposition-design gate requires** — so it advances the de-bloat without
  front-running it.

Explicitly OUT of 13A.2 (deferred with cause):

- **Dispatch-pipeline unification** — the ~15 parallel `command_from_*` / `_run_*` methods are
  genuine logic repetition, but collapsing them *is* most of the station decomposition. It rides
  with the gated Phase 16 refactor; 13A.2.4 produces the prerequisite state map, it does not
  collapse the pipeline.
- **Claim-type unification** — the 6 claim-ish types (`ClaimRecord`, `ObservationClaim`,
  `ExecutionClaim`, `StationActiveClaims`, `GroundedObjectEntry`, `NamedConcept`) are tempting to
  merge, but blueprint rule 16 forbids a new schema family until an eval proves the current
  representation is lossy. Deferred.
- **`door`/color substrate vocabulary** — already catalogued and sequenced (Phase 14 cheap,
  Phase 15 structural). 13A.2.2 reduces its surface but must not front-run it.

Acceptance criteria (whole spike):

- Each slice: red-bar probe lands first, then goes green; `eval_master` + `pytest -q tests` green
  between every slice.
- No concept among { coordinate metric (13A.1 ✅), stable fingerprint, capability handle,
  distance-metric name } is hand-rolled outside its single home — a probe enforces each.
- Per-turn state flows through one typed `TurnState`; `LabelledEpisode` is projected from it; no
  raw `last_*`/`pending_*` poking across the station↔orchestrator boundary.
- Public `session` attribute reads preserved via compat properties; MiniGrid golden path intact.
- The shared-state map (13A.2.4 Deliverable A) is committed as a doc, reusable verbatim as the
  Phase 16 de-bloat gate artifact.

#### 13A.3 - Substrate-handle routing (routable-now) — investigated; folded into Phase 14

Investigated per the "routable-now only" scope. Result: **effectively empty**. The cheap
handle leftovers were already done (12D, 13A.2.2). The one remaining candidate —
`mission_cortex`'s `task.go_to_object.door` — is built inside the module-level
`parse_inline_metric_request` → `_continuation_intent` chain, which receives `registry` but
not `PlanningSemantics`; that chain is reached from the module-level `classify_utterance`
fast-path (also registry-only). Routing it would mean threading `PlanningSemantics` through
the classify/inline-parse path for **grammar-only** value (the `"door"` vocabulary stays an
explicit arg until Phase 14 derives `object_type` from context). That is the exact
`classify_utterance`/compiler substrate-boundary Phase 14 owns and AI2-THOR validates. Forcing
it now is churn Phase 14 would partly redo. **Decision: fold this single literal into the
Phase 14 pass.** Useful confirmation: the remaining handle/vocabulary coupling genuinely
clusters in the Phase 14 substrate boundary, not in cleanly-separable spots — the plan's
sequencing holds.

#### 13A.4 - Operator-station decomposition design (the Phase 16 gate artifact)

Status: **accepted and banked** (Phase 16 gate artifact). The decomposition design lives in
`PlanOfAction/operator_station_decomposition.md`: target modules, the shared-state map
(extending 13A.2.4's `TurnState`), the `TurnOrchestrator` reach-in problem, and an ordered
green-able extraction sequence. The decision is **state-first, not method-first**:
`StationRuntime` is the seed of the eventual substrate-independent kernel, carrying
construction-time collaborators plus typed `TurnState` and future `PendingState` boundaries.

Important sequencing decision: **do not start operator-station de-bloat now**, including safe
leaf extractions (`EnvironmentTracker`, `ConceptService`, `PendingFlowController`, service
migration, or `TurnOrchestrator` rewiring), unless explicitly instructed later. 13B will
redefine the partial-observability, evidence, ask-for-help, and claim-freshness structures that
the later station boundary carving must respect. Claim-type unification also stays **after 13B**
(13B's per-claim-kind freshness needs the distinctions).

Next implementation focus: **13B**.

### 13B - Partial observability + ask-for-help + meta primitives

Status: in progress. Makes steering *necessary* (omniscient answers become impossible, so
JEENOM must search/ask). Gives `scope` steering its teeth (`visible_only` vs
`search_allowed`). Workstreams:

- MiniGrid FOV: stop parsing the whole grid in `sense.py`; introduce a
  visible-vs-unseen distinction.
- `needs_evidence` readiness status + typed ask-for-help (a `ClarificationRequest`
  schema parallel to `PrimitiveDefinitionRequest`); operator reply recorded in
  `LabelledEpisode.steering`.
- Minimal deterministic `meta` primitives (`mode="deterministic"` per ORPI v0.1):
  `searched_region`, `reachability`, `behind` — infer claims from observed claims
  (produces the `inferred` **status**).
- Claim freshness under partial observability — the reviewed spike below.

#### 13B.1 — claim freshness kernel under partial observability

Status: complete. The synthetic red-bar probe graduated into the main architecture suite.
Delivered the first 13B freshness kernel without changing MiniGrid FOV or `FullyObsWrapper`:

- `CLAIM_FRESHNESS`: `current | unverifiable | stale | unknown`; `unverifiable` remains
  freshness-only and is not a claim status.
- New `jeenom/claim_freshness.py`: `UNVERIFIABLE_DECAY_STEPS`, `ClaimTTL`,
  `framing_satisfiable`, and `next_freshness`.
- The state machine now distinguishes look-away (`current -> unverifiable`) from
  env/world change (`current -> stale`), supports free snap-back while provenance is intact,
  requires fresh observation to recover from `unknown`, and keeps non-line-of-sight claim kinds
  from becoming `unverifiable`.
- `claim_custody_unverifiable_freshness_probe.py` moved from `expected_fail` to the main
  `architecture` suite.

#### 13B.2 — MiniGrid FOV boundary under partial observability

Status: complete. The red-bar probe graduated into the main architecture suite. MiniGrid now
supports explicit full-observation vs partial-observation eval lanes. Production runtime and
13B probes use native egocentric observations; legacy regression probes can still opt into
full observation without weakening the partial-observability invariants.

- `run_demo.build_env` constructs unwrapped MiniGrid envs.
- `evals.harness.build_env` is the legacy full-observation helper, while
  `evals.harness.build_partial_env` is the partial-observation helper.
- `build_minigrid_runtime_package(..., observability="full" | "partial")` makes the same
  split available to station/session evals; eval harness sessions default to `full`, and
  13B probes explicitly request `partial`.
- `MiniGridAdapter` annotates each observation with substrate-owned pose, grid-size,
  observation-model, and visible-cell metadata, so Sense does not reverse-engineer the
  MiniGrid FOV transform.
- `MiniGridSense` projects visible local-view cells into global coordinates, records
  `visible_cells`/`unseen_cells`, preserves global agent pose, and builds occupancy from
  observed cells only.
- `SceneModel` and `WorldModelSample` now carry the observation model and visibility boundary.
- `substrate_minigrid_fov_probe.py` asserts runtime and harness are partial-observation based,
  scene dimensions remain global, scene objects are limited to visible global cells, and unseen
  cells are explicit.

#### 13B.3 — `needs_evidence` readiness + typed ask-for-help

Status: complete. The red-bar probe graduated into the main architecture suite.

- `ReadinessGraph` now accepts `needs_evidence`; `_next_action` maps it to
  `ask_clarification`.
- `ClarificationRequest` is a typed schema parallel to `PrimitiveDefinitionRequest`, with
  `request_type=needs_evidence`, `evidence_scope`, target, options, and provenance.
- `RequestPlanStep.constraints` can now declare visible-only evidence requirements for
  grounding/ranking and selector steps.
- `CortexSession`/`evaluate_request_plan` accept a compact `evidence_state` so readiness can
  block supported-but-unanswerable visible-only steps before execution.
- `OperatorStationSession` records a typed `ClarificationRequest`, preserves it on pending
  clarification state, and exposes it in `LabelledEpisode.steering.pending_context`.
- `substrate_partial_observability_needs_evidence_probe.py` verifies the schema, readiness
  transition, station response, pending state, and ORPI steering context.
- Follow-up guard: a pending `needs_evidence` clarification no longer traps unrelated new
  operator intents; explicit new commands cancel the pending clarification and route normally.
- Post-motion sensing: raw motor commands advance a persistent adapter and immediately refresh
  the partial `SceneModel`, so visible evidence follows the agent's current POV instead of the
  stale pre-motion idle scene.
- LLM-route guard: motor-only sequences are advertised to the live compiler as
  `motor_sequence`, and the new `llm_path` eval suite verifies route provenance
  (`LLMCompiler` transport called), prompt contract, post-LLM semantic normalization, and
  deterministic parity for covered utterances before declaring the behavior green.
- LLM operator-matrix spike: `intent_fidelity_llm_operator_matrix_probe.py` now forces
  representative operator-intent families through `LLMCompiler` with fake transport and
  compares the normalized station outcome against the smoke path. Covered families include
  motor commands, motor sequences, LLM-misclassified all-motor `sequence_instruction`,
  task navigation, partial-observability `needs_evidence`, ambiguous navigation,
  unsupported pickup, conditional sense-before-motor, and concept teaching. The probe also
  asserts that the LLM prompt's advertised `supported.intent_types` matches the
  `OperatorIntent` schema enum, so future runtime intent types cannot quietly become
  regex-only.

Known follow-on: partial-observation task/ranking paths now correctly surface `NEEDS EVIDENCE`
from empty views. The next 13B slices should decide which partial paths move to
`search_allowed`/deterministic meta-primitives.

#### 13B.4 — Eval pipeline + tool-call discipline (deterministic gate + live-LLM suite)

Status: complete. A spike to make the test pipeline exercise BOTH the deterministic/regex path
and the genuine LLM path, on the principle that **the LLM emits typed tool-call decisions and
deterministic code owns execution + operator-facing statements** (the LLM is plumbing for
decisions, not a prose generator).

Headline root cause found + fixed: the LLM path was **silently falling back to regex on every
task compile**. `compile_operator_intent` was capped at `max_tokens=256`; the strict
`json_schema` `OperatorIntent` (26 fields) overflowed it → JSON truncated mid-response → parse
error → smoke/regex fallback. `DEFAULT_METHOD_MAX_TOKENS` is now sized to fit each schema
(`compile_operator_intent`/`compile_procedure` 1024; `compile_task`/`compile_sense_plan`/
`compile_skill_plan`/`compile_memory_updates` 768). `max_tokens` is an upper bound, so this is
cost-neutral for well-formed output — it only stops the truncation. The pinned `compile_task`
budget test was updated (256 → 768).

Tool-call discipline:
- **Refusal text is deterministic.** `_arbitrate_gap`'s refuse branch emits
  `cap_match.operator_message()` — the canonical `MISSING SKILLS: <handles>` — not the
  arbitrator's free text. The LLM/smoke arbitrator only *decides* to refuse; its reasoning
  stays in `last_arbitration_trace`. `cap_match.operator_message()` was consolidated to that one
  wording (matches the `missing_skills` verdict).
- **Unsupported routing keys off structured fields, not prose.** A bare unsupported/ambiguous
  intent used to route to `kind="unsupported"` vs `clarification` based on whether the LLM's
  *reason text contained the substring "unsupported"*. It now routes on the typed
  `capability_status` (`unsupported` → `kind=unsupported`; otherwise → "I didn't understand"
  clarification, the parse-failure case). Decision deterministic; reason text may stay as helper
  text. (A test fixture that set `intent_type=unsupported` with `capability_status="executable"`
  was made consistent.)

Eval pipeline (two layers):
- **Deterministic gate** — `eval_master` strips `OPENROUTER_API_KEY` for every suite except
  `live_llm`, so no gate probe can flake on (or pay for) a network call. Verified: all 78 gate
  probes pass with the key unset; one enforcement point replaces fragile per-probe key-pops.
- **`live_llm` opt-in suite** (`intent_fidelity_live_llm_probe.py`, in a separate
  `LIVE_LLM_SPECS` so it never runs under `all`): genuine model calls, **skip-if-no-key**
  (keyless CI stays green), asserts only STABLE tool-call facts — the LLM was actually used (no
  fallback) and produced the right `intent_type` decision — never free-text or scene-dependent
  `command_kind`. Backend-swappable: pointing `build_compiler` at a local model (e.g. LLAMA) is a
  backend change, not a probe change (the LLAMA swap itself is out of scope for now).
- The deterministic `intent_fidelity_llm_operator_matrix_probe.py` (fake transport) is the parity
  counterpart inside the gate; it forces the arbitrator offline so the gate stays reproducible.

Follow-on fix (granular semantic normalization of the decision vocabulary): the live LLM emitted
`grounding_query_plan.answer_fields=["distances"]` — the one dispatch field that was a free string
list, not an enum — so a "distance to the doors" query parsed clean but dead-ended in compose
("could not compose"). Fixed two ways: (1) `schemas._ensure_canonical_answer_fields` normalizes at
the single grounding-plan parse chokepoint — conservative aliases repair near-misses
(`"distances"→"distance"`, `"nearest"→"closest"`), ordinal forms pass through, unknown values fail
**closed** (→ regex fallback, else honest clarify); `GROUNDING_QUERY_ANSWER_FIELDS` is the
canonical set; (2) `_compose_grounding_query_plan` gained a branch for the `answer`+`distance`/
`ranked` shape so it produces a ranked-distance answer instead of dead-ending. This is
substrate-independent (shared vocabulary), the counterpart to the per-substrate domain helper.
`tests/test_answer_field_normalization.py` covers the canonicalizer (repair, casing, ordinal
pass-through, fail-closed) and the compose integration. The narrow "easy win"; the *full*
audit/enum-closure of every dispatch field + the hostile-input suite remain Phase 17.

Second follow-on (single-step motor sequence): live testing of a compound partial-observability
flow ("turn right; go forward twice; closest door; go to it") surfaced that
`go forward twice` → motor_sequence with one canonical step `["move_forward:2"]` dead-ended on a
`len(sequence) < 2` guard in `turn_orchestrator` ("Could not parse motor sequence steps."). A
single repeated action is valid; the guard is now `if not sequence` (execute any non-empty
sequence; only zero parseable steps is unparseable — matching the `sequence_instruction` path).
`tests/test_motor_sequence_single_step.py` covers single-step, multi-step, and empty. The full
compound flow now works end-to-end under partial observability.

Verification: `pytest -q tests` 308 passed; `eval_master` 78/78; `--suite live_llm` 1/1 (real
calls, stable across runs).

#### 13B spike — claim freshness under partial observability

Spike-first: design reviewed and red-barred before any code. **No `FullyObsWrapper`
or enum/predicate/helper/tick change until the red-bar probe is green-as-failing.**

The problem: today claim validity is a single equality `StationActiveClaims.is_valid_for`
over `(agent_x, agent_y, step_count)`, so **any agent move blanket-invalidates every
grounding claim as `stale`** — conflating *the world changing* with *the agent merely
looking away*.

**Axes (locked):** keep the two relevant axes (`freshness`, `status`). `status` untouched
(`inferred`/`hypothesis` already belong there). The only change is **one new freshness
value**: `current | unverifiable | stale | unknown`. Rationale: *verifiable* =
`freshness==current` (not stored); *not-knowable* = a property of the claim **kind** (not
stored); only **unverifiable** — "fresh as far as I know, but I can't currently re-confirm
it" — is unexpressible today. Do **not** add visible/verifiable/not-knowable as states.

**1. Triggers — two causes, never conflated:**

| Transition | Driver | Mechanism |
|---|---|---|
| `current → stale` | time/world-driven | world changed underneath the claim: env-identity fingerprint change, or a real scene mutation. **Not** mere agent movement. |
| `current → unverifiable` | grounding-driven, action-triggered, instantaneous | agent pose change removes the claim's supporting cell from view. **Not a timer** — a TTL can't detect loss of line-of-sight. |
| `current → current` | agent moved, cell still in view | **new**: today any move blanket-stales; the spike stops that. |
| `unverifiable → unknown` | time/step-driven decay | the *only* timer (§3). Out-of-view claim degrades the longer it's unseen. |
| `unverifiable → current` | snap-back, **free** | cell re-enters view **and** world fingerprint matches → restored from intact provenance, no re-grounding. Vouching for your own recent, unbroken observation. |
| `unknown → current` | **fresh observation only** (NOT snap-back) | claim is dead. Past decay the world could have changed during the unseen window and you hold **no fingerprint proving it didn't** — re-sighting is a new grounding that **overwrites** the dead claim. Old provenance is lineage/audit only, never a restoration source. |

**2. Mechanism — decompose the existing hook (no new subsystem).** Hook today:
`_claims_valid_for_current_environment` → `is_valid_for` → one equality → blanket
invalidation. Split the verdict into **three separate checks, never merged**:
(a) env-identity fingerprint mismatch → `stale`; (b) world-mutation check → `stale` —
*kept a distinct predicate even though static GoToDoor collapses it into (a) today, so 13C
retrofits nothing when doors open*; (c) agent pose/own-step change, same world → **per
grounding claim**, evaluate `framing_satisfiable(claim, pose) -> bool`: satisfiable ⇒
stays `current`, else ⇒ `unverifiable`. Per-kind (§5), never a blanket "looked away →
unverifiable" rule. Provenance retained so re-sighting is snap-back, not re-derivation.

**3. TTL — decay-edge only.** The eternal/timed/conditional TTL helper **does not exist
yet** (only inert `ClaimRecord.valid_until`); the spike specifies it as a small value type
`{ eternal | timed(n_steps) | conditional(predicate) }` built on `valid_until`, read by
**exactly one** call site: the `unverifiable → unknown` decay tick. Observation-gone-
unverifiable ⇒ `timed(UNVERIFIABLE_DECAY_STEPS)`; `operator_assertion`/`fact`/`procedure`
⇒ `eternal`. Transitions into/out of `unverifiable` are action-triggered; only the
terminal decay is time-driven.

**4. Do-not-collapse: `unverifiable` ≠ `hypothesis`.** `unverifiable` was observed →
snaps back free, confidence decays toward `unknown`; `hypothesis` was never grounded,
confidence climbs toward `confirmed`. Terminal of too-long-unseen is `unknown` (no false
current value), not `hypothesis`. Provenance kept intact through `unverifiable` (free
snap-back); at `unknown` the claim is dead (re-grounding overwrites, no restoration).
`unverifiable` is **freshness**; `hypothesis`/`inferred` are **status** — orthogonal.

**5. Per-claim-kind framing table** (framing = finite checklist of assumptions + checker):

| kind | scope | framing assumptions | checked by | `unverifiable`? |
|---|---|---|---|---|
| `observation` | grounding | (a) supporting cell in view from pose; (b) world unchanged; (c) env identity unchanged | (a) **in-view predicate on pose hook → `unverifiable`**; (b)/(c) world & env fingerprint → `stale` | **YES (only kind)** |
| `operator_assertion` | operator | (a) not retracted | retraction event only | **NO** — no line-of-sight; never unverifiable on look-away |
| `fact` | episodic | (a) same env identity | env-identity fingerprint | **NO** |
| `procedure` | procedure | (a) KB still defines the concept; (b) composed handles still registered | KB membership + capability registry | **NO** |

Critical guard: only `observation` has an in-view predicate — the table prevents the
look-away bug for the other three. Carry: `procedure`'s registered-handles framing is
**embodiment-scope** — cross-reference `NamedConcept.scope` (site/embodiment/universal,
`claim_custody_knowledge_scope_probe.py`) so the two "valid in this embodiment" notions
don't drift.

**6. Decay parameter.** Unit = **agent action steps** (discrete, deterministic, replayable
— not wall-clock). Per `unverifiable` claim: a `steps_unseen` counter, ++ each agent step
while the cell stays out of view, reset on snap-back. Threshold
`UNVERIFIABLE_DECAY_STEPS`, default `16` — an underived tuning target, possibly
manifest-driven later; the number is **not** a contract (probes assert behavior
parameterized on the imported constant, never the literal). At threshold:
`freshness → unknown`.

**Scope fences (locked):** only `inferred` (status; meta-primitive workstream) and
`unverifiable` (freshness; this spike) are forced by partial observability.
`belief`/`hypothesis`/self-generated `unknown` stay enum-ready-but-inert. No `status`-axis
changes here.

**Probe triage — by reading, not by ripping out `FullyObsWrapper`.** 0 probes assume an
omniscient grid (verified across all `claim_custody_*`). Re-read these 3 when `unverifiable`
lands, to confirm none asserts `stale` on *mere agent-pose* change (each currently asserts
stale on env/seed change, which correctly stays `stale`):
`claim_custody_env_assumption_probe.py`, `claim_custody_plan_reuse_probe.py`,
`claim_custody_stale_claim_probe.py`.

**Red-bar:** `claim_custody_unverifiable_freshness_probe.py` (synthetic; started in the
expected-fail suite and graduated into the main `architecture` suite in 13B.1).

### 13C - Curriculum + KB reuse + MTBCI

Status: later. The longitudinal learning proof.

- Curriculum runner over the env-size ladder (`minigrid_envs.py`: 8/10/12/16/32) with
  progressively harder partial-observability rungs.
- KB reuse across rungs via `PlanReuseCache` + scoped `KnowledgeBase` (already mature).
- MTBCI metric (mean turns between clarification/intervention), accumulated per episode,
  with a `regression_` probe showing MTBCI improves as reuse kicks in.

Acceptance criteria (whole phase):

- Operator steering changes typed plan constraints. (13A ✅)
- A grounding claim distinguishes `current` / `unverifiable` / `stale` / `unknown`, with
  look-away → `unverifiable` (not `stale`) and out-of-view decay → `unknown`; non-spatial
  kinds never go `unverifiable`. (13B) Note: the epistemic surface is the freshness axis
  plus the `inferred` status — *not* a new visible/searchable/not-knowable state set.
- Evidence gathering is represented as a plan, not a phrase branch. (13B)
- MTBCI improves across repeated runs of the same ladder rung as KB reuse kicks in. (13C)
- Steering is proven on MiniGrid. Cross-substrate sameness is explicitly out of scope
  here - it is the AI2-THOR spike's job (post-Phase-14), so a steering failure can never
  be confused with a substrate leak.

## Phase 14 - Cheap Leak Removal

Status: later (after Phase 13). Removes leaks tagged `cheap` and `not
curriculum-touching`; curriculum-touching leaks were removed before Phase 13.
Any `structural` leak is handled per its go/no-go flag, not silently absorbed here.

Goal: make the substrate boundary clean *before* the AI2-THOR spike, so a spike
failure means architecture leaked — not that we walked in with known dirt.

### Coupling audit table

Per site: `{cheap | structural}` × `{curriculum-touching | not}`.

| Site | ~Lines | Classification | Status |
|---|---|---|---|
| `llm_compiler.py` — fast-path grammar | 13 | structural × curriculum-touching | ✅ 12D.3 |
| `operator_station.py` — `_startup_warmup_instruction` | 22 | cheap × curriculum-touching | ✅ 12D.3 |
| `primitive_library.py` — door grounding primitives | 60 | cheap × not curriculum-touching | ✅ 12D.4 |
| `request_planner.py` — `rank_scene_doors` step_id | 8 | cheap × not curriculum-touching | ✅ 12D.4 |
| `llm_compiler.py` — LLM prompt strings | ~100 | cheap × not curriculum-touching | remaining |
| `operator_station.py` — MiniGrid imports + default env | ~8 | cheap × not curriculum-touching | remaining |
| `operator_station.py` — request parsing (color/door) | ~30 | cheap × not curriculum-touching | remaining |
| `sense.py` — `MiniGridSense` | ~100 | structural × not curriculum-touching | Phase 15 |
| `spine.py` — `MiniGridSpine` | ~80 | structural × not curriculum-touching | Phase 15 |

Planned work:

- `llm_compiler.py` prompt strings: swap door/color vocabulary examples for
  manifest-derived examples at session init.
- `operator_station.py` MiniGrid imports + default env: inject runtime_package;
  remove `minigrid_runtime_package` import from generic station module.
- `operator_station.py` request parsing with color/door: derive from manifest
  `symbol_mappings.object_index` / `color_index`.
- Widen `substrate_static_architecture_probe.py` to guard the now-cleaner core.

Acceptance criteria:

- Every `cheap`/`not curriculum-touching` leak from the audit is removed.
- No MiniGrid vocabulary remains in the files the audit scoped as the generic
  boundary; the static-architecture probe enforces it.
- Golden path preserved; full suite green.
- The `operator_station.py` bloat is untouched beyond the surgical leak scalpel -
  no early de-bloat.

### AI2-THOR substrate-independence spike (parallel branch)

A `ai2thor` branch is cut after Phase 14 lands the clean boundary. It is an
**exploratory requirements-discovery spike, not a committed port** - so it never
blocks `master`.

- It tests **only substrate-independence**: can the same kernel, ORPI
  contract/manifest/trace, and orchestration flow drive AI2-THOR? Steering is
  already proven (Phase 13), so any failure here isolates to the substrate
  boundary.
- Every place ORPI v0.1 bends or breaks is filed as a spec issue against
  `orpi_spec.md` and as a concrete kernel requirement.
- Output feeds Phase 15: the spike's findings drive a **second, targeted leak
  removal pass** before hardening - spike-discovered blockers, not just
  pre-audited ones.
- Prerequisite to note now: AI2-THOR is a heavyweight Unity-backed simulator, not
  a pip-light gym env. An environment/install check is a gate before the branch is
  viable.

## Phase 15 - Cross-Substrate Demonstration & ORPI v1 Freeze

Status: later. The AI2-THOR spike is the exploratory precursor; Phase 15 is the
committed port that merges the spike's learnings and freezes the interface.

Goal: prove the same architecture and the same ORPI contract/manifest/trace work
on MiniGrid and AI2-THOR - and use that port to freeze ORPI v0.1 to v1.

Pointer requirements:

- Same `OperatorIntent`, `RequestPlan`, `ReadinessGraph`, claims, tickets, and
  orchestration kernel; only the substrate adapter HOW differs.
- Both substrates register an ORPI manifest and pass the ORPI conformance items.
- The spike's filed spec issues are resolved or consciously deferred; a
  spike-driven second targeted leak-removal pass lands before this phase closes.

Acceptance criteria (also the v1 freeze gate):

- Second substrate is ORPI conformant; one MiniGrid task and one AI2-THOR task
  follow the same cognitive flow.
- Differences are confined to the substrate adapter and domain helpers; no
  MiniGrid vocabulary leaks into the generic kernel.
- ORPI freezes to v1: from v1, additive changes only; breaking changes require a
  major version.

## Phase 16 - Operational Hardening

Status: later.

Goal: make the architecture reliable after the steering and cross-substrate proofs
exist. **This phase absorbs the deferred `operator_station.py` de-bloat in full**
(per the 12D bloat worklist), gated by the decomposition-design prerequisite.

Planned work:

- `operator_station.py` extraction into a substrate-independent orchestration
  kernel plus adapters — decomposition design written and reviewed first.
- repair metrics, synthesis provenance, intervention counts, transfer evals
- missing primitive / ambiguity / no-path handling
- render-time guarantees preserved across substrates

### Bloat worklist (station extraction sketch — for design phase only)

`operator_station.py` is ~5,613 lines / 178 methods; only ~67 are substrate-coupled
(handled in Phase 14). The rest is substrate-independent orchestration.

| Candidate module | ~Methods | Notes |
|---|---|---|
| Intent dispatch + knowledge routing | ~20 | `TurnOrchestrator` already exists; station could delegate fully |
| Plan cache + reuse | ~15 | `PlanReuseCache` already separate; station holds glue only |
| Clarification / synthesis / definition state | ~30 | Pending-state machine; self-contained |
| Mission flow + continuation | ~25 | `MissionCortex` already separate |
| Auth + ticket management | ~12 | `CommandAuthority` already separate |
| Core `handle_utterance` orchestration | ~15 | Thin coordinator after extraction |

**Hard prerequisite:** any extraction — in Phase 16 or pulled forward — requires a
decomposition design (target modules, shared-state map, ordered green-able extraction
sequence) written and reviewed first. No station code moves before that design exists.

## Phase 17 - Capability Stress Tests

Status: later.

Use harder MiniGrid, real robotics, or ARC-like tasks only as architecture
pressure tests. The goal is not benchmark chasing; the goal is to expose missing
primitive contracts, missing claims, missing evidence, bad decomposition, and
bad steering.

### Phase 17 - Adversarial robustness & hostile-input hardening

Status: later (deferred by explicit decision). **Until this phase lands, JEENOM assumes a
good-faith operator** (see the "Threat model" note in Current Known State and the README). The
current architecture *contains* a misbehaving/jailbroken LLM for the dangerous cases — typed
tool-call outputs only, enum-validated decision fields, unknown primitives rejected, side effects
gated by station-minted tickets + `IntentVerifier` semantic preservation + `ReadinessGraph`, no
LLM in the render loop — but this containment is **not yet proven** against adversarial inputs,
and one dispatch field (`grounding_query_plan.answer_fields`) is still an open string list.

Scope when it lands:

- **Close the decision vocabulary fully + fail-closed.** Audit every LLM-controlled field that
  influences dispatch; enum-constrain or normalize it; an unknown value must produce an explicit
  reject/clarify, never a silent dead-end or a silent reroute. (`answer_fields` is the known open
  field — its narrow correctness/robustness fix is an *easy win* that can land before Phase 17;
  the full audit is Phase-17 scope.)
- **Side-effect authority proof.** Demonstrate that no LLM-controlled field *value* alone can mint
  an `ExecutionTicket`/`RawMotorTicket`/`MemoryWriteTicket` without the utterance's semantics and
  `ReadinessGraph` concurring — i.e. a crafted-but-schema-valid tool-call cannot route to an
  unintended side-effectful path.
- **Hostile tool-call / prompt-injection eval suite.** Adversarial probes (deterministic
  fake-transport with crafted payloads + a live lane): out-of-vocabulary field values, an
  actuation handle smuggled into a query intent, `operation`/`primitive_handle` mismatches, and
  injection text in the operator utterance. Assert the system fails **closed** (reject/clarify,
  zero unintended side effect) — turning "no leakage" into a continuously-verified invariant.
- **Untrusted-input channels.** If/when JEENOM ingests text it did not originate (documents, tool
  outputs, other agents), that becomes a distinct injection surface needing its own trust-boundary
  handling — out of scope until such a channel exists.


