## JEENOM Implementation Plan — Capability Ladder

### Phase 0 — MiniGrid smoke test
Status: done.
Prove MiniGrid wrapper, observe/act/render, and simple task execution.

Technical debt:
- Rendering and episode lifecycle are still MiniGrid-specific.
- GUI responsiveness is fragile during compile/prewarm unless the station keeps the preview alive.
- No abstraction yet for transferring observe/act/render semantics to a robot or simulator backend.

### Phase 1 — Minimal JEENOM vertical slice
Status: done.
Implement CorticalSchema contracts, Cortex, Sense, Spine, WorldModelSample, OperationalEvidence, Percepts, ExecutionContract, ExecutionReport.

Technical debt:
- Contracts are still shaped around the current GoToDoor loop and need pressure from more task families.
- Error and readiness reporting are usable but not yet rich enough for complex failure recovery.
- Cortex/Sense/Spine boundaries work, but primitive registration is still not fully dynamic.

### Phase 2 — LLM compiler boundary
Status: done.
LLM compiles typed schema objects. Runtime validates and executes deterministic primitives.

Technical debt:
- Some LLM structured-output schemas still need provider-hardening; compile_memory_updates has shown OpenRouter/Azure schema rejection.
- Fallback behavior is useful for tests but can hide provider/schema issues unless evals explicitly require live LLM use.
- Prompt/schema drift remains a risk until each compiler method has live-provider probes.

### Phase 3 — JIT template caching + prewarm
Status: done.
Cache ProcedureRecipe, SensePlanTemplate, SkillPlanTemplate. Prewarm before render. No LLM calls or cache misses during rendered control loop.

Technical debt:
- Cache keys and invalidation are still simple and may not survive richer primitive/version changes.
- Prewarm can add visible startup or handoff latency.
- Cache introspection is adequate for debugging but not yet a full operator-facing readiness report.

### Phase 3.5 — Regression/golden test
Status: done.
Freeze the current success case:
- instruction: "go to the red door"
- compiler: llm
- render/prewarm enabled
- task_complete=True
- runtime_llm_calls_during_render=0
- cache_miss_during_render=0
- final action=done

Technical debt:
- Golden path covers one instruction, one object family, and one known-good seed.
- It proves the cached runtime invariant, not broad task competence.
- Human render is patched/headless in test, so visual behavior still needs manual smoke checks.

### Phase 4 — Bigger same-task stress test
Status: done.
Run larger GoToDoor environment. Same task, same recipe, longer path. No Q/A yet.

Goal:
Run the same "go to the red door" task in a larger GoToDoor environment with the same LLM compiler + prewarm + cached runtime pattern.

Success criteria:
- same instruction: "go to the red door"
- bigger GoToDoor env
- compiler=llm
- prewarm enabled
- task_complete=True
- runtime_llm_calls_during_render=0
- cache_miss_during_render=0
- final action/skill is done
- Phase 3.5 golden eval still passes

Implemented:
- env: MiniGrid-GoToDoor-16x16-v0
- seed: 42
- loop_count=8
- final skill_plan=['done']
- runtime_llm_calls_during_render=0
- cache_miss_during_render=0

Technical debt:
- Still same task family and no blocked-path/replan pressure.
- Larger grids expose startup/prewarm/render latency but do not solve it.
- The test starts a fresh episode per run; it does not prove continuous-world task chaining.


### Phase 5 — CLI operator interface
Status: done.
Support interactive natural-language operator loop. CLI only.

Implemented:
- OperatorStationSession owns env_id, seed, compiler, OperationalMemory, PlanCache, render_mode, and last_result.
- run_operator_station.py starts an interactive READY prompt.
- Operator utterances are classified without command prefixes:
  - task_instruction
  - knowledge_update
  - status_query
  - cache_query
  - reset
  - quit
- Supported knowledge updates:
  - delivery_target is the canonical operator target fact
  - target_color and target_type are kept synchronized internally for runtime compatibility
  - accepted variants include "the red door is the delivery target", "your delivery target is the red door", "target is the red door", "remember the red door", and "set delivery target to the red door"
- Supported task phrasing includes "go to the red door", "go to red door", "reach the red door", "find the red door", "head to the red door", and "navigate to the red door".
- Task instructions use the existing JEENOM pipeline with task-specific prewarm before render.
- Target-absent failures report available targets and return to READY without follow-up Q/A.
- Reset clears episodic state and last_result while keeping durable knowledge by default.
- Phase 3.5 and Phase 4 regression guarantees still pass.

Technical debt:
- The interface is operational rather than conversational; responses are readable but not naturally dialogic.
- Startup/prewarm transparency is still log-style, not a polished operator experience.
- Interactive render windows still restart between task episodes by design.
- Natural-language coverage depends on the Phase 7 LLM path and remains constrained by typed schemas.

### Phase 6 — Memory-Grounded Reference Resolution

Status: done.

Goal:
Let the operator refer to remembered task targets without restating the full instruction.

Examples:
- the green door is the delivery target
- go to the delivery target
- go there again
- go to the same door
- repeat the last task
- what was the last target?

Scope:
- still doors only
- still GoToDoor / go_to_object-shaped task
- no continuous-world semantics
- each task can still start from a fresh MiniGrid episode

Success criteria:
- After a successful "go to the red door", "go there again" resolves to "go to the red door".
- "repeat the last task" resolves to the last canonical task instruction.
- "what was the last target?" prints the last target color/type.
- If no previous target exists, reference commands fail safely without asking follow-up questions.
- "go to the delivery target" continues to use durable delivery_target.
- Phase 3.5, Phase 4, and Phase 5 tests still pass.

Implemented:
- Added episodic reference fields:
  - last_target
  - last_task
  - last_successful_instruction
- "go there again" and "go to the same door" resolve from episodic last_target.
- "repeat the last task" resolves from episodic last_successful_instruction / last_task.
- "what was the last target?" reports color, object_type, and canonical instruction.
- Missing reference memory fails safely and returns to READY without follow-up Q/A.
- Successful task completion updates reference memory.
- Failed runs do not overwrite reference memory.
- Explicit reset clears episodic reference context and last_result.
- Durable delivery_target still works after reset.
- clear memory / forget everything clears durable target knowledge.

Technical debt:
- Episodic memory is shallow: last target/task only, not a structured event timeline.
- References are resolved against fresh-episode semantics, not current physical position.
- Failed or partial tasks do not yet produce useful recoverable memory beyond preserving the last successful target.

### Phase 6.5 — Operator Session Semantics
Status: done.

Goal:
Make the station behave coherently across multiple operator turns.

Decision:
- Each task starts a fresh MiniGrid episode for now.
- reset clears episodic state and last_result.
- reset keeps durable delivery_target.
- clear memory / forget everything clears durable target knowledge.
- last_result persists until reset, clear memory, or the next task overwrites it.
- last_target persists until reset, clear memory, or the next successful task overwrites it.
- failed runs do not overwrite last_target.
- durable delivery_target persists across station restarts.

Implemented:
- reset clears last_result and episodic reference context.
- reset keeps durable delivery_target available for "go to the delivery target".
- clear memory / forget everything clears delivery_target, target_color, and target_type.
- delivery_target persists through OperationalMemory disk reload.
- task onboarding clears per-episode runtime state while preserving reference context between turns.

Technical debt:
- Fresh-episode semantics remain surprising in an operator station because "again" does not mean continue from the current rendered location.
- Durable and episodic stores are still minimal and need clearer inspection/export tools.
- There is no multi-session audit trail for what was remembered, forgotten, or inferred.

### Phase 7 — LLM-Grounded Operator Understanding
Status: done.

Goal:
Replace brittle operator string matching with a typed, validated LLM intent compiler, while preserving deterministic fast paths and no runtime LLM calls during render.

Scope:
- classify natural operator utterances
- extract structured task, knowledge, status, control, and memory-reference intent
- emit a typed OperatorIntent schema
- support the current door/go_to_object task family
- keep regex/string parser as fast path and fallback
- allow LLM calls only before task execution/render
- validate all LLM outputs before acting
- invalid or unsupported LLM outputs must fail safely or fall back to deterministic parser
- the station, not the LLM, performs memory updates, readiness, prewarm, and runtime execution

Success criteria:
- "can you please head over to the blue door" resolves to go_to_object blue door
- "that red door is our delivery target" updates durable delivery_target
- "go back to the same one" resolves through episodic last_target
- "what did I ask you to do last time?" resolves to last task/status query
- invalid/ambiguous LLM parse does not execute unsafe behavior
- runtime_llm_calls_during_render remains 0
- cache_miss_during_render remains 0
- Phase 3.5, 4, 5, 6, and 6.5 guardrails still pass

Out of scope:
- continuous-world execution
- clarification Q/A
- mid-run operator correction
- open/unlock/pickup
- exploration/replan
- non-door object families

Implemented:
- Added typed OperatorIntent schema with:
  - intent_type
  - canonical_instruction
  - task_type
  - target
  - narrowly typed knowledge_update
  - reference
  - status_query
  - control
  - clear_memory
  - confidence
  - reason
- Added LLMCompiler.compile_operator_intent with typed schema validation and smoke-test fallback.
- Deterministic parser is now the exact-match fast path.
- Fuzzy/operator-natural utterances can route through the LLM intent compiler before execution.
- LLM output must validate as OperatorIntent before conversion to OperatorCommand.
- Unsupported and ambiguous LLM intents fail safely and do not execute.
- Station still owns memory updates, readiness, prewarm, and runtime execution.
- Supported fuzzy examples now covered:
  - "can you please head over to the blue door"
  - "that red door is our delivery target"
  - "go back to the same one"
  - "what did I ask you to do last time?"
- Fuzzy scene/status queries are covered:
  - "what do you see around you" maps to status_query=scene.
- Delivery-target questions are covered:
  - "and the delivery target?" maps to status_query=delivery_target.
- Question-shaped utterances cannot mutate memory even if the LLM incorrectly returns a knowledge_update intent.
- Verbose mode logs validated OperatorIntent type, confidence, and reason.
- Unsupported capability examples such as pickup/key requests do not execute.

Technical debt:
- The LLM can classify into typed intents, but it only understands what the schema and capability manifest let it express.
- Unsupported or underspecified requests can still feel blunt because open-ended dialogue is out of scope.
- Provider behavior must be continuously checked with live evals, not only fake transport tests.
- The memory-update compiler schema needs cleanup because OpenRouter/Azure rejected one structured output shape.


### Phase 7.5 — Typed Target Selector Grounding
Status: done.

Goal:
Ground fully specified relational target requests against the current MiniGrid scene using typed selectors and deterministic scene primitives. No clarification loop yet.

Scope:
- The LLM emits a TargetSelector.
- The station validates the selector before use.
- Scene grounding deterministically resolves the selector against current scene data.
- Only a grounded target is converted into an existing go_to_object task.
- The LLM must not choose the closest object itself.
- Door-only support:
  - closest door by Manhattan distance from agent
  - unique door matching include/exclude color constraints
  - visible door list/status queries

TargetSelector:
- object_type: door
- color: supported color | None
- exclude_color: supported color | None
- relation: closest | unique | None
- distance_metric: manhattan | euclidean | None
- distance_reference: agent | None

Validation:
- unsupported object type rejected
- unsupported color rejected
- unsupported relation rejected
- closest execution requires a registry-supported distance_metric and distance_reference
- ambiguous selector does not execute
- no-match selector does not execute

Examples:
- which doors are visible?
- which door is closest by Manhattan distance?
- go to the closest door using Manhattan distance
- go to the door that is not yellow
- make the closest door by Manhattan distance the delivery target

Implemented:
- Added TargetSelector schema and OperatorIntent.target_selector.
- Added station-side selector validation before grounding.
- Added deterministic MiniGrid scene grounding for door selectors.
- Added grounded target status output with target color/type/location and distance.
- Added closest-door grounding by Manhattan distance from the agent.
- Added selector-based go_to_object execution after grounding.
- Added selector-based delivery_target knowledge updates.
- Added exact visible-door query support.
- Added safety checks so implicit closest without a metric, ambiguous selectors, unsupported selectors, no-match selectors, and LLM-chosen closest targets do not execute.
- Phase 3.5, Phase 4, and operator-station guardrails still pass.

Out of scope:
- implicit closest without metric
- non-door object families
- left/right/front/behind
- continuous-world execution
- exploration/replan
- pickup/open/unlock
- clarification Q/A

Technical debt:
- Selector grounding is deterministic and narrow: doors, color filters, and closest-by-supported-metric only.
- The LLM proposes selectors, but the station still needs explicit selector schema fields for every relation we want to ground.
- Unsupported relations like facing-direction, left/right, occlusion, or semantic categories will not work until represented in the selector/capability layer.

### Phase 7.55 — Capability Registry and Primitive Manifest
Status: done.

Goal:
Stop patching operator language with regex. Give the LLM and station a typed capability substrate.

The registry should describe available task, grounding, sensing, and action primitives, including their input/output schemas, side effects, implementation status, and whether they are safe to synthesize.

Do not add primitive synthesis yet.
Do not add Euclidean distance implementation yet.
Do not add new object families.
Do not change runtime execution.

Implementation:
- Add PrimitiveManifest / PrimitiveSpec schema.
- Add CapabilityRegistry.
- Load MiniGrid primitive manifest from a YAML/JSON/Python dict.
- Expose compact manifest summary to compile_operator_intent.
- Make OperatorIntent validation check requested task/selectors against the registry.
- Make readiness report:
  executable
  missing_primitive
  unsupported
  synthesizable_missing_primitive
- Keep exact deterministic fast path only for trivial commands.
- Do not patch fuzzy phrases with regex.

Success criteria:
- manifest lists current door/go_to_object, scene, sensing, and action primitives
- "closest door by Manhattan distance" maps to implemented primitive
- "closest door by Euclidean distance" maps to missing but synthesizable grounding primitive
- "pick up the key" maps to unsupported/missing action primitive and does not execute
- LLM receives manifest summary before compiling OperatorIntent
- invalid requests fail safely
- Phase 3.5 through Phase 7 guardrails still pass
- runtime_llm_calls_during_render remains 0
- cache_miss_during_render remains 0

Implemented:
- Added PrimitiveSpec and PrimitiveManifest schemas.
- Added CapabilityRegistry with MiniGrid primitive manifest.
- Manifest lists task, grounding, sensing, and action primitives with inputs, outputs, side effects, implementation status, and safe_to_synthesize.
- compile_operator_intent receives compact capability_manifest summary.
- Station checks selector readiness against CapabilityRegistry before grounding/execution.
- Readiness statuses include:
  - executable
  - missing_primitive
  - unsupported
  - synthesizable_missing_primitive
- Manhattan closest-door grounding maps to implemented primitive grounding.closest_door.manhattan.agent.
- Euclidean closest-door grounding maps to missing but synthesizable primitive grounding.closest_door.euclidean.agent and does not execute.
- Pickup/key action maps to unsupported action primitive and does not execute.
- Fuzzy closest/not-yellow operator language now routes through LLM intent compilation with the manifest instead of station regex patches.
- Exact deterministic fast path is kept for trivial commands only.
- Added evals/operator_intent_probe.py to verify live LLM clarification arbitration.
- Full JEENOM suite passes.

Live LLM clarification eval:
- Command:
  python evals/operator_intent_probe.py --utterance "I see. What is the closest door to you"
- Passing result requires:
  - used_live_llm
  - intent_type=status_query
  - capability_status=needs_clarification
  - status_query=ground_target
  - target_selector.object_type=door
  - target_selector.relation=closest
  - target_selector.distance_metric=None
  - target_selector.distance_reference=None
- Do not use --allow-fallback as proof; that only sanity-checks the script without OpenRouter.

Station-level clarification eval:
- Command:
  python evals/operator_clarification_probe.py
- This is the real clarification-loop proof, not just intent inspection.
- Passing result requires:
  - used_live_llm
  - station_returned_clarify
  - clarify_mentions_closest
  - clarify_mentions_distance_metric
  - clarify_lists_manhattan
  - pending_clarification_created
  - pending_type_missing_field
  - pending_missing_field_distance_metric
  - pending_supports_manhattan
  - no_result_before_answer
  - "manhattan" answer clears the pending clarification
  - closest target query returns GROUNDED TARGET with target and distance
  - closest task request returns RUN COMPLETE
  - task_complete=True
  - runtime_llm_calls_during_render=0
  - cache_miss_during_render=0
  - final_skill_plan=["done"]
- You may run with --allow-fallback only to sanity-check eval wiring. It must show FAIL used_live_llm and does not prove live LLM clarification.

Technical debt:
- The manifest is currently a Python-defined MiniGrid manifest, not a discovered or generated registry.
- Capability statuses are useful, but readiness language still needs to be unified across compiler, station, and primitive registry.
- Missing-but-synthesizable primitives are detected but not synthesized until Phase 7.8.
- Manifest summaries must stay compact enough for LLM use while retaining enough detail for good arbitration.

### Phase 7.56 — Registry Unification
Status: done.

Goal:
Unify CapabilityRegistry with the actual primitive substrate. The registry must derive from or faithfully mirror primitive_library.py so the LLM sees the same task/sensing/action capabilities that the runtime can actually validate and execute.

This is not a new architecture. It is the implementation of the existing Readiness + Sensing Primitive Library + Action Primitive Library architecture.

Source of truth:
- primitive_library.py remains the source of truth for implemented runtime primitives.
- CapabilityRegistry is a typed view/index over that source.
- Do not create a second independent manifest that can drift from runtime.

Registry must expose:
- all TASK_PRIMITIVES
- all SENSING_PRIMITIVES
- all ACTION_PRIMITIVES
- grounding/selector primitives
- implementation_status: implemented | unsupported | synthesizable | planned
- consumes
- produces
- side_effects
- runtime_binding
- layer: task | grounding | sensing | action

Implemented:
- Extended primitive_library.PrimitiveSpec with implementation_status, safe_to_synthesize, side_effects, and runtime binding metadata.
- Added GROUNDING_PRIMITIVES to primitive_library.py for selector/query grounding primitives:
  - visible_doors
  - closest_door.manhattan.agent
  - closest_door.euclidean.agent
  - unique_door.color_filter
- CapabilityRegistry now builds its MiniGrid registry from:
  - TASK_PRIMITIVES
  - GROUNDING_PRIMITIVES
  - SENSING_PRIMITIVES
  - ACTION_PRIMITIVES
  - top-level task capability declarations such as task.go_to_object.door
- Compact registry summaries include layer, status, consumes/inputs, produces/outputs, side effects, runtime binding, and synthesis safety.
- Operator intent compilation continues to receive the compact registry summary.
- Help/capability queries now answer from CapabilityRegistry instead of a hardcoded "Try: ..." string.
- Readiness distinguishes task, grounding, sensing, and action layers in its report.

Success criteria:
- registry lists all TASK_PRIMITIVES, SENSING_PRIMITIVES, ACTION_PRIMITIVES from primitive_library.py
- registry marks each as implemented / unsupported / synthesizable / planned
- registry exposes consumes / produces / side effects / runtime binding
- LLM operator-intent prompt uses compact registry summary
- capability/help queries answer from registry, not hardcoded strings
- readiness can explain missing task vs sensing vs grounding vs action primitive
- no change to runtime execution behavior
- golden path still passes
- Phase 3.5 through Phase 7.5 guardrails still pass
- runtime_llm_calls_during_render remains 0
- cache_miss_during_render remains 0

Eval:
- Command:
  python evals/capability_registry_probe.py
- Passing result requires:
  - all task/sensing/action primitives from primitive_library.py appear in the registry
  - grounding primitives appear in the registry
  - action.plan_grid_path exposes consumes, produces, and runtime_binding
  - Euclidean closest-door grounding is reported as synthesizable missing primitive
  - pickup key is reported as an unsupported task capability
  - "what can you do" returns a CAPABILITIES report from the registry

Out of scope:
- primitive synthesis
- Euclidean implementation
- new object families
- new robot actions
- changing Sense/Spine execution semantics
- replacing PlanCache

Technical debt:
- Top-level task capabilities such as task.go_to_object.door are still declared beside the primitive dictionaries because Understanding recipes are not yet stored as first-class registry primitives.
- Capability help is registry-grounded but still formatted as a compact text report, not a natural conversational explanation.
- Sensing/action runtime bindings are indexed, but dynamic binding/discovery is still future work.


### Phase 7.57 — Persistent Scene Model
Status: done.

Goal:
Project WorldModelSample into a typed, active SceneModel after every sense tick. Grounding and scene queries use SceneModel directly — no live env resets to answer current-scene questions.

Core rule:
SceneModel is projected from WorldModelSample. It is not a second sensing path.

Source of truth:
- MiniGridSense.execute_plan() produces WorldModelSample (unchanged).
- After each tick, WorldModelSample is distilled into SceneModel and stored in OperationalMemory.
- Grounding queries read from SceneModel.
- If SceneModel is absent, one idle sense pass is performed against the existing adapter.

Implemented:
- Added SceneObject and SceneModel dataclasses to schemas.py.
  - SceneModel.from_world_model_sample() is the only construction path.
  - SceneModel.find() filters by object_type, color, exclude_color.
  - SceneModel.manhattan_distance_from_agent() computes distance from agent pose.
- Added OperationalMemory.scene_model slot and update_scene_model() method.
- SceneModel is cleared in reset_episode() (both operator reset and task onboard).
- Added MiniGridSense.sense_idle_scene(): builds SceneModel from a live observation
  using the existing parse_grid_objects + get_agent_pose primitives.
  source="idle_sense". Stored in memory.scene_model.
- MiniGridSense.execute_plan() projects to SceneModel after every tick.
  source="task_sense". Stored in memory.scene_model.
- Added OperatorStationSession._ensure_scene_model():
  - Returns memory.scene_model if already set.
  - Otherwise calls sense_idle_scene() against the live adapter.
  - Falls back to a temporary env only if no adapter exists (initial cold start).
  - Never resets an existing adapter.
- Replaced ground_target_selector() to use _ensure_scene_model() + SceneModel.find() + SceneModel.manhattan_distance_from_agent().
- Removed _scene_adapter() and _agent_pose() from grounding paths.
- Replaced scene_summary() to read from SceneModel. Now includes agent pose and source.
- unique_door.color_filter now produces distance in addition to grounded_target.
  Distance is computed as Manhattan distance from agent pose.
- Added _scene_object_to_dict() helper for grounding return values.
- Added evals/scene_model_probe.py.

Success criteria:
- SceneModel is populated after a task sense tick.
- SceneModel is populated by idle sense before any task if an operator asks a scene query.
- "which doors are visible?" answers from SceneModel.
- "which door is closest by Manhattan distance?" answers from SceneModel.
- After a task completes, scene queries use the final sensed agent pose, not spawn.
- No grounding query calls adapter.reset(seed=...) or equivalent.
- unique_door grounding returns distance.
- Phase 3.5 through 7.56 guardrails still pass.
- runtime_llm_calls_during_render remains 0.
- cache_miss_during_render remains 0.

Eval:
- Command:
  python evals/scene_model_probe.py
- Passing result requires:
  - scene_model_populated_after_task
  - scene_model_source_task_sense
  - scene_model_has_agent_pose
  - scene_model_has_door_objects
  - idle_sense_populates_scene_model
  - idle_sense_source_is_idle_sense
  - closest_grounding_ok
  - closest_grounding_returns_distance
  - grounding_did_not_call_adapter_reset
  - unique_grounding_ok
  - unique_grounding_returns_distance
  - scene_summary_has_agent
  - scene_summary_has_source
  - scene_summary_has_doors
  - scene_model_cleared_on_reset
  - scene_model_rebuilt_after_reset

Out of scope:
- continuous-world task execution
- replan
- clarification loop
- Euclidean primitive synthesis
- non-door object families
- pickup/open/unlock

Technical debt:
- SceneModel persists the last observed scene within an episode but does not accumulate across episodes.
- Idle sense uses a temporary env reset only on the first cold-start query; after any task or startup, the live adapter is always used.
- env_id and seed are not yet stored on SceneModel when built from the task sense path.

### Phase 7.58 — Active Situational Awareness (Inter-Turn Claims)
Status: done.

Goal:
Give the station a queryable, session-scoped Claims layer that persists grounding results
across READY turns. Grounding queries write typed Claims. Subsequent operator turns can
reference prior Claims in natural language without re-grounding from scratch.

Problem this solves:
Phase 7.57 gave the station a persistent SceneModel — it knows where things are. But
grounding results are returned as printed strings and immediately forgotten. The next
operator turn has no record of what was answered:

  operator: "which door is closest?"
  station:  GROUNDED TARGET — purple door@(4,0) distance=5
  operator: "and the next closest?"
  station:  [asks for distance metric again — has no memory of the prior answer]

The station answered correctly but cannot reason about its own prior output. This is not
a TargetSelector schema gap (adding rank=2 would be a spot fix). It is a missing
projection: grounding results must write into an active Claims layer, just as Sense
writes into Claims during task execution. Then "next closest" resolves against the
existing ranked_doors Claim, not a fresh grounding query.

Architecture mapping:
The workflow diagram already has this layer: Claims in the Active Operational Model.
During task execution, Cortex.claims is populated by Sense evidence. Between tasks,
Claims are empty. Phase 7.58 extends Claims to cover inter-turn scene awareness.

  grounding result → Claim (typed, named, session-scoped)
  operator reference ("the next one", "from there", "is it closer than the red one?")
    → resolved against active Claims, not re-grounded

This is the correct substrate for:
  - "and the next closest door?" → Claims["ranked_scene_doors"][1]
  - "go to the closer of the two" → execute against a ranked Claim
  - "how far is the red door from the purple one?" → inter-Claim distance query
  - "is the blue door further than the red one?" → compare two distance Claims

Scope:
- Session-scoped Claims store in OperatorStationSession (not OperationalMemory — not durable).
- Claims written by grounding queries:
  - last_grounded_target: the most recently grounded SceneObject + distance
  - ranked_scene_doors: full sorted list of doors with distances at time of grounding
  - last_grounding_query: the selector that produced the last result
- Claim references resolved by the LLM intent compiler before grounding:
  - "the next one" / "and the next closest" → ranked_scene_doors[last_rank + 1]
  - "the other door" / "the remaining one" → ranked_scene_doors excluding last_grounded_target
  - "from there" / "from where you are" → use agent pose from SceneModel (already have this)
- Claims survive across READY turns within a session.
- Claims are cleared on reset.
- Claims are cleared when a new task runs (task execution starts fresh).
- Claims do not write to durable OperationalMemory.
- No new LLM calls to resolve Claim references — resolution is deterministic against the
  Claims store; the LLM classifies the intent type, the station resolves the reference.

Implementation constraints:
- Do not add Claim resolution to the task execution render loop.
- Do not change Cortex.claims (task-execution Claims are separate from inter-turn Claims).
- Do not change SceneModel.
- Do not add a new sensing path.
- Claims are typed dataclasses, not a raw dict blob.
- LLM intent compiler receives a compact Claims summary (active_claims) so it can
  classify "the next one" as a Claim reference intent rather than a new grounding query.

Success criteria:
- After "which door is closest?", Claims["last_grounded_target"] = purple door, Claims["ranked_scene_doors"] = [purple@5, yellow@6, ...]
- "and the next closest door?" resolves to yellow door from ranked_scene_doors[1] without re-grounding or asking for distance metric.
- "the other door" resolves from ranked_scene_doors excluding last_grounded_target.
- Claims are cleared on reset.
- Claims are cleared when a new task instruction runs.
- No grounding query re-runs when a Claim reference can answer the question.
- Phase 3.5 through 7.57 guardrails still pass.
- runtime_llm_calls_during_render remains 0.
- cache_miss_during_render remains 0.

Out of scope:
- Claim persistence across sessions (durable memory).
- Inter-Claim distance queries ("how far is the red door from the purple one?") — future phase.
- Claim provenance tracking — future phase.
- Non-door object families.
- Continuous-world execution.
- Euclidean synthesis.

Implemented:
- schemas.py: GroundedDoorEntry dataclass (color, x, y, distance, as_dict()).
- schemas.py: StationActiveClaims dataclass (scene_fingerprint, ranked_scene_doors,
  last_grounded_target, last_grounded_rank, last_grounding_query).
  Methods: is_valid_for(scene), next_ranked(), other_doors(), compact_summary().
- schemas.py: OPERATOR_CLAIM_REFERENCES = ("next_closest", "other_door").
- schemas.py: OperatorIntent.claim_reference field + JSON schema + from_dict() parsing.
- schemas.py: "claim_reference" added to OPERATOR_INTENT_TYPES.
- operator_station.py: self.active_claims: StationActiveClaims | None added to session.
- operator_station.py: _write_ranked_claims() — builds and stores StationActiveClaims after
  closest-relation grounding; fingerprints against current SceneModel.
- operator_station.py: _resolve_claim_reference(ref_type) — checks fingerprint validity;
  returns next_closest (rank+1) or other_door (remaining doors) deterministically.
- operator_station.py: claim_reference_summary(ref_type) — formatted output for the operator.
- operator_station.py: handle_utterance() — claim_reference branch dispatches to
  claim_reference_summary before the task_instruction path.
- operator_station.py: active_claims cleared on reset() and at start of run_task().
- operator_station.py: command_from_llm_intent() passes active_claims_summary to compiler.
- llm_compiler.py: CompilerBackend ABC, SmokeTestCompiler, LLMCompiler all accept
  active_claims_summary: dict | None = None in compile_operator_intent().
- llm_compiler.py: SmokeTestCompiler detects "next closest" / "next one" → claim_reference=next_closest
  and "other door" / "remaining door" → claim_reference=other_door before the closest-door branch.
- llm_compiler.py: LLMCompiler includes active_claims_summary in payload; system prompt
  explains claim_reference intent type and OPERATOR_CLAIM_REFERENCES.
- llm_compiler.py: LLMCompiler fallback call passes active_claims_summary through.
- llm_compiler.py: "claim_reference" added to supported.intent_types payload.
- tests/test_jeenom_minigrid.py: TestStationActiveClaims — 10 tests covering claims written
  after closest grounding, rank-0 invariant, next_closest resolution, reset clearing,
  stale fingerprint failure, compact_summary shape, is_valid_for() behavior.
- evals/active_claims_probe.py: 17 checks — all PASS.
- evals/scene_model_probe.py: 16 checks still PASS (no regression).
- Full test suite: 101 tests PASS.

### Phase 7.59 — Intent Readiness Requirement Matching
Status: done.

Goal:
Introduce a deterministic CapabilityMatcher that independently verifies whether a compiled
OperatorIntent can be fulfilled against the current CapabilityRegistry — before any execution
is attempted. This replaces the LLM's self-assessed capability_status with an authoritative,
substrate-independent verdict.

Problem this solves:
The LLM currently does two jobs: parsing operator intent (what the operator wants) and
assessing whether the station can fulfill it (capability_status). These are different jobs and
the second one is not reliable. When the LLM finds a partial match — a capability that covers
most of the intent but not all of it — it silently degrades the query rather than emitting
missing_skills. The station has no independent check to catch this:

  operator: "tell me the doors closest to you in descending order using manhattan distance"
  LLM: finds grounding.closest_door.manhattan → capability_status=executable
  station: runs closest-door grounding, returns rank=0 only
  result: "in descending order" (a ranked-listing requirement) is silently dropped

capability_status never fired because the LLM thought it was doing something executable.
The station had no independent view of what the intent actually required. This is the same
class of failure that the task-execution readiness check was built to prevent — but the
readiness check only runs at procedure-execution time, not at intent time.

Architecture mapping:
Task-execution readiness (existing) answers: "do I have the actions and evidence to run
this procedure step?" Intent readiness (new) answers: "do I have the capabilities to
attempt what this intent requires at all?" Both are deterministic checks against the registry.
Neither is the LLM's opinion. Together they form a two-level readiness gate:

  Level 1 — Intent Readiness (Phase 7.59):
    OperatorIntent → extract required capability handles
    → CapabilityMatcher.check(intent, registry)
    → verdict: executable | needs_clarification | synthesizable | missing_skills
    → overrides LLM's capability_status

  Level 2 — Procedure Readiness (existing):
    ProcedureRecipe → check actions and evidence
    → ExecutionReadiness (already implemented)

Design principles:
- CapabilityMatcher is layer-independent. It operates on structured capability handles
  (e.g., "grounding.ranked_doors", "sensing.get_agent_pose", "actuation.move_base") and
  CapabilityRegistry entries. It has no knowledge of MiniGrid, ROS, or any substrate.
- Intent requirements are declared by the LLM compiler as structured handles in
  `required_capabilities`, not inferred post-hoc. The LLM declares what it requires; the
  matcher checks whether it exists.
- The LLM capability_status field is retained but demoted to a hint. If the LLM says
  capability_status=executable but the matcher returns missing_skills or synthesizable,
  the matcher wins. Always.
- CapabilityMatcher runs after LLM compilation, before command dispatch, every turn.
- When a required handle is missing from the registry, the matcher emits missing_skills
  with the specific missing handle — not a generic error. This enables the station to give
  the operator a precise response: "I can find the closest door but do not have a ranked
  listing primitive."
- synthesizable is not executable. A synthesizable result means the capability is absent
  but marked safe to generate. The station reports this; actual synthesis is Phase 7.8.
  synthesizable does not unblock execution in this phase.
- When multiple handles are required and some are missing, the matcher identifies all gaps
  at once, not just the first.
- No weakening. The matcher enforces exact handle matching:
  - grounding.closest_door does NOT satisfy grounding.ranked_doors
  - grounding.closest_door does NOT satisfy grounding.nth_closest_door
  - task.go_to_object does NOT satisfy task.pickup
  - grounding.visible_doors does NOT satisfy grounding.ranked_doors
  A broader capability never silently covers a more specific requirement.

Scope:
- Add `required_capabilities: list[str] | None` field to OperatorIntent — structured
  capability handles the intent needs to be fulfilled. Emitted by the LLM compiler. The
  LLM declares "I need these capabilities"; the matcher checks whether they exist.
- Add CapabilityMatcher class (new module: jeenom/capability_matcher.py):
  - match(intent: OperatorIntent, registry: CapabilityRegistry) -> CapabilityMatchResult
  - CapabilityMatchResult: verdict, matched, missing, synthesizable_handles fields
  - Pure function over typed inputs — no env, no sense, no memory, no substrate imports.
- Integrate into command_from_operator_intent — matcher runs before any execution branch.
- LLM system prompt updated to emit `required_capabilities` as a list of handles from the
  manifest. If the intent needs a handle not in the manifest, include it anyway — the
  matcher will classify it as missing_skills.
- SmokeTestCompiler updated to emit `required_capabilities` deterministically for each
  parsed intent type.
- CapabilityRegistry gains a `lookup(handle: str) -> CapabilityEntry | None` method.
- OperatorCommand gains a `capability_match: CapabilityMatchResult | None` field so the
  station can report precisely what was missing to the operator.

What this does NOT do:
- Does not add new grounding, actuation, or sensing primitives — the matcher surfaces gaps,
  it does not fill them. Adding new capabilities is Phase 7.8.
- Does not change the task-execution readiness check.
- Does not change SceneModel or StationActiveClaims.
- Does not add substrate-specific readiness contracts for actuation or sensing — those are
  a later phase. This phase covers the matching layer; the contracts come next.

Portability:
CapabilityMatcher is the same code whether the substrate is MiniGrid, a ROS robot arm, or
a simulated warehouse. The registry entries differ; the matcher logic does not. When porting
to robotics: register the robot's actual capability handles (e.g.,
"sensing.lidar_occupancy_grid", "actuation.move_base", "grounding.nearest_object.euclidean")
and the matcher immediately tells you what any operator intent requires vs. what the robot
can do. No environment-specific code in the matching layer.

Success criteria:
- "tell me the doors in descending order by manhattan distance" → required_capabilities
  includes grounding.ranked_doors.manhattan; not in registry → verdict=missing_skills,
  missing=["grounding.ranked_doors.manhattan"]. Station responds with the specific gap.
- "go to the closest door" → required_capabilities=["grounding.closest_door.manhattan",
  "task.go_to_object"]; both present → verdict=executable, proceeds as before.
- "go to the closest door using euclidean distance" → required_capabilities includes
  grounding.closest_door.euclidean; registry marks it synthesizable →
  verdict=synthesizable, station reports it but does NOT execute.
- LLM says capability_status=executable for ranked_doors; matcher returns missing_skills;
  matcher wins; station reports missing_skills.
- CapabilityMatcher imports nothing from minigrid, gymnasium, or any sense/execution module.
- Phase 3.5 through 7.58 guardrails still pass.
- runtime_llm_calls_during_render remains 0.
- cache_miss_during_render remains 0.

Out of scope:
- Generating or synthesizing missing primitives (Phase 7.8).
- Substrate-specific readiness contracts for actuation and sensing primitives.
- Multi-robot capability federation.
- Capability versioning or deprecation.

Implemented:
- jeenom/capability_matcher.py: new module — CapabilityMatcher, CapabilityMatchResult,
  verdict constants (executable, missing_skills, synthesizable, skipped, unsupported).
  Pure function over typed inputs. No minigrid, gymnasium, sense, spine, or env imports.
  Verified by AST import analysis in probe.
- CapabilityMatcher.match(intent, registry) → CapabilityMatchResult.
  Exact handle lookup via registry.lookup() — no prefix relaxation, no weakening.
  All missing handles reported at once. synthesizable ≠ executable.
- jeenom/capability_registry.py: added lookup(handle) → PrimitiveSpec | None.
  Exact dict lookup — no subsumption, no fuzzy matching.
- jeenom/schemas.py: added required_capabilities: list[str] field to OperatorIntent.
  Parsed in from_dict(); included in JSON schema with description enforcing no-weakening.
- jeenom/operator_station.py: CapabilityMatcher runs at the top of
  command_from_operator_intent every turn. matcher verdict gates all execution branches.
  If verdict=missing_skills or unsupported → kind=missing_skills command with specific handles.
  If verdict=synthesizable → kind=synthesizable command, does not unblock execution.
  LLM's capability_status field is now a hint only; matcher wins on conflict.
  Added missing_skills and synthesizable branches to handle_utterance dispatch.
- jeenom/operator_station.py: OperatorCommand gains capability_match field.
- jeenom/llm_compiler.py: SmokeTestCompiler emits required_capabilities for every parsed
  intent type. Ranked-listing queries detected before closest branch; emit
  grounding.ranked_doors.manhattan.agent as missing handle — not degraded to closest.
- jeenom/llm_compiler.py: LLMCompiler system prompt instructs LLM to emit
  required_capabilities with exact handles. No-weakening rule stated explicitly.
  grounding.ranked_doors distinguished from grounding.closest_door.
- evals/capability_matcher_probe.py: 27 checks — all PASS.
- evals/scene_model_probe.py, active_claims_probe.py: all prior checks still PASS.
- Full test suite: 101 tests PASS.

### Phase 7.595 — Proactive Intent Signal Verification
Status: done.

Goal:
Add an IntentVerifier that deterministically extracts semantic signals from the operator's
utterance and injects the correct required_capabilities into the compiled OperatorIntent
before the CapabilityMatcher fires — regardless of what the LLM declared. Stops intent
inversion and silent degradation before they reach execution.

Problem this solves:
The CapabilityMatcher (Phase 7.59) only fires when the LLM populates required_capabilities.
When the LLM silently degrades or inverts intent — finding a nearby executable capability
and substituting it — required_capabilities stays empty and the matcher defers:

  operator: "go to the farthest door"
  LLM: finds closest_door (executable) → substitutes, emits task_instruction
  CapabilityMatcher: required_capabilities=[] → verdict=skipped
  station: executes closest-door task — agent moves to wrong location

  operator: "what is the distance of all the doors from you"
  LLM: emits unique selector → clarification for single door
  CapabilityMatcher: skipped
  station: asks which single door — completely wrong response

These are not edge cases. Any time the LLM finds a partial match it may degrade.
For a physical robot this means actuation in the wrong direction with no warning.
Blueprint Rule 9 prohibits this. Intent inversion must be a hard stop.

Architecture:
IntentVerifier sits between the LLM compiler output and the CapabilityMatcher.
It is proactive — it acts on the utterance text directly, not on what the LLM said.

  utterance + compiled OperatorIntent
    → IntentVerifier.verify(utterance, intent) → list[IntentSignal]
    → IntentVerifier.inject_required_capabilities(intent, signals) → enriched intent
    → CapabilityMatcher.match(enriched_intent, registry) → verdict
    → station acts on verdict

IntentSignal carries: signal_type, detected_term, required_handle.
The required_handle is injected into required_capabilities if not already present.
The CapabilityMatcher then fires on the enriched intent — the mechanism is unchanged.

Signal classes (substrate-independent):
- SUPERLATIVE: farthest, furthest, most distant, longest way
  → requires grounding.farthest_door.{metric}.agent (missing → missing_skills)
- CARDINALITY: all/every/each + object, distance of all, sort/rank/list + objects
  → requires grounding.ranked_doors.{metric}.agent (missing → missing_skills)
- ORDINAL: second/third/2nd/3rd + closest/nearest
  → requires grounding.nth_closest_door.{metric}.agent (missing → missing_skills)

Metric is inferred from utterance when specified; defaults to manhattan otherwise.
Signal detection is case-insensitive, whitespace-normalised, order-independent.

Design:
- IntentVerifier is a pure class: no env, no sense, no memory, no LLM, no substrate imports.
- Verified by AST analysis in probe (same check as CapabilityMatcher).
- Injected handles are logged so the operator can see what signal was detected.
- If the LLM already correctly declared required_capabilities, no injection occurs
  (no double-injection — set union).
- IntentVerifier does not change intent_type or any other field — only enriches
  required_capabilities. The CapabilityMatcher decides the verdict.

Scope:
- New module jeenom/intent_verifier.py: IntentSignal, IntentVerifier.
- Integration in operator_station.command_from_operator_intent: runs before CapabilityMatcher.
- SmokeTestCompiler updated to detect superlative and cardinality patterns natively
  (belt-and-suspenders — IntentVerifier is the safety net; SmokeTestCompiler catches early).
- CapabilityMatcher unchanged — IntentVerifier enriches the input, not the matcher.
- Tests and probe.

Success criteria:
- "go to the farthest door" → IntentVerifier detects SUPERLATIVE signal "farthest",
  injects grounding.farthest_door.manhattan.agent → missing_skills → no task executed.
- "what is the distance of all the doors" → CARDINALITY signal "all the doors",
  injects grounding.ranked_doors.manhattan.agent → missing_skills.
- "sort the doors by distance" → CARDINALITY signal, missing_skills.
- "go to the second closest door" → ORDINAL signal, missing_skills.
- "go to the closest door" → no signals → CapabilityMatcher proceeds as before → executable.
- "go to the red door" → no signals → task executes as before → golden path preserved.
- IntentVerifier has no substrate imports (verified by AST).
- Phase 3.5 through 7.59 guardrails still pass.
- runtime_llm_calls_during_render remains 0.

Out of scope:
- Implementing farthest/ranked/ordinal grounding — that is Phase 7.7.
- Semantic similarity matching beyond keyword signals.
- Multi-language signal detection.

Implemented:
- blueprint.md: Rule 9 added — intent semantic preservation is mandatory. Intent
  inversion and silent degradation are hard stops. IntentVerifier is the enforcement.
- workflow_diagram.mmd: IV (Intent Verifier) node added between L (LLM Interface) and
  CS (CorticalSchema). Utterance signal extraction sits in the architecture diagram.
- jeenom/intent_verifier.py: new module. IntentSignal, IntentVerificationResult,
  IntentVerifier. Pure class, no substrate imports (verified by AST analysis).
  Three signal classes: SUPERLATIVE (farthest/furthest/most distant),
  CARDINALITY (all doors/sort/rank/list by distance), ORDINAL (second/third closest).
  enrich(utterance, intent) → (enriched_intent, result): primary API.
  Metric inferred from utterance (euclidean if specified, manhattan otherwise).
  No double-injection: set union with existing required_capabilities.
- jeenom/operator_station.py: default_verifier.enrich() runs before CapabilityMatcher
  in command_from_operator_intent every turn. Injected handles logged.
- jeenom/llm_compiler.py: SmokeTestCompiler detects superlative terms (farthest/furthest
  etc.) before closest/ranked branches and emits required_capabilities=[farthest handle].
- evals/intent_verifier_probe.py: 29 checks — all PASS.
  Verified: farthest→missing_skills+no task, all_doors→missing_skills, sort→missing_skills,
  second_closest→missing_skills, golden path unaffected.
- All prior probes still PASS. Full test suite: 101 tests PASS.

### Phase 7.596 — Capability Arbitrator
Status: done.

Goal:
Replace the hard-coded "MISSING CAPABILITIES" dead end with a typed, LLM-capable
arbitration layer that reasons about what to do when a capability gap is detected.

Problem this solves:
When CapabilityMatcher or IntentVerifier detects a gap, the station currently returns
a blunt "MISSING CAPABILITIES" message with no reasoning. The operator cannot tell
whether the station could substitute a different capability, ask for clarification,
synthesize the missing primitive, or genuinely has no path forward. The LLM should
reason about this decision — not regex rules. This is the architectural premise:
"use LLMs to arbitrate about these decisions about whether we need a new type of
primitive or not, whether to render new code, whether the capability registry is
adequate or not."

Architecture:
  CapabilityMatcher detects gap (verdict=missing_skills/synthesizable)
    → CapabilityArbitrator.arbitrate(utterance, intent, cap_match)
    → ArbitrationDecision (typed, validated)
    → ArbitrationTrace (provenance)
    → Station acts on decision (refuse/clarify/substitute/synthesize)

Hard rules (Blueprint Rule 9):
  - refuse/synthesize decisions are never safe_to_execute.
  - substitute requires safe_to_execute=True and a concrete suggested_handle.
  - Intent inversion (closest for farthest) is never a valid substitute.
  - Silent degradation (closest for ranked_doors) is never a valid substitute.

Scope:
  - ArbitrationDecision schema with typed decision_types and validation.
  - ArbitrationTrace for provenance recording.
  - ArbitratorBackend ABC — no substrate imports.
  - SmokeTestArbitrator — deterministic rule-based arbitration, no LLM.
  - LLMArbitrator — makes a compile-time reasoning call, falls back to smoke.
  - build_arbitrator(compiler_name) factory.
  - OperatorStationSession.arbitrator and last_arbitration_trace.
  - TargetSelector.exclude_color (str) → exclude_colors (list[str]).
  - Migration helper _migrate_exclude_color for legacy LLM responses.
  - SceneModel.find() updated to exclude_colors list.
  - SmokeTestCompiler updated to emit exclude_colors lists.
  - Multi-exclusion parsing: "not purple or yellow" → exclude_colors=["purple", "yellow"].
  - JSON schema updated: exclude_colors array type instead of exclude_color string.

Success criteria:
  - CapabilityArbitrator has no substrate imports (verified by AST).
  - SmokeTestArbitrator refuses missing handles with MISSING SKILLS message.
  - ArbitrationDecision refuses/synthesize are safe_to_execute=False (validated).
  - ArbitrationTrace records utterance, intent_type, handles, decision.
  - LLMArbitrator falls back to smoke when no API key.
  - Farthest-door session sets last_arbitration_trace=refuse.
  - Golden-path "go to the red door" sets no arbitration trace.
  - exclude_colors multi-exclusion works: "not purple or yellow" → list of two.
  - Legacy exclude_color migrated to exclude_colors correctly.
  - SceneModel.find(exclude_colors=["purple","yellow"]) filters both.
  - Phase 3.5 through 7.595 guardrails still pass.
  - 101 tests pass.

Implemented:
  - schemas.py: ARBITRATION_DECISION_TYPES constant.
  - schemas.py: ArbitrationDecision dataclass with __post_init__ validation.
  - schemas.py: ArbitrationTrace dataclass with compact() method.
  - schemas.py: TargetSelector.exclude_color → exclude_colors: list[str].
  - schemas.py: _migrate_exclude_color helper — migrates legacy dict in-place.
  - schemas.py: _ensure_target_selector validates exclude_colors as list.
  - schemas.py: operator_intent_json_schema() uses exclude_colors array type.
  - schemas.py: SceneModel.find() updated to exclude_colors: list[str] | None.
  - jeenom/capability_arbitrator.py: ArbitratorBackend ABC, SmokeTestArbitrator,
    LLMArbitrator (with fallback), build_arbitrator factory, default_arbitrator.
  - jeenom/capability_arbitrator.py: arbitration_decision_json_schema() for LLM.
  - jeenom/capability_arbitrator.py: _parse_arbitration_decision() with hard-rule
    enforcement (refuse/synthesize cannot be safe_to_execute=True).
  - operator_station.py: imports ArbitrationTrace, build_arbitrator.
  - operator_station.py: self.arbitrator and self.last_arbitration_trace.
  - operator_station.py: _arbitrate_gap() — fires arbitrator on gap, records trace,
    acts on decision (refuse/clarify/substitute/synthesize).
  - operator_station.py: command_from_operator_intent routes missing_skills/
    synthesizable/unsupported through _arbitrate_gap instead of returning directly.
  - operator_station.py: exclude_color → exclude_colors in ground_target_selector.
  - llm_compiler.py: all target_selector dicts use exclude_colors: [] instead of
    exclude_color: None.
  - llm_compiler.py: not_color_match → not_color_matches (list); supports multi-exclusion
    via "not X or Y" pattern. Parses "or <color>" as additional exclusions.
  - evals/capability_arbitrator_probe.py: 32 checks — all PASS.
  - All prior probes still PASS. Full test suite: 101 tests PASS.

Out of scope:
  - LLM arbitration for substitute decision with re-routing.
  - Persistent arbitration trace across sessions.
  - Synthesis of missing primitives (Phase 7.8).

Technical debt:
  - substitute decision type is parsed and validated but does not yet re-route
    execution through the substitute handle — the station still returns missing_skills.
    Full substitute execution requires re-building the intent with the new handle.
  - LLMArbitrator replicates transport code from LLMCompiler. Extracting a shared
    transport module is a cleanup opportunity.
  - Arbitration traces are not yet surfaced to the operator or stored in episodic memory.


### Phase 7.6 — Operator Clarification Loop For Grounding
Status: done.

Goal:
When a grounding request is underspecified or ambiguous but potentially supported, the station should produce a structured clarification, remember the pending grounding intent, accept the answer, then continue safely.

Example:
- operator: "go to the closest door"
- station: asks which distance metric to use
- operator: "manhattan"
- station: completes the selector as distance_metric=manhattan and distance_reference=agent
- station: grounds the target using Phase 7.5 selector grounding
- station: runs the existing go_to_object pipeline
- operator: "go to the door that is not yellow"
- station: grounds multiple matching candidates and asks which one to use
- operator: "red"
- station: selects the red candidate and runs the existing go_to_object pipeline

Scope:
- session-local pending clarification state
- no durable memory writes for pending clarification
- one outstanding clarification at a time
- clarify missing selector fields and ambiguous candidate sets
- support missing distance_metric for closest-door selectors
- support candidate choice for ambiguous door selectors
- accept answers like "manhattan", "use manhattan", "manhattan distance", and "by manhattan distance"
- accept candidate answers like "red", "the red one", and "red door"
- unsupported answers such as "euclidean" must not execute yet
- status/cache queries can still be answered while a clarification is pending
- reset/cancel clears pending clarification
- a new task instruction while pending cancels the pending clarification and runs the new instruction

Success criteria:
- "go to the closest door" does not execute immediately.
- station prints a CLARIFY prompt asking for the distance metric.
- pending clarification is stored in session state.
- "manhattan" completes the selector and executes the grounded task.
- "go to the door that is not yellow" asks which matched door to use when multiple candidates exist.
- "red" selects the red candidate and executes the grounded task.
- runtime_llm_calls_during_render remains 0.
- cache_miss_during_render remains 0.
- final_skill_plan == ["done"].
- "euclidean" does not execute and reports unsupported metric until primitive synthesis.
- "cancel" clears pending clarification without running a task.
- reset clears pending clarification.
- status/cache queries work while keeping pending clarification.
- new task while pending cancels pending clarification and runs.
- Phase 3.5, 4, 5, 6, 7, and 7.5 guardrails still pass.

Implemented:
- Added PendingClarification session state to OperatorStationSession.
- Added an exact deterministic selector fast path for "go to the closest door".
- Added an exact deterministic selector fast path for "go to the door that is not yellow".
- Added missing_required_clarifiable / invalid_unsupported / ambiguous / no_match selector grounding statuses.
- Ambiguous grounding results now carry candidate doors into the clarification contract.
- "go to the closest door" now asks for a distance metric instead of executing or hard-failing.
- "manhattan" completes the pending selector with distance_metric=manhattan and distance_reference=agent.
- "go to the door that is not yellow" now asks which candidate to use when multiple non-yellow doors match.
- Candidate answers such as "red" resume the pending grounding and execute the selected task.
- Completed selectors re-enter Phase 7.5 grounding and then run the existing go_to_object path.
- "euclidean" reports "I cannot use Euclidean distance yet. Supported: manhattan." and clears pending clarification.
- cancel and reset clear pending clarification.
- status/cache queries work while leaving pending clarification intact.
- new task instructions cancel pending clarification and run normally.
- Full JEENOM suite passes.

Out of scope:
- primitive synthesis
- Euclidean distance implementation
- multiple simultaneous clarifications
- open-ended Q/A
- ambiguous references like "the one on the left"
- non-door object families
- continuous-world execution

Technical debt:
- Clarification is solved for modeled grounding gaps only: missing distance_metric and ambiguous candidate choice.
- It is not a general Socratic/operator-dialogue engine yet.
- If the request needs an unmodeled relation, object type, metric, or missing primitive, the station should fail safely or mark it synthesizable; it will not invent a useful clarification unless that gap is represented.
- Broad clarification requires the capability registry plus future primitive synthesis/readiness machinery, not more phrase patches.


### Phase 7.7 — Grounding Result Composition
Status: done.

Goal:
Use outputs from existing registered grounding/query primitives to answer operator questions or
produce grounded targets for existing go_to_object tasks. Composition must use registered
primitive outputs, SceneModel, and ActiveClaims. It must not invent capabilities, bypass the
registry, or weaken operator intent.

Core rule:
- The station composes from the registered primitive handle:
  - grounding.all_doors.ranked.manhattan.agent
- If no ranked claim exists, the station refreshes it through the registered ranked primitive.
- If the ranked claim is stale, the station refreshes it before composing.
- Unsupported metrics still go through missing/synthesizable capability handling.

Supported composition:
- closest = ranked[0]
- farthest = ranked[-1], unless tied
- second closest = ranked[1]
- color reference = matching object in the active ranked claim
- distance reference = matching object(s) in the active ranked claim
- answer query = display composed result
- task query = compose grounded target, then run existing go_to_object

Tie handling:
- Answer queries display ties.
- Task execution with tied targets asks clarification and does not execute until resolved.
- Second/ordinal target execution asks clarification if the requested rank is tied.
- The station must never silently pick from ties.

Implemented:
- Added composition from StationActiveClaims ranked door outputs.
- "which door is closest and which is farthest?" returns both from the ranked claim.
- "go to the farthest door" clarifies on a tied farthest result.
- A color answer such as "red" resumes the tied-target clarification and runs go_to_object.
- "go to the second closest door" composes ranked[1] into a go_to_object task.
- "go to the door with a distance of 7" resolves from ranked claims when the match is unique.
- After a ranked display, "go to the red one" resolves using active claims.
- After a grounding answer such as "third farthest=purple...", "go to that" resolves to the
  last grounded target from ActiveClaims through the registered
  grounding.claims.last_grounded_target handle.
- Added tests/test_jeenom_minigrid.py::TestGroundingResultComposition.
- Added evals/grounding_composition_probe.py.

Proved:
- Composition uses grounding.all_doors.ranked.manhattan.agent as the primitive source.
- Answer queries do not execute tasks.
- Task queries still execute through the existing cached JEENOM go_to_object path.
- runtime_llm_calls_during_render remains 0.
- cache_miss_during_render remains 0.
- Ties are surfaced to the operator instead of silently selected.
- "second farthest" does not degrade into "farthest"; if the requested ordinal rank is tied,
  the station asks for a candidate instead of moving.

Test plan:
- python evals/grounding_composition_probe.py
- python -m pytest -q tests/test_jeenom_minigrid.py -k GroundingResultComposition
- python -m pytest -q tests/test_jeenom_schemas.py tests/test_jeenom_minigrid.py

Technical debt:
- Composition is currently over ranked visible doors only.
- Color-only follow-up references work when a current ranked claim exists; broader pronoun/reference
  resolution still needs a more general discourse model.
- Ties are handled at the station layer, but multi-turn candidate dialogue is still single-pending
  clarification only.
- Semantic signal extraction still relies on deterministic text patterns as a safety verifier.
  This is not the desired long-term understanding layer. Phase 7.75 replaces this with an
  LLM-emitted typed semantic query plan plus deterministic validation.
- This does not add new grounding primitives; Euclidean and other metrics remain future synthesis work.

Out of scope:
- Euclidean distance
- primitive synthesis
- non-door objects
- left/right/front/behind
- continuous-world execution
- pickup/open/unlock
- blocked-path replan

### Phase 7.75 — LLM Semantic Query Planner
Status: done.

Goal:
Make the LLM solve the language-understanding part by emitting a typed semantic query plan,
instead of relying on regex-like signal extraction for ranked/superlative/ordinal grounding.

Core rule:
- The LLM may interpret the operator's language.
- The LLM must not choose objects or answer from imagination.
- The LLM emits a typed GroundingQueryPlan.
- JEENOM validates that plan against the CapabilityRegistry, SceneModel, and ActiveClaims.
- Only validated plans are composed into answers or existing go_to_object tasks.

Why this phase exists:
- Phase 7.7 proved the composition substrate works once the intended ranked/ordinal operation
  is known.
- The remaining weakness is semantic parsing: phrases such as "second farthest" should be
  understood by the LLM as an ordinal over a ranked set, not recovered by deterministic patterns.
- Regex/verifier logic remains as a safety backstop during the transition, but it should no
  longer be the primary semantic parser after this phase.

New schema:
- GroundingQueryPlan:
  - object_type: door
  - operation: list | filter | rank | select | answer
  - primitive_handle: registered capability handle, if known
  - metric: manhattan | euclidean | None
  - reference: agent | None
  - order: ascending | descending | None
  - ordinal: integer | None
  - color: supported color | None
  - exclude_colors: list[supported color]
  - distance_value: integer | None
  - tie_policy: clarify | display
  - answer_fields: list[str]
  - required_capabilities: list[str]
  - preserved_constraints: list[str]

Implementation:
- Added GroundingQueryPlan schema and validation helpers.
- Extended OperatorIntent with optional grounding_query_plan.
- Updated operator-intent LLM prompt to use the compact CapabilityRegistry and emit
  grounding_query_plan for grounding/status/task requests.
- Updated SmokeTestCompiler only as a deterministic fixture for tests; do not grow it into the
  production parser.
- Added validation that the plan preserves important operator constraints:
  - ordinal words such as second/third must appear as ordinal values.
  - farthest/furthest/least close must produce descending ranking semantics.
  - closest/nearest must produce ascending ranking semantics.
  - explicit colors and exclusions must be preserved.
  - explicit metrics must be preserved.
- Converted valid GroundingQueryPlan into the existing Phase 7.7 composition path.
- If the plan needs a missing/synthesizable primitive, route through capability matching and
  arbitration; do not execute.
- Kept runtime execution unchanged.

Success criteria:
- "can you navigate to the second farthest door" produces a plan with:
  - operation=select
  - primitive_handle=grounding.all_doors.ranked.manhattan.agent
  - order=descending
  - ordinal=2
  - tie_policy=clarify
- "which door is closest and which is farthest?" produces an answer plan over the ranked handle.
- "which door is closest and second closest?" produces a multi-answer ranked plan with
  answer_fields=["closest", "second_closest"] rather than collapsing the second constraint.
- "how far is the red door?" produces a distance answer plan using the ranked/scene grounding
  substrate, not delivery_target status.
- "is there a green door?" produces a filter/answer plan and returns yes/no from SceneModel.
- "go to the door with distance 7" produces distance_value=7 and executes only if unique.
- "rank the doors from nearest to farthest" produces a ranked display plan.
- "go to the closest door using Euclidean distance" routes to synthesizable/missing handling and
  does not execute.
- The deterministic verifier catches LLM constraint dropping during tests, but normal successful
  paths are driven by grounding_query_plan, not pattern-only extraction.
- runtime_llm_calls_during_render remains 0.
- cache_miss_during_render remains 0.
- Phase 3.5 through 7.7 guardrails still pass.

Test plan:
- Unit tests for GroundingQueryPlan schema validation.
- Fake-LLM tests that emit correct typed plans for second farthest, closest/farthest answer,
  red-door distance, green-door existence, distance-value selection, and ranked display.
- Fake-LLM negative tests where constraints are dropped; validator must reject.
- Live eval probe for OpenRouter that prints the raw typed query plan for a small suite of
  language variants.
- Station integration tests proving valid query plans compose through Phase 7.7.

Implemented:
- jeenom/schemas.py:
  - GroundingQueryPlan dataclass.
  - grounding_query_plan field on OperatorIntent.
  - JSON schema support for structured LLM query-plan output.
- jeenom/llm_compiler.py:
  - LLM prompt now asks for grounding_query_plan for door/distance/ranked/ordinal queries.
  - LLM prompt now instructs multi-answer ranked queries such as closest+second_closest to use
    typed answer_fields instead of a single degraded ordinal.
  - LLM prompt now maps clear/delete/forget delivery-target requests to a narrow
    knowledge_update with delivery_target=null.
  - SmokeTestCompiler emits query plans for:
    - second/farthest ordinal requests
    - closest/farthest answer requests
    - color-specific distance questions
    - color-specific existence questions
    - distance-value selection
    - ranked door displays
    - closest + second closest multi-answer displays
    - active-claim references such as "go to that"
- jeenom/operator_station.py:
  - Validated grounding_query_plan is now the primary path before regex/verifier fallback.
  - Query plans route through CapabilityMatcher and CapabilityRegistry.
  - Valid plans compose through Phase 7.7 ranked claims and existing go_to_object execution.
  - Constraint-preservation validation rejects dropped ordinals, flipped closest/farthest
    semantics, dropped explicit colors, and dropped explicit metrics.
  - If a typed semantic grounding plan needs a distance metric but omits it, the station stores
    a pending clarification and resumes the original query after the operator answers
    "manhattan".
  - Multi-answer fields such as closest, farthest, second_closest, and second_farthest are
    composed from ranked ActiveClaims.
  - LLM-emitted delivery_target=null updates clear durable target knowledge instead of being
    treated as unsupported.
- jeenom/primitive_library.py:
  - Added grounding.claims.last_grounded_target as an implemented grounding handle over
    session-scoped ActiveClaims.
- tests/test_jeenom_schemas.py:
  - GroundingQueryPlan schema tests.
  - OperatorIntent accepts grounding_query_plan.
  - Invalid ranked plan shape is rejected.
- tests/test_jeenom_minigrid.py:
  - TestLLMSemanticQueryPlanner covers fake-LLM query plans for second farthest,
    bad dropped-ordinal plans, red-door distance, green-door existence, and distance-value
    execution.
  - Tests cover semantic metric clarification/resume and closest+second_closest answer
    composition.
- evals/operator_query_plan_probe.py:
  - Prints raw query plans from the operator-intent compiler.
  - By default this is a live OpenRouter probe; --allow-fallback only verifies eval wiring.
  - Includes a closest+second_closest case.

Proved:
- A typed plan can drive "second farthest" without relying on pattern-only composition.
- A bad LLM plan that turns "second farthest" into ordinal=1 is rejected before execution.
- "how far is the red door" is answered from SceneModel/ranked grounding, not from
  delivery_target status.
- "is there a green door" is answered from SceneModel/ranked grounding.
- "go to the door with distance 7" can execute from a typed distance-value plan.
- Follow-up navigation such as "go to that" resolves through the most recent grounded
  ActiveClaims target instead of stale durable delivery-target knowledge.
- "go to that" is no longer handled by a local pronoun regex shortcut; the compiler emits
  primitive_handle=grounding.claims.last_grounded_target and the station resolves that handle.
- A missing metric in a semantic ranked query is now a real pending clarification, so an answer
  like "use manhattan distance" can resume the original query.
- "which door is closest and second closest?" composes both answers from the ranked grounding
  claim instead of failing validation or dropping the second constraint.
- runtime_llm_calls_during_render remains 0.
- cache_miss_during_render remains 0.
- Full JEENOM suite passes.

Validation:
- python evals/operator_query_plan_probe.py --allow-fallback
- python -m pytest -q tests/test_jeenom_minigrid.py -k "LLMSemanticQueryPlanner or GroundingResultComposition"
- python -m pytest -q tests/test_jeenom_schemas.py tests/test_jeenom_minigrid.py

Out of scope:
- Primitive synthesis
- Euclidean implementation
- non-door objects
- left/right/front/behind
- continuous-world execution
- pickup/open/unlock
- blocked-path replan
- removing the verifier entirely; it remains a safety backstop until the planner is proven.

### Phase 7.8 — Validated Grounding Primitive Synthesis
Status: done.

Goal:
Generate missing pure scene/query primitives when a grounded operator request requires a
deterministic primitive that does not exist yet.

Example:
- operator: "go to the closest door using Euclidean distance"
- LLM emits a validated TargetSelector with distance_metric=euclidean
- station detects that Euclidean ranking/selection is not available
- system synthesizes a pure scene/query primitive
- system tests the primitive against a contract and fixture scenes
- system registers the primitive
- station grounds the target using the new primitive
- existing go_to_object execution runs after grounding

Scope:
- pure deterministic grounding/query primitives only
- no robot actuation primitives
- no unsafe filesystem/runtime mutation without validation
- generated primitive must be inspectable and testable
- primitive must have an explicit input/output contract
- primitive must pass unit tests before registration
- failed synthesis or failed validation must not execute the task

Implemented:
- jeenom/primitive_synthesizer.py: SynthesizerBackend ABC, SmokeTestSynthesizer (always
  refuses), LLMSynthesizer (calls OpenRouter, falls back on missing API key).
  - Rejects generated code with disallowed imports (only math and typing permitted).
  - No substrate imports (AST-verified by probe).
  - build_synthesizer(compiler_name) factory; default_synthesizer = SmokeTestSynthesizer.
- jeenom/primitive_validator.py: PrimitiveValidator, ValidationFixture, ValidationResult.
  - Executes generated code in a restricted namespace (no env, no MiniGrid, no numpy).
  - EUCLIDEAN_FIXTURES: 7 deterministic SceneModel fixture tests covering empty scene,
    single door, ordering correctness, color filter, exclude_color, and the critical
    "distance_is_euclidean_not_manhattan" fixture that catches Manhattan masquerading as Euclidean.
  - default_validator = PrimitiveValidator().
  - No substrate imports (AST-verified by probe).
- jeenom/capability_registry.py: register_synthesized(handle, fn) promotes a synthesizable
  spec to implemented in-place; get_synthesized_callable(handle) returns the callable.
- jeenom/operator_station.py:
  - self.synthesizer and self.validator added to session state.
  - _arbitrate_gap: when decision=synthesize and spec.safe_to_synthesize=True, calls
    _try_synthesize_primitive before falling back to the synthesize refusal message.
  - _try_synthesize_primitive: synthesize → validate → register → re-route to
    _execute_synthesized_grounding. On validation failure, returns an honest operator
    message without executing. On synthesis refusal, returns None (falls through).
  - _try_synthesize_primitive now runs one bounded repair attempt after validation failure,
    passing the exact validation error and previous candidate back to the synthesizer. A
    second failure still does not register or execute anything.
  - _execute_synthesized_grounding: calls the registered callable, writes ranked claims,
    then either returns task_instruction (go_to_object) or a display result.
  - ground_target_selector: checks get_synthesized_callable before refusing unknown metrics.
  - _ground_with_synthesized_callable: runs a synthesized fn against current SceneModel,
    writes ranked claims, returns a grounded dict result compatible with existing paths.
- evals/primitive_synthesis_probe.py: 20 checks covering AST imports, SmokeTest refusal,
  LLM fallback, validator pass/reject/import-reject, registry promotion, synthesized
  callable output, end-to-end session synthesis with fake transport, second-call reuse,
  syntax-error repair, and golden-path preservation.

Success criteria:
- "go to the closest door using Euclidean distance" detects missing Euclidean grounding support.
- A candidate Euclidean ranking primitive is generated.
- The candidate is validated with deterministic tests.
- The primitive is registered only after validation.
- The selector is grounded through the registered primitive.
- The final task still uses the existing go_to_object pipeline.
- runtime_llm_calls_during_render remains 0.
- cache_miss_during_render remains 0.
- Phase 3.5 through 7.7 guardrails still pass.

Validation:
- python evals/primitive_synthesis_probe.py

Technical debt:
- LLMSynthesizer replicates transport code from LLMArbitrator and LLMCompiler; a shared
  transport module should be extracted.
- Synthesized primitives are registered in-memory per session; they do not persist across
  restarts. A durable synthesis cache would allow re-use without re-synthesis.
- FIXTURE_SETS only covers closest_door.euclidean.agent; new primitive types need their
  own fixture sets added to primitive_validator.py.
- Synthesis is triggered only from the arbitration path (synthesize decision). The
  GroundingQueryPlan path (Phase 7.75) does not yet trigger synthesis when the required
  primitive is synthesizable.
- The repair loop is intentionally bounded to one retry. Future phases need a richer
  repair protocol with operator-visible diff/inspection before registration.

Out of scope:
- motion-control primitive synthesis
- pickup/open/unlock synthesis
- non-door object families unless already supported by the scene schema
- continuous-world execution
- exploration/replan
- clarification Q/A

### Phase 7.9 — Collaborative Capability Composition
Status: in progress.

Goal:
Close the capability gap through dialogue. When the system detects a missing primitive that
could be composed from existing ones, it proposes the composition to the operator, waits for
approval, then synthesizes and registers the new primitive. The operator is the specification;
the system is the implementer.

Problem this solves:
Phase 7.8 synthesizes silently — the LLM writes code and the system registers it without
the operator knowing what was built or why. This is fine for well-defined primitives like
euclidean distance, but breaks down for anything ambiguous. The operator may want
"closest non-red door by euclidean distance" composed differently than the system assumes.
Silent synthesis of the wrong thing is worse than asking.

The collaborative loop:
1. Arbitrator detects a gap and identifies which existing primitives could cover parts of it.
2. Instead of synthesizing immediately, the system explains the gap and proposes a composition
   plan in natural language: "I don't have that capability, but I have X and Y. I can combine
   them to build Z. Should I?"
3. Operator approves, rejects, or redirects ("yes", "no", "use euclidean not manhattan").
4. On approval, synthesis + validation + registration runs as in Phase 7.8.
5. The new primitive is immediately available for the current and future turns.

Architecture:
- New pending state: pending_synthesis_proposal — stores the proposed handle, description,
  and which existing primitives it composes from.
- Arbitrator emits a new command kind: synthesis_proposal, with a natural-language
  explanation and the proposed composition.
- handle_pending_synthesis_proposal() resolves "yes/no/redirect" operator responses.
- On approval: _try_synthesize_primitive runs as in 7.8.
- On rejection or redirect: pending state is cleared; operator redirect is fed back into
  the arbitration loop as a new utterance.
- The proposal message must name the existing primitives being composed, the new handle
  being created, and what the validator will test.

Scope:
- Grounding primitives only. No actuation or sensing synthesis.
- Composition from primitives already in the registry (implemented or synthesizable).
- Single-turn approval ("yes") or rejection ("no") or redirect (new utterance).
- Proposal is one natural-language message — no multi-step negotiation yet.

Out of scope:
- Multi-turn negotiation over the composition design.
- Operator-authored fixture specification.
- Synthesis of action or sensing primitives.
- Persistent primitive library updates across restarts.

Success criteria:
- "go to the closest door using euclidean distance" → system proposes composing from
  existing grounding primitives → operator says "yes" → primitive synthesized, validated,
  registered → task executes.
- Operator says "no" → pending state cleared, system returns to READY without executing.
- Operator redirects (new utterance) → pending cleared, new utterance handled normally.
- Proposal names the handle being built and similar implemented primitives.
- Second call after synthesis uses registered primitive directly — no re-proposal.
- reset clears pending_synthesis_proposal.
- Golden path "go to the red door" fires no proposal.
- runtime_llm_calls_during_render remains 0.
- Phase 3.5 through 7.8 guardrails still pass.

Implemented:
- jeenom/schemas.py: ArbitrationDecision gains proposed_handle and proposed_description fields.
  Arbitrator sets these when it determines a new primitive can be synthesized from the SceneModel
  API — handle need not be pre-declared in the registry.
- jeenom/capability_arbitrator.py: LLMArbitrator prompt rewritten. Arbitrator now receives the
  full SceneModel API surface (scene.find, agent_x/y/dir, manhattan_distance_from_agent, math).
  Instructs LLM to reason: "can this request be expressed as fn(scene, selector) → ranked list?"
  If yes → synthesize with proposed_handle + proposed_description. No longer limited to
  synthesizable_handles — any pure spatial/logical computation is synthesizable.
  arbitration_decision_json_schema updated with proposed_handle/proposed_description fields.
- jeenom/capability_registry.py: register_dynamic(handle, description, fn) — creates a brand-new
  registry entry for a handle that was not pre-declared. Used by the dynamic synthesis path.
- jeenom/operator_station.py: PendingSynthesisProposal dataclass (handle, original_utterance,
  intent, cap_match, similar_handles, proposed_description).
- jeenom/operator_station.py: self.pending_synthesis_proposal added to session state.
- jeenom/operator_station.py: _propose_synthesis() — accepts proposed_description for dynamic
  handles. Builds SYNTHESIS PROPOSAL message. Sets pending state.
- jeenom/operator_station.py: _find_similar_implemented() — returns up to 2 implemented
  primitives in the same layer as the proposed handle.
- jeenom/operator_station.py: handle_pending_synthesis_proposal() — resolves yes/no/redirect.
  yes → _try_synthesize_primitive → execute_command on success.
  no → clears pending, returns refusal message.
  redirect → clears pending, re-enters handle_utterance with new utterance.
- jeenom/operator_station.py: _arbitrate_gap synthesize branch prefers arbitrator's
  proposed_handle/proposed_description; falls back to synthesizable_handles if absent.
- jeenom/operator_station.py: _try_synthesize_primitive uses register_dynamic for handles
  not in registry; register_synthesized for pre-declared synthesizable ones.
- jeenom/operator_station.py: handle_utterance checks pending_synthesis_proposal before
  pending_clarification. synthesis_proposal added to dispatch. reset() clears proposal.
- evals/collaborative_synthesis_probe.py: 16 checks covering proposal, yes/no/redirect,
  second-call, reset, and golden path.
- jeenom/primitive_synthesizer.py: prompt hardened to require a complete syntactically valid
  function, no Markdown fences, and exact def-line conformance.
- jeenom/primitive_synthesizer.py / jeenom/operator_station.py: synthesis approval now supports
  synthesize → validate → repair once → validate → register. This directly handles malformed
  LLM candidates such as mismatched brackets without weakening the safety gate.
- jeenom/primitive_validator.py: corrected the Euclidean-vs-Manhattan regression fixture so
  Manhattan masquerading as Euclidean is actually rejected.
- jeenom/primitive_library.py: added grounding.all_doors.ranked.euclidean.agent as a
  pre-declared synthesizable ranked-list handle. This is the exact capability needed for
  "Euclidean distance to all doors" questions.
- jeenom/capability_registry.py: synthesizing grounding.closest_door.euclidean.agent now also
  promotes the equivalent grounding.all_doors.ranked.euclidean.agent handle because both share
  the same ranked-list callable contract. The registry still uses exact handles; it does not
  weaken closest into ranked at match time.
- jeenom/intent_verifier.py: ranked, ordinal, and superlative distance signals now inject
  grounding.all_doors.ranked.<metric>.agent instead of the legacy unregistered
  grounding.ranked_doors.<metric>.agent form.
- jeenom/operator_station.py: ranked composition and display now use the primitive handle's
  metric and synthesized callable, so a post-synthesis "give me Euclidean distance to all the
  doors" displays a Euclidean ranking rather than falling back to Manhattan or refusing.
- evals/primitive_synthesis_probe.py: added repair-loop checks proving malformed first code is
  not registered, the validation error is sent back, and the repaired second candidate can be
  registered.
  - Also checks that the ranked Euclidean alias is promoted and displayable after synthesis.

Eval:
- Command:
  python evals/collaborative_synthesis_probe.py
- Command:
  python evals/primitive_synthesis_probe.py
- Command:
  python evals/primitive_composition_probe.py
- Requires gymnasium + minigrid environment.

Current proven behavior:
- Operator-approved synthesis does not register malformed code.
- If the first candidate fails validation, JEENOM asks the synthesizer for one repair using the
  exact compiler/validation failure.
- If the repaired candidate passes, it is registered and the original grounding/task resumes.
- If the repaired candidate fails, JEENOM returns an honest failure and does not execute.
- After Euclidean synthesis, both the specific closest-door handle and the all-doors ranked
  Euclidean handle are available in the registry.
- Follow-up requests like "give me the Euclidean distance to all the doors" route through the
  exact ranked Euclidean handle and display a Euclidean ranking.
- Added a second synthesis target type: claims-filter primitives with signature
  fn(entries, condition) -> list[GroundedDoorEntry]. These operate on typed ActiveClaims
  rather than SceneModel.
- Added claims.filter.threshold.euclidean and claims.filter.threshold.manhattan as
  pre-declared, safe-to-synthesize claims primitives. They are parametric: threshold and
  comparison are supplied through condition, not baked into the generated function.
- CapabilityRegistry now indexes the claims layer from primitive_library.py, and the schema
  accepts claims as a first-class primitive layer.
- The LLM/arbitrator prompts now distinguish scene-grounding synthesis from claims-filter
  synthesis. Threshold filters over already-ranked claims should propose claims.filter.threshold.<metric>.
- Operator approval now dispatches claims.* handles through the claims-filter synthesizer and
  validator path, not the SceneModel grounding signature.
- Multiple matches from a claims filter create a normal candidate clarification instead of
  silently selecting a target.
- Added evals/claims_filter_synthesis_probe.py proving: ranked claims -> threshold synthesis
  proposal -> approval -> claims-filter validation/register -> multiple-match clarification
  -> selected target executes with runtime_llm_calls_during_render=0 and cache_miss_during_render=0.
- Fixed proposal handoff bug: arbitration-created claims-filter proposals now preserve
  proposed_condition (threshold, comparison, metric) through operator approval. This prevents
  threshold filters from defaulting to threshold=0 or the wrong metric after synthesis.
- Claims-filter composition now supports selecting from a filtered set using order/ordinal.
  Example: "highest Euclidean distance below 10" is represented as filter below 10, then
  order=descending, ordinal=1. The station selects that target if unique, clarifies on ties,
  and never silently degrades the request.
- Exact synthesizable registry matches now override unreliable arbitration refusals. If the
  CapabilityMatcher says a required handle is present and safe_to_synthesize, the station
  proposes synthesis instead of accepting an LLM arbitrator refusal.


### Phase 7.95 — Typed RequestPlan and ReadinessGraph
Status: in progress.

Goal:
Introduce a first-class RequestPlan schema and ReadinessGraph so operator requests are
decomposed into typed steps before clarification, synthesis, memory lookup, query answering,
or task execution. This is the architectural layer that prevents the station from guessing
from a raw utterance or from a single overloaded OperatorIntent.

Relationship to Phase 7.9:
This is separate from Phase 7.9.
Phase 7.9 is the collaborative capability-composition loop: propose, approve, synthesize,
validate, register, and reuse a missing primitive.
Phase 7.95 decides what needs to happen in the first place. It turns the operator request into
a typed plan, evaluates every plan node against the registry, memory, ActiveClaims, SceneModel,
and synthesis policy, then chooses the next safe action.

Problem this solves:
Right now the station is split across OperatorIntent, IntentVerifier, CapabilityMatcher,
CapabilityArbitrator, ActiveClaims, and several station fallbacks. That makes behavior janky:
the system can recognize a phrase, propose a primitive, answer a query, or execute a task, but
the decomposition is implicit and scattered. Complex requests like "go to the door with the
highest Euclidean distance below 10" need a graph:
1. produce/refresh Euclidean ranked door claims,
2. filter claims below threshold 10,
3. select highest remaining candidate,
4. clarify if tied or ambiguous,
5. convert the grounded target into go_to_object,
6. prewarm and run cached runtime.

Core rule:
Raw operator utterances are never the final execution object. The station must compile or
derive a RequestPlan, validate that it preserves the utterance, and run the ReadinessGraph over
the plan. Clarification, synthesis proposals, query answers, and execution all come from graph
verdicts, not ad hoc phrase handling.

RequestPlan schema:
- request_id
- original_utterance
- objective_type: task | query | knowledge_update | control
- objective_summary
- steps: ordered/dependency-aware RequestPlanStep list
- preservation_signals: superlative, ordinal, cardinality, negation, metric, threshold,
  object_type, color, reference terms
- expected_response: execute_task | answer_query | ask_clarification | propose_synthesis |
  update_memory | refuse

RequestPlanStep schema:
- step_id
- layer: task | grounding | claims | sensing | action | memory | answer | control
- operation: rank | filter | select | ground | execute | answer | update | reset | refuse
- required_handle
- implementation_status expectation: implemented | synthesizable | planned | unsupported
- inputs
- outputs
- depends_on
- constraints
- tie_policy: clarify | display_ties | fail
- memory_reads
- memory_writes
- scene_fingerprint requirement if using ActiveClaims

Example RequestPlan:
Utterance:
"go to the door that has the highest Euclidean distance below 10"

Plan:
1. rank_doors_euclidean
   - layer: grounding
   - operation: rank
   - required_handle: grounding.all_doors.ranked.euclidean.agent
   - output: active_claims.ranked_scene_doors
2. filter_below_10
   - layer: claims
   - operation: filter
   - required_handle: claims.filter.threshold.euclidean
   - input: active_claims.ranked_scene_doors
   - constraints: comparison=below, threshold=10
3. select_highest
   - layer: claims
   - operation: select
   - constraints: order=descending, ordinal=1
   - tie_policy: clarify
4. execute_go_to_object
   - layer: task
   - operation: execute
   - required_handle: task.go_to_object.door
   - input: grounded door target

ReadinessGraph:
- Builds a node for every RequestPlanStep.
- Evaluates each node against:
  - CapabilityRegistry
  - primitive_library-derived primitive specs
  - OperationalMemory durable knowledge
  - episodic memory
  - ActiveClaims with scene fingerprint/staleness
  - current SceneModel
  - synthesis policy
  - PlanCache/JIT prewarm requirements for runtime execution
- Emits per-node status:
  - executable
  - needs_clarification
  - synthesizable
  - missing_skills
  - unsupported
  - stale_claims
  - blocked_by_dependency
- Emits one graph-level next action:
  - ask one clarification question
  - refresh/recompute claims
  - propose synthesis
  - answer query
  - execute task
  - update memory
  - refuse with exact blocking node and reason

Memory policy:
- Durable Knowledge stores operator-taught facts such as delivery_target.
- Episodic memory stores last_request_plan, last_grounded_target, last_task, last_result.
- ActiveClaims stores computed scene facts, tied to a scene fingerprint and provenance handle.
- RequestPlan steps must declare memory_reads and memory_writes before execution.
- Computed analysis does not become durable truth unless explicitly promoted by the operator.

Clarification policy:
- Clarification questions are generated from incomplete or ambiguous RequestPlan nodes.
- Supported initial cases:
  - missing metric for closest/farthest/ranked distance query
  - tied target for task execution
  - multiple filtered candidates for a task
  - missing object type when the registry supports more than one candidate family
- Status/cache/capability queries must not destroy the pending RequestPlan.
- A new task request cancels the pending plan unless it is clearly an answer to the pending
  clarification.

Implementation plan:
- Add RequestPlan, RequestPlanStep, ReadinessGraph, ReadinessNode, and ReadinessVerdict schemas.
- Add a request planner module that converts validated OperatorIntent + semantic query plan +
  active context into RequestPlan.
- Keep LLM compilation separate from readiness arbitration. The LLM may propose a plan, but the
  station validates it before use.
- Replace direct capability arbitration from raw utterance with ReadinessGraph arbitration over
  RequestPlan nodes.
- Route synthesis proposals through graph nodes rather than one-off handle matching.
- Route claims-filter and grounding composition through graph dependencies.
- Teach station responses to explain the blocking node: missing primitive, ambiguous selector,
  stale claims, unsupported action, or pending synthesis.

Files likely touched:
- jeenom/schemas.py
- jeenom/request_planner.py
- jeenom/readiness_graph.py
- jeenom/operator_station.py
- jeenom/llm_compiler.py
- jeenom/capability_arbitrator.py
- tests/test_jeenom_schemas.py
- tests/test_jeenom_minigrid.py
- evals/request_plan_probe.py

Success criteria:
- "go to the door with the highest Euclidean distance below 10" becomes a multi-step
  RequestPlan instead of a single guessed intent.
- If Euclidean ranking is missing but synthesizable, the graph proposes that primitive first.
- If threshold filtering is missing but synthesizable, the graph proposes that primitive next.
- If multiple filtered candidates remain, the graph asks which candidate to use and does not
  execute silently.
- If a query can be answered from fresh ActiveClaims, the graph answers without rerunning
  unnecessary sensing.
- If ActiveClaims are stale, the graph refreshes or reports why it cannot.
- "pick up the key" produces an unsupported/missing action node and does not execute.
- Status/capability queries answer from the registry and graph state.
- Golden path remains unchanged.
- runtime_llm_calls_during_render remains 0.
- cache_miss_during_render remains 0.

Evals:
- evals/request_plan_probe.py
  - prints RequestPlan steps, dependencies, required handles, preservation signals, and graph
    verdicts for representative utterances.
- Cases:
  - "what doors do you see?"
  - "how far are all doors from you?"
  - "what is the farthest door?"
  - "go to the closest door"
  - "go to the door with the highest Euclidean distance below 10"
  - "pick up the key"
  - "go to that" after a grounded claim

Implemented:
- Added typed RequestPlan, RequestPlanStep, ReadinessNode, and ReadinessGraph schemas.
- Added jeenom/request_planner.py to decompose validated OperatorIntent + grounding query plans
  into a side-effect-free RequestPlan.
- Added jeenom/readiness_graph.py to evaluate plan nodes against CapabilityRegistry,
  ActiveClaims validity, and dependency status.
- OperatorStationSession now records last_request_plan and last_readiness_graph into episodic
  memory on operator-intent handling. This makes the plan/graph inspectable without changing
  rendered runtime execution.
- Added evals/request_plan_probe.py, which prints the plan and readiness graph for key cases.
- Added regression tests for schema validation, thresholded Euclidean plan decomposition,
  first blocking synthesizable node detection, and station episodic plan/graph recording.

Current proven behavior:
- "go to the door with the highest Euclidean distance below 10" decomposes into:
  rank Euclidean doors → threshold filter → select highest → execute go_to_object.
- The ReadinessGraph identifies grounding.all_doors.ranked.euclidean.agent as the first
  synthesizable blocking node.
- "go to the closest door" produces a needs_clarification graph before execution.
- "pick up the key" produces an unsupported task node and refuses.
- Status/scene query plans are executable answer plans.
- The existing Phase 3.5/Phase 4 guardrails still pass.

Out of scope:
- New robot actions.
- New object families.
- Continuous-world execution.
- Mid-run correction.
- Broad conversational Q/A.
- Primitive synthesis beyond the existing safe synthesis policy.
- Replacing PlanCache or runtime Sense/Spine execution.


### Phase 8 — GoToObject/general object variant
Status: planned.

Goal:
Extend the existing go_to_object recipe beyond doors to supported MiniGrid object types.


### Phase 8.5 — Readiness-only transfer demo
Status: planned.
Same Understanding, different primitive set. MiniGrid executable; Jackal/Nav2 partial/executable depending primitives.

### Phase 9 — Failure/replan/operator ask
Status: planned.
Handle missing primitive, ambiguity, no path, blocked task. Ask operator or propose fallback.

### Phase 10 — Harder MiniGrid integration
Status: later.
MultiRoom/DoorKey/KeyCorridor only after adding exploration, toggle/open, pickup, unlock, and replan primitives.

### Phase 11 — Port
Status: later.
First port primitive sets/readiness. Then port Sense/Spine bindings. Then run on real/sim robot substrate.
