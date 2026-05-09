## JEENOM Implementation Plan — Capability Ladder

### Phase 0 — MiniGrid smoke test
Status: done.
Prove MiniGrid wrapper, observe/act/render, and simple task execution.

### Phase 1 — Minimal JEENOM vertical slice
Status: done.
Implement CorticalSchema contracts, Cortex, Sense, Spine, WorldModelSample, OperationalEvidence, Percepts, ExecutionContract, ExecutionReport.

### Phase 2 — LLM compiler boundary
Status: done.
LLM compiles typed schema objects. Runtime validates and executes deterministic primitives.

### Phase 3 — JIT template caching + prewarm
Status: done.
Cache ProcedureRecipe, SensePlanTemplate, SkillPlanTemplate. Prewarm before render. No LLM calls or cache misses during rendered control loop.

### Phase 3.5 — Regression/golden test
Status: next.
Freeze the current success case:
- instruction: "go to the red door"
- compiler: llm
- render/prewarm enabled
- task_complete=True
- runtime_llm_calls_during_render=0
- cache_miss_during_render=0
- final action=done

### Phase 4 — Bigger same-task stress test
Status: next.
Run larger GoToDoor environment. Same task, same recipe, longer path. No Q/A yet.

### Phase 5 — CLI operator interface
Status: planned.
Support interactive instruction/correction loop. CLI only.

### Phase 6 — Few-shot memory reuse
Status: planned.
Run 1 stores knowledge. Run 2 uses it, e.g. "go there again."

### Phase 7 — GoToObject/general object variant
Status: planned.
Test same recipe over non-door objects.

### Phase 8 — Readiness-only transfer demo
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