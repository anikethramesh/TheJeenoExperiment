# Phase 12D — Substrate Coupling Leak Audit

**Date:** 2026-06-13  
**Status:** v2 (curriculum-touching leaks removed; Phase 13 gate cleared)

The audit separates two orthogonal problems: **substrate coupling** (on the critical path to
substrate-independence) and **structural bloat** (orthogonal to both proofs, parked). Only the
coupling table drives phase ordering.

---

## Coupling Table

Per site: `{cheap | structural}` × `{curriculum-touching | not}` — the product decides phase order.

| Site | Lines | Classification | Notes |
|---|---|---|---|
| `llm_compiler.py` — fast-path grammar | ~13 | **structural × curriculum-touching** ✅ removed | `color_pattern` derived from `capability_manifest.symbol_mappings.color_index` |
| `llm_compiler.py` — LLM prompt strings | ~100 | cheap × not curriculum-touching | Door/color examples in prompt text; swap for manifest vocabulary |
| `operator_station.py` — MiniGrid imports + default env | ~8 | cheap × not curriculum-touching | `build_minigrid_runtime_package` imported directly; `env_id` defaults to MiniGrid |
| `operator_station.py` — `_startup_warmup_instruction` | ~22 | cheap × curriculum-touching ✅ removed | Fallback now derives default target from `manifest.symbol_mappings.color_index` + `object_vocabulary`; returns `None` if manifest has no vocabulary (skips warmup) |
| `operator_station.py` — request parsing (color/door) | ~30 | cheap × not curriculum-touching | Color and door-type extraction baked into station; should derive from manifest symbol_mappings |
| `primitive_library.py` — door/grounding primitives | ~60 | cheap × not curriculum-touching | `visible_doors`, `closest_door.*`, `unique_door.*`, `all_doors.*` live in a generic file; move to `minigrid_primitive_library.py` |
| `sense.py` — `MiniGridSense` class | 393 total / ~100 coupled | structural × not curriculum-touching | Whole class is substrate-bound by design; coupling is intentional; replacement requires AI2-THOR adapter work |
| `spine.py` — `MiniGridSpine` class | 355 total / ~80 coupled | structural × not curriculum-touching | Same as sense.py; substrate-bound by design |
| `request_planner.py` — `rank_scene_doors` step_id | ~8 | cheap × not curriculum-touching | Hardcoded step_id string; derive from manifest declared operations |

---

## Ordering Verdict

**Must remove before Phase 13 (curriculum-touching) — ✅ both removed:**

### `llm_compiler.py` — fast-path grammar (structural + curriculum-touching) ✅

The primary structural leak. Two sites:

1. **Line 263** — `r"go to the (?P<color>\w+) (?P<object_type>door)"` — MiniGrid-specific fast-path
   regex that short-circuits LLM compilation for navigation intents. This pattern is MiniGrid
   vocabulary baked into a generic compiler.

2. **Line 1290** — `f"go to the {color} door"` — canonical instruction reconstruction from parsed
   targets uses MiniGrid phrasing.

**Why structural:** Removal requires replacing the hardcoded `go to the <color> door` grammar with
`OperationalContext`-derived patterns read from the manifest's `symbol_mappings`. Not a surgical
edit — it requires a principled grammar derivation path from the manifest at compiler init.

**Why curriculum-touching:** Phase 13 proves steering under partial observability. Steering means
the operator's intent is correctly compiled under various PO scenarios (target not yet visible,
ambiguous color, etc.). If the compiler falls back to a hardcoded MiniGrid grammar, the compilation
path during PO gaps is substrate-coupled. The steering signal then measures MiniGrid-grammar
quality, not genuine intent compilation. The curriculum must not be built on top of this.

**Go/no-go:** This is a `structural` flag. Confirm the grammar derivation design is achievable
before committing Phase 13 scope. Estimated effort: medium (2–3 days for manifest-driven grammar
derivation + re-verification of fast-path coverage).

### `operator_station.py` — `_startup_warmup_instruction` (cheap + curriculum-touching) ✅

`_startup_warmup_instruction` (lines 517–530) seeded the plan cache with `"go to the red door"` as
a hardcoded fallback. At startup, Phase 13 curriculum sessions warm the cache using this method; if
the warmup string is substrate-hardcoded, the first cache hit in the curriculum is a MiniGrid
artefact rather than a manifest-derived plan.

**Removed:** fallback now reads `manifest.symbol_mappings.color_index` + `object_vocabulary` to
pick the first declared color and object type. If the manifest has no vocabulary, returns `None`
and `prewarm_known_task_family` skips the warmup silently. The compiler manifest (`_compiler_manifest`)
also now includes `symbol_mappings` so `SmokeTestCompiler.compile_operator_intent` can derive
`color_pattern` from it.

---

**Remove in Phase 14 cheap-removal pass (not curriculum-touching):**

- `llm_compiler.py` prompt strings (~100 lines): swap door/color vocabulary examples for
  manifest-derived examples at session init
- `operator_station.py` MiniGrid imports + default env (~8 lines): inject runtime_package;
  remove `minigrid_runtime_package` import from generic station module
- `operator_station.py` request parsing with color/door (~30 lines): derive from manifest
  `symbol_mappings.object_index` / `color_index`
- `primitive_library.py` door/grounding primitives (~60 lines): move to
  `minigrid_primitive_library.py`; register via manifest at adapter init
- `request_planner.py` `rank_scene_doors` step_id (~8 lines): derive step_id from manifest
  declared operations

**Structural leaks deferred to Phase 15 (substrate independence work):**

- `sense.py` `MiniGridSense` + `spine.py` `MiniGridSpine`: coupling is intentional;
  replacement happens when the AI2-THOR adapter is built. The Phase 14 spike informs what the
  `SubstrateAdapter` interface must generalize. Any coupling found in the spike that blocks the
  second substrate becomes a go/no-go for Phase 15 scope.

---

## Bloat Worklist (separately parked — orthogonal to both proofs)

`operator_station.py`: 5,613 lines / 178 methods. Of these, only ~67 lines are substrate-coupled
(covered in the coupling table above). The remaining ~5,546 lines are substrate-independent
orchestration logic (intent dispatch, plan caching, clarification handling, mission flow, synthesis,
teach/forget paths, knowledge channel, cortex session delegation, etc.).

**This is not a coupling problem.** De-bloating does not remove leaks and does not unblock either
proof. Deferred to **Phase 16 — Operational Hardening** in full.

**Hard prerequisite:** any `operator_station.py` extraction — in Phase 16 or pulled forward — is
gated by a decomposition design (target modules, shared-state map, ordered green-able extraction
sequence) written and reviewed first. No station code moves before that design exists.

Bloat worklist sketch for Phase 16 planning (not for ordering decisions now):

| Candidate module | Methods | Notes |
|---|---|---|
| Intent dispatch + knowledge routing | ~20 | `TurnOrchestrator` already exists; station could delegate fully |
| Plan cache + reuse | ~15 | `PlanReuseCache` already separate; station holds glue only |
| Clarification / synthesis / definition state | ~30 | Pending-state machine; self-contained |
| Mission flow + continuation | ~25 | `MissionCortex` already separate |
| Auth + ticket management | ~12 | `CommandAuthority` already separate |
| Core `handle_utterance` orchestration | ~15 | Should be thin coordinator after extraction |

---

## Summary

| Category | Remove before Phase 13 | Remove in Phase 14 | Deferred |
|---|---|---|---|
| Structural + curriculum-touching | ~~`llm_compiler.py` grammar~~ ✅ removed | — | — |
| Cheap + curriculum-touching | ~~`operator_station.py` `_startup_warmup_instruction`~~ ✅ removed | — | — |
| Cheap + not curriculum-touching | — | 5 sites listed above | — |
| Structural + not curriculum-touching | — | — | `sense.py`, `spine.py` (Phase 15 spike) |
| Bloat (orthogonal) | — | — | Phase 16 |

**Phase 13 gate: cleared.** Both curriculum-touching leaks removed; 70/70 evals green.
