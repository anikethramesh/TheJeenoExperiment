# ORPI v0.1 — Open Robotics Primitive Interface

**Status: v0.1, explicitly unstable.** Freeze to v1 only after the interface survives the
Phase 15 cross-substrate port (the validation event). Standards extracted from n=1 substrates
ossify the wrong abstractions; v0.1 exists to be broken by the second substrate, deliberately.

This document is the authoritative contract/manifest/procedure/trace reference for ORPI. It is built
and versioned alongside the code. It does not track current phase status; the implementation
roadmap lives in [task_plan.md](task_plan.md). The chronological Phase 12 implementation record,
including the decisions and discoveries summarized below, lives in
[task_plan.md](task_plan.md#phase-12---orpi-v01).

---

## 1. What ORPI Is

ORPI is to robot cognition what MCP is to LLM tooling: a typed interface standard between a
cognition layer (JEENOM) and an embodiment (MiniGrid, Jackal, UR5). It has **two halves**:

1. **Inbound — what a substrate exposes to cognition.** Every capability of a robot is published
   as a *primitive* with a machine-readable contract. Cognition never sees hardware, drivers, or
   policies; it sees contracts.
2. **Outbound — what a deployment emits to learning.** Every executed turn produces a *labelled
   episode trace* in a standard format: what was intended, what was grounded, which contracts were
   checked, what executed, what the postcondition verification said, and — on failure — which
   component's contract was violated. This is the supervision artifact that self-improving
   deployment loops consume. No other part of the robotics stack produces this by construction;
   ORPI does.

The grounding obligation is discharged **at the primitive boundary**. Primitives convert reality
into typed claims and intents into physical effects. The cognition layer above provides grounding
*accounting* (custody, validity, arbitration, composition), never grounding itself. A primitive is
swappable precisely when its grounding obligation is fully discharged at its contract.

## 2. Scope and Non-Goals

**In scope:** the contract schema, the manifest schema, the trace schema, registration/conformance
rules, versioning policy.

**Out of scope:** how primitives are implemented (policies, planners, controllers, models —
substrate's business); transport/serialization beyond "JSON-serializable dataclasses" (v0 is
in-process Python; wire protocol is a v1+ question); tasks whose success conditions don't compress
into checkable object-centric predicates (contact-rich, deformable, aesthetic — explicitly fenced
out).

## 3. Primitive Classes

Every primitive declares exactly one class:

| Class | Role | Examples | Cadence home |
|---|---|---|---|
| `sense` | reality → claims | `detect_door`, `get_pose`, `gripper_contact_check` | perception rate |
| `actuation` | approved command → physical effect | `navigate_to`, `open_door`, `pick_up` | control rate (Spine) |
| `meta` | claims → claims (computation over the operational model) | spatial relations (`ahead_of`, `visible`), metric ranking, claim queries | deliberation or perception rate |

`meta` primitives further declare `mode: deterministic | deliberative`:
- **deterministic** meta-primitives may be referenced inside compiled plans and run during execution
  (e.g., egocentric frame transforms, manhattan-distance ranking, derived-claim inference).
- **deliberative** meta-primitives invoke the LLM compiler (recompilation, repair, synthesis). They
  are exception handlers that pause execution. **Compiled plans may never reference deliberative
  meta-primitives.** This is the enforcement point for the no-LLM-in-the-loop invariant.

These three classes are the ORPI taxonomy. `OrpiContract.primitive_type` always returns one of
`{sense, actuation, meta}` via `orpi_primitive_type_for()`.

**Compatibility bridge (v0):** `schemas.PrimitiveSpec.primitive_type` still accepts both the legacy
values (`task | grounding | sensing | action | claims`) and the ORPI values (`sense | actuation |
meta`). The mapping is: `sensing → sense`, `action → actuation`, `task / grounding / claims →
meta`. MiniGrid primitives are authored with legacy values; `OrpiContract` projects them into the
ORPI taxonomy. The hard schema-level remap — where primitives are authored with ORPI types directly
— is deferred until the contract/manifest/trace boundary survives the Phase 15 cross-substrate
port. This is the right sequencing: standardise the interface before forcing every primitive
author to use the new vocabulary.

## 4. The Contract

The contract is the existing `schemas.PrimitiveSpec`, serialized. Fields marked **NEW** are v0
additions; everything else already exists in the repo.

| Field | Meaning | Notes |
|---|---|---|
| `name`, `primitive_type`, `layer`, `description` | identity | `OrpiContract.primitive_type` ∈ {sense, actuation, meta} (mapped from legacy values by `orpi_primitive_type_for()`); `layer` remains the implementation/registry grouping during the v0.1 compatibility bridge |
| `inputs` / `outputs` | typed parameters | continuous params (grasp pose, force threshold) live here — the symbol layer passes them through, never chooses them |
| `preconditions` | claims that must hold, with min confidence | evaluated by ReadinessGraph |
| `postconditions` | **object-centric state deltas** (Δg), not action descriptions | "door(d).state: closed→open", not "arm moved" — this is what makes procedures retargetable across embodiments |
| `postcondition_primitive` | the `sense` primitive that verifies the postcondition | a postcondition is a proposition *plus its checker*; closes the contract system over the primitive vocabulary. `None` = implicit/free (MiniGrid degenerate case) |
| `required_claims` / `produced_claims` | claim kinds consumed/emitted | |
| `units`, `frame_id`, `required_frames` | dimensional and frame contracts | |
| `safety_class` | risk category | drives required claim confidence and verification tier via the manifest risk policy |
| `authority_level` | who may invoke | ticket system input |
| `failure_modes` | **typed** failure outcomes this primitive can emit | each maps to a `FailureOutcome.category`; failures are states, not just unmet postconditions |
| `validation_hooks` | preflight checks before execution | a world-model shadow rollout is a validation hook — this is the ConsequencePredictor socket |
| `substrate_fingerprint` | binds the contract to a substrate version | staleness detection for cached plans |
| `mode` **NEW** | `deterministic \| deliberative` (meta only) | see §3 |
| `cadence` **NEW** | declared execution rate class: `control \| perception \| deliberation` | substrate enforces: nothing at deliberation cadence sits on a control-cadence path |
| `invariant_level` **NEW** | which invariant the primitive preserves: `pose \| contact \| object_state \| intent` | ReadinessGraph metadata for primitive substitution across embodiments |

## 5. The Manifest

One per substrate, registered at adapter init. Extends the existing `register_domain_vocabulary` /
`OperationalContext` pattern:

- `substrate_id`, `substrate_fingerprint`, `orpi_version`
- `object_vocabulary` — the registered object types (exists today)
- `symbol_mappings` — domain constants the substrate owns (e.g., MiniGrid IDX_TO_COLOR/IDX_TO_OBJECT —
  moved here per the Phase 11C partition; partly done today via `register_domain_index_maps`)
- `frames` and `units` registries
- `risk_policy` — table mapping `safety_class` → required claim confidence, required verification
  tier, required validation hooks. **Policy is auditable manifest data, not buried thresholds.**
- `primitives` — list of contracts (§4)
- `bundled_procedures` — optional OEM-vouched procedure contracts (§6)

Registration is **fail-closed in spirit**: validation is permissive only before any manifest is
registered, and a conformance probe must assert each adapter registers at init, so the permissive
window is provably never live.

## 6. Procedures

ORPI v0.1 includes `OrpiProcedure` for vouched-for recipes that a substrate or embodiment vendor
can expose as first-class interface objects. A procedure is not a second recipe hierarchy; it is the
serialized interface view over the existing recipe/plan-cache record.

Serialized fields:

- `name`
- `steps` — ordered primitive references by name plus effect/postcondition metadata
- `declared_postconditions`
- `declared_preconditions`
- `provenance` — `oem | synthesized | operator`
- `safety_class` — max risk class across constituent primitives
- `authority_level` — max authority requirement across constituent primitives
- `substrate_fingerprint`

`OrpiManifest.bundled_procedures` is reserved for OEM-vouched procedures only. Manifest validation
rejects any bundled entry whose provenance is not `oem`, and rejects any procedure step that does
not reference a primitive present in the same manifest. Synthesized and operator procedures may be
recorded and traced, but they are not allowed to masquerade as substrate-bundled OEM capability.

Planner parity rule: a bundled procedure is selectable by declared postcondition alongside primitive
contracts. Once selected, it expands to primitive handles before ticket issuance and readiness/
authority checks, so procedure selection never bypasses per-primitive gates.

## 7. The Trace (Outbound Half)

Every turn emits a `LabelledEpisode`. It is a thin aggregator/serializer over existing trace
machinery (`OperatorIntent`, `RequestPlan`, `ReadinessGraph`, `CommandResult`, `FailureOutcome`,
`CorticalEnvelope`) — not a parallel type tree.

```
intent          — OperatorIntent as compiled (+ verifier verdict)
grounding       — claims consumed, with provenance/confidence/freshness at read time
plan            — RequestPlan + ReadinessGraph verdicts per step
                  + candidate kind/provenance (primitive vs procedure)
authority       — tickets issued, scope, issuer
execution       — per-step CommandResult, incl. typed FailureOutcome
verification    — postcondition_primitive results vs. predicted postconditions  [*]
attribution     — on failure: which contract was violated                        [*]
                  (stale_claim | miscompiled_intent | unmet_postcondition |
                   missing_authority | substrate_fault)
steering        — operator interventions, clarifications, and active steering
                  + KB writes and per-scope KB reuse counters
```

**[*] Closed in Phase 12D.** `verification` and `attribution` were the highest-value fields for
component-level credit assignment — and the most incomplete in v0.

- `verification` now invokes the contract's named `postcondition_primitive` (
  `sensing.parse_grid_objects` for MiniGrid action primitives) and records the checker name. The
  degenerate case (`postcondition_primitive=None`) is labelled explicitly as `"degenerate_boolean"`.
- `attribution` maps `FailureOutcome.category` through the ORPI taxonomy:
  `stuck | progress → unmet_postcondition`, `blocking_claim → stale_claim`,
  `timeout → substrate_fault`. The raw category is preserved alongside.

Rich Δg verification (invoking the checker against a re-observed state rather than the execution
result already in `final_state`) remains future runtime-verification work tracked in
`task_plan.md`.

The trace is the product's audit story ("what did the operator tell it, when, and did the robot
honour it") and the learning story (component-level credit assignment for deployment loops) in one
artifact. Failed episodes are **kept**, with attribution — failures are future skills if properly
labelled.

## 8. Knowledge Scope

Durable knowledge records carry a transfer scope. This is the falsifiable replacement for an
informal information/knowledge/wisdom ladder:

| Scope | Invalidated by | Default for |
|---|---|---|
| `episodic` | scene change | claims only; not stored in `KnowledgeBase` |
| `site` | site/map change | operator-taught facts and constraints |
| `embodiment` | substrate fingerprint change | OEM procedures and recipes naming morphology-specific primitives |
| `universal` | task ontology change | recipes expressed purely in postcondition/effect vocabulary |

`KnowledgeBase` rejects `episodic` records. `derive_scope(record, manifest)` deterministically
derives broader scopes from ORPI contracts: direct substrate-fingerprinted primitive references are
`embodiment`; effect-only recipes whose steps stay inside the manifest effect vocabulary can be
`universal`; OEM bundled procedures are always `embodiment`.

`KnowledgeChannel` is the gated write/read surface used by station/orchestration paths. It enforces
writer identity by scope, emits durable KB writes into `LabelledEpisode.steering.knowledge`, and
records per-scope reuse counters for the planned curriculum/reuse evaluation.

## 9. Conformance

A substrate is ORPI-v0.1 conformant when:
1. Every capability is registered through a contract; no side-channel capabilities.
2. The manifest is registered at init (probe-enforced).
3. All postconditions are object-centric deltas; all actuation primitives with `safety_class ≠ query`
   name a `postcondition_primitive`.
4. Cadence declarations are honoured: no actuation contract has non-control cadence; no sense
   contract has non-perception cadence; no control-cadence contract is deliberative. (v0 probe:
   data check over manifest contracts. Full AST static check — tracing Spine call paths — is
   deferred to v1 when the second substrate makes it worth the investment.)
5. Every executed turn emits a `LabelledEpisode`.
6. No compiled plan references a `deliberative` meta-primitive. This invariant is enforced at
   runtime in `CortexSession.plan` (raises `SchemaValidationError` on violation) and covered by
   a probe. (v0 probe: exercises the enforcement path with a synthetic deliberative plan. Full
   AST static check of all plan-build paths is deferred to v1.)
7. Any bundled procedure is OEM-provenance only and references only primitive contracts present in
   the same manifest.
8. Durable KB writes use the scope-gated knowledge channel; `episodic` records are not persisted in
   the KB.

## 10. Versioning Policy

- v0: in-process, Python dataclasses, MiniGrid is the only conformant substrate. Breaking changes
  allowed freely.
- v0.1: adds `OrpiProcedure`, optional `bundled_procedures`, and scoped knowledge traces without
  changing the primitive contract compatibility bridge.
- The Phase 15 port (second substrate) is the validation event. Every place v0.1 bends or breaks during
  the port is recorded as a spec issue.
- v1 freeze happens only after the second substrate is conformant. From v1: additive changes only;
  breaking changes require a major version.

## 11. Design And Implementation History

This section preserves why ORPI v0.1 has its current shape. It is not a roadmap.

### Why ORPI Has An Inbound And Outbound Half

The first design pressure was not only "how does cognition call a robot?" A retrofit cognition
layer also needs to explain and learn from deployment:

- what the operator intended;
- which claims were used;
- which capability contracts were considered;
- what authority was granted;
- what executed;
- whether effects were verified;
- which component owned a failure.

An inbound-only primitive interface would leave that audit and supervision story to ad hoc logs.
`LabelledEpisode` therefore belongs to the interface standard rather than being an optional
analytics layer.

### Why Contracts Are More Than Capability Names

Earlier readiness logic could treat a present/implemented primitive as executable even when its
frames, claims, authority, validation hooks, or safety policy were not satisfied. ORPI retains the
full contract because:

- availability is not feasibility;
- feasibility is not authority;
- execution is not verified success;
- a failure mode is a typed outcome, not merely a false postcondition.

The contract is intentionally object/effect-centric. "Door state changed from closed to open" is
portable; "joint trajectory 17 ran" is embodiment detail.

### Why The Legacy Taxonomy Bridge Remains

The implementation already grouped primitives as task, grounding, sensing, action, and claims.
Forcing every authoring site to switch immediately to `sense | actuation | meta` would have mixed
interface extraction with a repository-wide migration before a second substrate could validate
the taxonomy.

The accepted sequence was:

1. project the legacy layers into the ORPI taxonomy at the interface;
2. enforce contracts, manifests, cadence, traces, and conformance;
3. let the second substrate expose mismatches;
4. freeze and migrate authoring vocabulary only after the interface survives.

### Why Procedures Are Interface Objects But Not A New Hierarchy

Substrates and OEMs already possess vouched-for recipes. Treating them only as local cache entries
would hide valuable capability, while creating a new procedure system would duplicate
`ProcedureRecipe` and plan-cache machinery.

`OrpiProcedure` is therefore a serialized interface view. It can be selected by declared
postcondition, but expands to primitive handles before readiness and ticket issuance. Procedure
selection never bypasses primitive-level authority.

### Phase 12D Consolidation Discoveries

The consolidation pass separated two problems that had been conflated:

- **substrate coupling:** critical to a second-substrate proof;
- **station bloat:** operational debt, but not a blocker for steering or interface validity.

The leak audit classified sites on two axes:

| Axis | Meaning |
|---|---|
| cheap vs structural | whether removal is surgical or changes a real boundary |
| curriculum-touching vs not | whether Phase 13 would build new behavior on top of the leak |

Curriculum-touching leaks were pulled forward. Non-curriculum leaks were allowed to remain for
Phase 14. Structural station extraction stayed parked behind the decomposition design.

Two cheap leaks were removed during consolidation:

- MiniGrid grounding primitives moved out of the generic primitive library;
- ranked request-plan step ids became context/pluralization driven.

### Trace Completion And Remaining Verification Debt

Phase 12D closed the highest-value trace gaps:

- failure categories map to the ORPI attribution taxonomy while retaining the raw category;
- named `postcondition_primitive` checkers are invoked and recorded;
- degenerate boolean verification is labelled honestly.

Rich predicted-delta versus re-observed-state verification remains incomplete. It depends on
runtime outcome normalization and post-action evidence, and is intentionally coupled to the
mission-termination/action-outcome work rather than being simulated by trace formatting alone.

### Knowledge-Scope Discovery

`universal` scope exists but is rarely earned by current MiniGrid plans because they name concrete
primitive handles. Real universal transfer requires plans expressed in effect/postcondition
vocabulary across more than one embodiment. The second-substrate port is the proof event; a
synthetic MiniGrid fixture only proves the derivation mechanism exists.

### Conformance Philosophy

The v0.1 probes intentionally enforce the most important invariants now:

- manifest-at-init;
- contract coverage;
- valid cadence/mode combinations;
- no deliberative meta-primitive in compiled runtime plans;
- OEM-only bundled procedures;
- labelled episode emission;
- scope-gated knowledge writes.

More expensive whole-program static checks are deferred until v1, when a second substrate makes
their cost worthwhile and provides a real counterexample set.
