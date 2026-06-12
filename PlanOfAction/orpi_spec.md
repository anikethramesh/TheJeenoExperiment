# ORPI v0 — Open Robotics Primitive Interface

**Status: v0, explicitly unstable.** Freeze to v1 only after the interface survives the
Phase 15 cross-substrate port (the validation event). Standards extracted from n=1 substrates
ossify the wrong abstractions; v0 exists to be broken by the second substrate, deliberately.

This document is the authoritative contract/manifest/trace reference for ORPI. It is built and
versioned alongside the code. The implementation roadmap lives in
[task_plan.md](task_plan.md) Phase 12.

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

These three classes are the value set of `schemas.PrimitiveSpec.primitive_type` under ORPI v0. The
prior value set (`task | grounding | sensing | action | claims`) is remapped — see Phase 12 in
[task_plan.md](task_plan.md) for the migration.

Implementation note for the completed MiniGrid Phase 12 boundary: the code exposes the ORPI
taxonomy through `OrpiContract` while preserving the legacy `PrimitiveSpec.primitive_type` values
for existing registry grouping and prompt summaries. This compatibility bridge is intentional; the
hard schema remap can happen after the contract/manifest/trace boundary survives more substrate
pressure.

## 4. The Contract

The contract is the existing `schemas.PrimitiveSpec`, serialized. Fields marked **NEW** are v0
additions; everything else already exists in the repo.

| Field | Meaning | Notes |
|---|---|---|
| `name`, `primitive_type`, `layer`, `description` | identity | `primitive_type` ∈ {sense, actuation, meta}; `layer` ∈ {sense, cortex, spine} |
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

Registration is **fail-closed in spirit**: validation is permissive only before any manifest is
registered, and a conformance probe must assert each adapter registers at init, so the permissive
window is provably never live.

## 6. The Trace (Outbound Half)

Every turn emits a `LabelledEpisode`. It is a thin aggregator/serializer over existing trace
machinery (`OperatorIntent`, `RequestPlan`, `ReadinessGraph`, `CommandResult`, `FailureOutcome`,
`CorticalEnvelope`) — not a parallel type tree.

```
intent          — OperatorIntent as compiled (+ verifier verdict)
grounding       — claims consumed, with provenance/confidence/freshness at read time
plan            — RequestPlan + ReadinessGraph verdicts per step
authority       — tickets issued, scope, issuer
execution       — per-step CommandResult, incl. typed FailureOutcome
verification    — postcondition_primitive results vs. predicted postconditions
attribution     — on failure: which contract was violated
                  (stale_claim | miscompiled_intent | unmet_postcondition |
                   missing_authority | substrate_fault)
steering        — any operator interventions in this episode (Phase 13 ask-for-help)
```

The trace is the product's audit story ("what did the operator tell it, when, and did the robot
honour it") and the learning story (component-level credit assignment for deployment loops) in one
artifact. Failed episodes are **kept**, with attribution — failures are future skills if properly
labelled.

## 7. Conformance

A substrate is ORPI-v0 conformant when:
1. Every capability is registered through a contract; no side-channel capabilities.
2. The manifest is registered at init (probe-enforced).
3. All postconditions are object-centric deltas; all actuation primitives with `safety_class ≠ query`
   name a `postcondition_primitive`.
4. Cadence declarations are honoured (static probe: no deliberation-cadence call inside Spine paths).
5. Every executed turn emits a `LabelledEpisode`.
6. No compiled plan references a `deliberative` meta-primitive (static probe).

## 8. Versioning Policy

- v0: in-process, Python dataclasses, MiniGrid is the only conformant substrate. Breaking changes
  allowed freely.
- The Phase 15 port (second substrate) is the validation event. Every place v0 bends or breaks during
  the port is recorded as a spec issue.
- v1 freeze happens only after the second substrate is conformant. From v1: additive changes only;
  breaking changes require a major version.
