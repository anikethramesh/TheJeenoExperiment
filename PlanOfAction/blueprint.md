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