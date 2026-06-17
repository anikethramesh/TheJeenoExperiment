"""Conservative RequestPlan reuse — Phase 8.3.

A PlanReuseCache stores compiled RequestPlans keyed by their structural
fingerprint. Before compiling a fresh plan the station checks whether a
structurally identical plan already exists and whether its environment
assumptions still hold. The reuse policy is always `if_valid`: the stored
plan is accepted only when the ReadinessGraph returns `executable` against
the current environment identity.

Nothing here touches grounding, execution, or sensing. Reuse affects only
which RequestPlan is presented as `last_request_plan`; the execution path
is unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import fingerprint as _fp

if TYPE_CHECKING:
    from .capability_registry import CapabilityRegistry
    from .schemas import EnvironmentIdentity, RequestPlan


def _step_constraint_fingerprint(constraints: dict[str, Any]) -> str:
    """Stable string over the semantically meaningful constraint fields."""
    relevant = {
        k: v
        for k in ("color", "exclude_colors", "metric", "order", "ordinal", "threshold", "comparison")
        if (v := constraints.get(k)) is not None
    }
    return _fp.canonical_json(relevant)


def plan_semantic_key(plan: RequestPlan) -> str:
    """Stable structural key for a RequestPlan.

    Two plans with identical step handles, operations, layers, and semantic
    constraints produce the same key regardless of utterance phrasing or
    the random request_id assigned at compile time.
    """
    steps_sig = [
        (
            step.layer,
            step.operation,
            step.required_handle or "",
            _step_constraint_fingerprint(step.constraints),
        )
        for step in plan.steps
    ]
    return _fp.fingerprint(
        {"objective_type": plan.objective_type, "steps": steps_sig},
        sort_keys=False,
        length=16,
    )


@dataclass
class ReuseVerdict:
    verdict: str  # "reuse" | "recompile" | "refresh_claims"
    reason: str
    blocking_assumption_ids: list[str] = field(default_factory=list)


@dataclass
class PlanReuseEntry:
    key: str
    plan: RequestPlan
    stored_at: float = field(default_factory=time.time)
    reuse_count: int = 0
    last_outcome: str | None = None  # "success" | "failure"


@dataclass
class ReuseHistoryRecord:
    key: str
    verdict: str
    reason: str
    env_id: str | None = None
    seed: int | None = None


class PlanReuseCache:
    """Session-scoped cache of structural RequestPlan entries.

    Can be shared across sessions by passing the same instance to multiple
    OperatorStationSession constructors, enabling cross-session plan reuse.
    """

    def __init__(self) -> None:
        self.entries: dict[str, PlanReuseEntry] = {}
        self.history: list[ReuseHistoryRecord] = []

    # ── storage ───────────────────────────────────────────────────────────────

    def store(self, plan: RequestPlan) -> PlanReuseEntry:
        """Store plan under its semantic key; idempotent if already present."""
        key = plan_semantic_key(plan)
        if key not in self.entries:
            self.entries[key] = PlanReuseEntry(key=key, plan=plan)
        return self.entries[key]

    def lookup(self, plan: RequestPlan) -> PlanReuseEntry | None:
        """Return the cached entry whose key matches plan, or None."""
        return self.entries.get(plan_semantic_key(plan))

    # ── reuse decision ────────────────────────────────────────────────────────

    def can_reuse(
        self,
        entry: PlanReuseEntry,
        registry: CapabilityRegistry,
        environment_identity: EnvironmentIdentity | None,
    ) -> ReuseVerdict:
        """Evaluate whether the stored plan is valid in the current context.

        Runs the ReadinessGraph over the stored plan. Verdict:
          reuse           — all required assumptions hold, all primitives available
          recompile       — a required environment assumption failed, or identity unknown
          refresh_claims  — plan is structurally valid but claims are stale

        If environment_identity is None we cannot verify any assumption, so we
        always recompile — it is safer than silently reusing an unverified plan.
        """
        if environment_identity is None:
            return ReuseVerdict(
                verdict="recompile",
                reason="Cannot verify environment assumptions: environment identity not yet established.",
            )

        from .readiness_graph import evaluate_request_plan

        graph = evaluate_request_plan(
            entry.plan,
            registry=registry,
            environment_identity=environment_identity,
        )
        if graph.graph_status == "executable":
            return ReuseVerdict(
                verdict="reuse",
                reason="All assumptions hold and all primitives are available.",
            )
        if graph.graph_status == "environment_assumption_failed":
            return ReuseVerdict(
                verdict="recompile",
                reason=f"Required environment assumptions failed: {graph.violated_assumption_ids}",
                blocking_assumption_ids=list(graph.violated_assumption_ids),
            )
        if graph.graph_status == "stale_claims":
            return ReuseVerdict(
                verdict="refresh_claims",
                reason="Claims are stale; re-ground before execution.",
            )
        return ReuseVerdict(
            verdict="recompile",
            reason=f"Plan not directly reusable: graph_status={graph.graph_status}",
        )

    # ── history ───────────────────────────────────────────────────────────────

    def record_reuse(
        self,
        key: str,
        verdict: str,
        reason: str,
        *,
        env_id: str | None = None,
        seed: int | None = None,
    ) -> None:
        entry = self.entries.get(key)
        if entry is not None and verdict == "reuse":
            entry.reuse_count += 1
        self.history.append(
            ReuseHistoryRecord(key=key, verdict=verdict, reason=reason, env_id=env_id, seed=seed)
        )

    def record_outcome(self, key: str, outcome: str) -> None:
        entry = self.entries.get(key)
        if entry is not None:
            entry.last_outcome = outcome
