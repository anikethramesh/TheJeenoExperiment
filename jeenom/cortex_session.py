from __future__ import annotations

from typing import Any

from .capability_registry import CapabilityRegistry
from .planning_semantics import PlanningSemantics
from .readiness_graph import evaluate_request_plan
from .request_planner import build_request_plan
from .schemas import (
    EnvironmentIdentity,
    OperatorIntent,
    ReadinessGraph,
    RequestPlan,
    StationActiveClaims,
)


class CortexSession:
    """Substrate-neutral planning kernel: intent → (RequestPlan, ReadinessGraph).

    Centralises all calls to build_request_plan and evaluate_request_plan so the
    station never invokes the planning pipeline directly.  Stateless between calls:
    the caller passes the current session state each time.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        planning_semantics: PlanningSemantics,
    ) -> None:
        self.registry = registry
        self.planning_semantics = planning_semantics

    def plan(
        self,
        utterance: str,
        intent: OperatorIntent,
        *,
        active_claims: StationActiveClaims | None = None,
        claims_valid: bool = False,
        environment_identity: EnvironmentIdentity | None = None,
    ) -> tuple[RequestPlan, ReadinessGraph]:
        """Build and evaluate a request plan for an intent."""
        active_summary = (
            active_claims.compact_summary()
            if active_claims is not None and claims_valid
            else None
        )
        plan = build_request_plan(
            utterance,
            intent,
            active_claims_summary=active_summary,
            environment_identity=environment_identity,
            planning_semantics=self.planning_semantics,
        )
        graph = evaluate_request_plan(
            plan,
            registry=self.registry,
            active_claims=active_claims,
            claims_valid=claims_valid,
            environment_identity=environment_identity,
        )
        return plan, graph

    def evaluate(
        self,
        plan: RequestPlan,
        *,
        active_claims: StationActiveClaims | None = None,
        claims_valid: bool = False,
        environment_identity: EnvironmentIdentity | None = None,
    ) -> ReadinessGraph:
        """Evaluate an already-built plan (e.g. after repair, cache reuse, or manual construction)."""
        return evaluate_request_plan(
            plan,
            registry=self.registry,
            active_claims=active_claims,
            claims_valid=claims_valid,
            environment_identity=environment_identity,
        )
