"""Named Concept Knowledge Base — Phase 8.35.

Stores operator-defined named concepts: short operator labels that expand to
full utterances and optionally carry pre-compiled RequestPlans.

A concept named 'bingo' can expand to 'go to the red door'.  When the
operator later says 'bingo', the station resolves the concept, retrieves its
pre-compiled plan from PlanReuseCache (Phase 8.3), and executes without a
recompile cycle.

Persistence: concepts are serialised to knowledge_base.json inside the
session memory_root and survive station restarts.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schemas import RequestPlan


KNOWLEDGE_SCOPES = ("episodic", "site", "embodiment", "universal")


@dataclass
class NamedConcept:
    """An operator-asserted claim: a durable, named shorthand for a full instruction.

    claim_scope="operator": asserted by the operator, durable across restarts,
    invalidated only by explicit retraction (forget / clear memory).

    concept_type="atomic": a single instruction expansion.
    concept_type="procedure": an ordered sequence of atomic concept names.
    """

    name: str
    utterance: str
    plan: RequestPlan | None = None
    stored_at: float = field(default_factory=time.time)
    recall_count: int = 0
    tags: list[str] = field(default_factory=list)
    claim_scope: str = "operator"
    concept_type: str = "atomic"          # "atomic" | "procedure"
    steps: list[str] = field(default_factory=list)  # procedure: ordered concept names
    scope: str = "site"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "utterance": self.utterance,
            "plan": self.plan.as_dict() if self.plan is not None else None,
            "stored_at": self.stored_at,
            "recall_count": self.recall_count,
            "tags": list(self.tags),
            "claim_scope": self.claim_scope,
            "concept_type": self.concept_type,
            "steps": list(self.steps),
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NamedConcept:
        from .schemas import RequestPlan

        plan: RequestPlan | None = None
        if data.get("plan") is not None:
            try:
                plan = RequestPlan.from_dict(data["plan"])
            except Exception:
                pass
        return cls(
            name=data["name"],
            utterance=data["utterance"],
            plan=plan,
            stored_at=data.get("stored_at", time.time()),
            recall_count=data.get("recall_count", 0),
            tags=list(data.get("tags", [])),
            claim_scope=data.get("claim_scope", "operator"),
            concept_type=data.get("concept_type", "atomic"),
            steps=list(data.get("steps", [])),
            scope=data.get("scope", "site"),
        )


def _manifest_contracts(manifest: Any) -> dict[str, dict[str, Any]]:
    if manifest is None:
        return {}
    primitives = getattr(manifest, "primitives", None)
    if primitives is None and isinstance(manifest, dict):
        primitives = manifest.get("primitives", [])
    contracts: dict[str, dict[str, Any]] = {}
    for primitive in primitives or []:
        payload = primitive.as_dict() if hasattr(primitive, "as_dict") else dict(primitive)
        name = payload.get("name")
        if isinstance(name, str):
            contracts[name] = payload
    return contracts


def derive_scope(record: Any, manifest: Any = None) -> str:
    """Derive transfer scope from a concept/procedure and an ORPI manifest."""

    provenance = getattr(record, "provenance", None)
    if provenance == "oem":
        return "embodiment"

    contracts = _manifest_contracts(manifest)
    effect_vocabulary: set[str] = set()
    for payload in contracts.values():
        effect_vocabulary.update(str(item) for item in payload.get("postconditions", []))
        effect_vocabulary.update(str(item) for item in payload.get("outputs", []))

    primitive_step_names = getattr(record, "primitive_step_names", None)
    if callable(primitive_step_names):
        names = primitive_step_names()
        if any(contracts.get(name, {}).get("substrate_fingerprint") for name in names):
            return "embodiment"
        return "site"

    plan = getattr(record, "plan", None)
    steps = list(getattr(plan, "steps", []) or [])
    direct_handles = [
        getattr(step, "required_handle", None)
        for step in steps
        if getattr(step, "required_handle", None)
    ]
    if direct_handles:
        if any(contracts.get(handle, {}).get("substrate_fingerprint") for handle in direct_handles):
            return "embodiment"
        if all(handle in effect_vocabulary for handle in direct_handles):
            return "universal"
        return "site"
    return "site"


class KnowledgeBase:
    """Session-scoped store of named operator concepts.

    Concepts survive station restarts via JSON persistence in memory_root.
    Teaching a concept immediately persists it.  Plans are stored alongside
    concepts and seeded into PlanReuseCache by the station at teach-time so
    future recalls hit the cache rather than recompiling.
    """

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Canonical concept key: lowercase, strip leading/trailing non-word characters."""
        return re.sub(r"^[^\w]+|[^\w]+$", "", name.strip().lower())

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self._concepts: dict[str, NamedConcept] = {}
        if storage_path is not None and storage_path.exists():
            self.load()

    # ── write ─────────────────────────────────────────────────────────────────

    def _is_sequence(self, utterance: str) -> list[str] | None:
        """Return ordered concept-name list if utterance is a comma-separated sequence of known concepts."""
        parts = [self._normalize_name(p) for p in utterance.split(",")]
        if len(parts) < 2:
            return None
        if all(parts) and all(self._concepts.get(p) is not None for p in parts):
            return parts
        return None

    def teach(
        self,
        name: str,
        utterance: str,
        plan: RequestPlan | None = None,
        *,
        scope: str | None = "site",
        manifest: Any = None,
    ) -> NamedConcept:
        """Store a named concept; updates utterance/plan if the name already exists.

        If utterance is a comma-separated list of known concept names (>=2), the
        concept is stored as concept_type="procedure" with the resolved steps list.
        """
        key = self._normalize_name(name)
        seq = self._is_sequence(utterance)
        concept_type = "procedure" if seq is not None else "atomic"
        steps = seq if seq is not None else []
        if scope is not None and scope not in KNOWLEDGE_SCOPES:
            raise ValueError(f"Unknown knowledge scope: {scope}")
        if scope == "episodic":
            raise ValueError("KnowledgeBase does not store episodic records")

        existing = self._concepts.get(key)
        if existing is not None:
            existing.utterance = utterance
            existing.concept_type = concept_type
            existing.steps = steps
            if plan is not None:
                existing.plan = plan
            existing.scope = scope or derive_scope(existing, manifest)
            self.persist()
            return existing
        concept = NamedConcept(
            name=key,
            utterance=utterance,
            plan=plan,
            concept_type=concept_type,
            steps=steps,
            scope=scope or "site",
        )
        concept.scope = scope or derive_scope(concept, manifest)
        self._concepts[key] = concept
        self.persist()
        return concept

    def forget(self, name: str) -> bool:
        """Remove a concept by name. Returns True if it existed."""
        key = self._normalize_name(name)
        if key in self._concepts:
            del self._concepts[key]
            self.persist()
            return True
        return False

    # ── read ──────────────────────────────────────────────────────────────────

    def recall(self, name: str) -> NamedConcept | None:
        """Look up a concept by exact name (case-insensitive). Increments recall_count."""
        key = self._normalize_name(name)
        concept = self._concepts.get(key)
        if concept is not None:
            concept.recall_count += 1
            self.persist()
        return concept

    def search(self, query: str) -> list[NamedConcept]:
        """Return all concepts whose name or utterance contains query as a substring."""
        q = self._normalize_name(query)
        return [c for k, c in self._concepts.items() if q in k or q in c.utterance.lower()]

    def all_concepts(self) -> list[NamedConcept]:
        return list(self._concepts.values())

    def purge_scope(self, scope: str) -> int:
        if scope not in KNOWLEDGE_SCOPES:
            raise ValueError(f"Unknown knowledge scope: {scope}")
        removed = [name for name, concept in self._concepts.items() if concept.scope == scope]
        for name in removed:
            del self._concepts[name]
        if removed:
            self.persist()
        return len(removed)

    # ── persistence ───────────────────────────────────────────────────────────

    def persist(self) -> None:
        if self.storage_path is None:
            return
        data = [c.as_dict() for c in self._concepts.values()]
        self.storage_path.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        if self.storage_path is None or not self.storage_path.exists():
            return
        try:
            raw = json.loads(self.storage_path.read_text())
            for item in raw:
                concept = NamedConcept.from_dict(item)
                concept.name = self._normalize_name(concept.name)
                self._concepts[concept.name] = concept
        except Exception:
            self._concepts = {}

    # ── compact summary ───────────────────────────────────────────────────────

    def compact_summary(self) -> str | None:
        if not self._concepts:
            return None
        lines = [f"  {c.name!r} → {c.utterance!r}" for c in self._concepts.values()]
        return "Known concepts:\n" + "\n".join(lines)
