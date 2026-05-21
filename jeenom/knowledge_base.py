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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schemas import RequestPlan


@dataclass
class NamedConcept:
    """An operator-asserted claim: a durable, named shorthand for a full instruction.

    claim_scope="operator": asserted by the operator, durable across restarts,
    invalidated only by explicit retraction (forget / clear memory).
    """

    name: str
    utterance: str
    plan: RequestPlan | None = None
    stored_at: float = field(default_factory=time.time)
    recall_count: int = 0
    tags: list[str] = field(default_factory=list)
    claim_scope: str = "operator"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "utterance": self.utterance,
            "plan": self.plan.as_dict() if self.plan is not None else None,
            "stored_at": self.stored_at,
            "recall_count": self.recall_count,
            "tags": list(self.tags),
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
        )


class KnowledgeBase:
    """Session-scoped store of named operator concepts.

    Concepts survive station restarts via JSON persistence in memory_root.
    Teaching a concept immediately persists it.  Plans are stored alongside
    concepts and seeded into PlanReuseCache by the station at teach-time so
    future recalls hit the cache rather than recompiling.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path
        self._concepts: dict[str, NamedConcept] = {}
        if storage_path is not None and storage_path.exists():
            self.load()

    # ── write ─────────────────────────────────────────────────────────────────

    def teach(
        self,
        name: str,
        utterance: str,
        plan: RequestPlan | None = None,
    ) -> NamedConcept:
        """Store a named concept; updates utterance/plan if the name already exists."""
        key = name.strip().lower()
        existing = self._concepts.get(key)
        if existing is not None:
            existing.utterance = utterance
            if plan is not None:
                existing.plan = plan
            self.persist()
            return existing
        concept = NamedConcept(name=name.strip(), utterance=utterance, plan=plan)
        self._concepts[key] = concept
        self.persist()
        return concept

    def forget(self, name: str) -> bool:
        """Remove a concept by name. Returns True if it existed."""
        key = name.strip().lower()
        if key in self._concepts:
            del self._concepts[key]
            self.persist()
            return True
        return False

    # ── read ──────────────────────────────────────────────────────────────────

    def recall(self, name: str) -> NamedConcept | None:
        """Look up a concept by exact name (case-insensitive). Increments recall_count."""
        key = name.strip().lower()
        concept = self._concepts.get(key)
        if concept is not None:
            concept.recall_count += 1
            self.persist()
        return concept

    def search(self, query: str) -> list[NamedConcept]:
        """Return all concepts whose name or utterance contains query as a substring."""
        q = query.strip().lower()
        return [c for k, c in self._concepts.items() if q in k or q in c.utterance.lower()]

    def all_concepts(self) -> list[NamedConcept]:
        return list(self._concepts.values())

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
                self._concepts[concept.name.lower()] = concept
        except Exception:
            self._concepts = {}

    # ── compact summary ───────────────────────────────────────────────────────

    def compact_summary(self) -> str | None:
        if not self._concepts:
            return None
        lines = [f"  {c.name!r} → {c.utterance!r}" for c in self._concepts.values()]
        return "Known concepts:\n" + "\n".join(lines)
