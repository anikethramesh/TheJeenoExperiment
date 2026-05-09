from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from .schemas import (
    EvidenceFrame,
    ExecutionContext,
    ExecutionContract,
    PlanCacheEntry,
    PlanCacheStats,
    ProcedureRecipe,
    SensePlanTemplate,
    SkillPlanTemplate,
    TaskRequest,
)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return value


def normalize_semantic_params(params: dict[str, Any] | None) -> str:
    params = params or {}
    payload = {
        "color": params.get("color"),
        "object_type": params.get("object_type"),
        "target_location_required": params.get("target_location") is not None,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def procedure_key(task_request: TaskRequest) -> str:
    payload = {
        "task_type": task_request.task_type,
        "params": normalize_semantic_params(task_request.params),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def sense_key(
    evidence_frame: EvidenceFrame,
    execution_context: ExecutionContext,
    resolved_task_params: dict[str, Any] | None,
) -> str:
    merged_params = dict(resolved_task_params or {})
    merged_params.update(evidence_frame.context)
    payload = {
        "evidence_needs": list(evidence_frame.needs),
        "active_skill": execution_context.active_skill,
        "params": normalize_semantic_params(merged_params),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def skill_key(execution_contract: ExecutionContract) -> str:
    payload = {
        "skill": execution_contract.skill,
        "params": normalize_semantic_params(execution_contract.params),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class PlanCache:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.entries: dict[str, PlanCacheEntry] = {}
        self.stats = PlanCacheStats()
        self.failure_counts: dict[str, int] = {}

    def lookup(self, key: str) -> PlanCacheEntry | None:
        if not self.enabled:
            return None

        entry = self.entries.get(key)
        if entry is None:
            self.stats.misses += 1
            return None

        entry.hit_count += 1
        self.stats.hits += 1
        if entry.compiler_backend == "llm_compiler":
            self.stats.llm_calls_saved += 1
        return entry

    def store(
        self,
        key: str,
        template_type: str,
        template: ProcedureRecipe | SensePlanTemplate | SkillPlanTemplate,
        created_at_loop: int,
    ) -> PlanCacheEntry:
        entry = PlanCacheEntry(
            key=key,
            template_type=template_type,
            template=template,
            compiler_backend=template.compiler_backend,
            source=template.source,
            created_at_loop=created_at_loop,
            hit_count=0,
        )
        self.entries[key] = entry
        self.failure_counts.pop(key, None)
        return entry

    def invalidate(self, key: str) -> None:
        self.entries.pop(key, None)
        self.failure_counts.pop(key, None)

    def record_failure(self, key: str, immediate: bool = False, threshold: int = 2) -> bool:
        if not self.enabled or key not in self.entries:
            return False
        if immediate:
            self.invalidate(key)
            return True

        count = self.failure_counts.get(key, 0) + 1
        self.failure_counts[key] = count
        if count >= threshold:
            self.invalidate(key)
            return True
        return False

    def clear_failures(self, key: str) -> None:
        self.failure_counts.pop(key, None)

    def summary(self, include_entries: bool = True) -> dict[str, Any]:
        summary = {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "llm_calls_saved": self.stats.llm_calls_saved,
            "entries": [],
        }
        if include_entries:
            summary["entries"] = [
                {
                    "key": entry.key,
                    "template_type": entry.template_type,
                    "template": _json_safe(entry.template),
                    "compiler_backend": entry.compiler_backend,
                    "source": entry.source,
                    "created_at_loop": entry.created_at_loop,
                    "hit_count": entry.hit_count,
                }
                for entry in sorted(self.entries.values(), key=lambda item: (item.template_type, item.key))
            ]
        return summary
