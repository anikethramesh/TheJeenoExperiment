from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capability_registry import CapabilityRegistry
from .schemas import OperationalContext, SchemaValidationError
from .substrate_adapter import SubstrateAdapter


@dataclass
class RuntimePackage:
    """Runtime bundle that adapts JEENOM to a concrete substrate and situation."""

    substrate: SubstrateAdapter
    operational_context: OperationalContext
    domain_helper: Any
    capability_registry: CapabilityRegistry | None = None

    def __post_init__(self) -> None:
        if self.substrate is None:
            raise SchemaValidationError("RuntimePackage requires substrate")
        if not isinstance(self.operational_context, OperationalContext):
            raise SchemaValidationError("RuntimePackage requires OperationalContext")
        if self.domain_helper is None:
            raise SchemaValidationError("RuntimePackage requires domain_helper")
        helper_context = getattr(self.domain_helper, "operational_context", None)
        if helper_context is not self.operational_context:
            raise SchemaValidationError(
                "RuntimePackage domain_helper must be bound to operational_context"
            )
        if (
            self.capability_registry is not None
            and not isinstance(self.capability_registry, CapabilityRegistry)
        ):
            raise SchemaValidationError(
                "RuntimePackage capability_registry must be a CapabilityRegistry"
            )

    def resolve_capability_registry(self) -> CapabilityRegistry:
        if self.capability_registry is not None:
            return self.capability_registry
        registry = self.substrate.capability_registry()
        if not isinstance(registry, CapabilityRegistry):
            raise SchemaValidationError(
                "RuntimePackage substrate returned non-CapabilityRegistry"
            )
        self.capability_registry = registry
        return registry
