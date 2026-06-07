from __future__ import annotations

from .schemas import OperationalContext


class MiniGridOperationalContext(OperationalContext):
    """MiniGrid situation frame for the current GoToDoor stress substrate."""

    @classmethod
    def default(
        cls,
        *,
        env_id: str = "MiniGrid-GoToDoor-8x8-v0",
    ) -> "MiniGridOperationalContext":
        return cls(
            context_id="minigrid.goto-door",
            substrate_id="minigrid",
            version="1",
            object_vocabulary=["door"],
            attribute_vocabulary=[
                "color",
                "position",
                "distance",
                "agent_pose",
                "visibility",
            ],
            task_families=[
                {
                    "task_type": "go_to_object",
                    "canonical_pattern": "go to the {color} {object_type}",
                    "object_types": ["door"],
                    "required_attributes": ["color", "object_type"],
                    "capability_handle": "task.go_to_object.door",
                }
            ],
            reference_semantics={
                "closest": {
                    "requires_metric": True,
                    "default_metric": "manhattan",
                    "reference": "agent",
                    "tie_policy": "clarify",
                },
                "farthest": {
                    "requires_metric": True,
                    "default_metric": "manhattan",
                    "reference": "agent",
                    "tie_policy": "display",
                },
                "same": {"source": "episodic.last_target"},
                "other": {"source": "active_claims.other_doors"},
                "delivery_target": {"source": "operator_claim.delivery_target"},
            },
            grounding_semantics={
                "visibility_model": "fully_observed_currently",
                "object_types": ["door"],
                "attribute_values": {
                    "color": ["red", "green", "blue", "yellow", "purple", "grey"],
                },
                "attribute_aliases": {
                    "color": {"gray": "grey"},
                },
                "distance_metrics": ["manhattan", "euclidean"],
                "distance_references": ["agent"],
                "rankable_relations": ["closest", "farthest"],
                "tie_policy": "clarify_or_display",
            },
            claim_rules={
                "grounding_scope": "session",
                "grounding_fingerprint": "agent_pose_and_step_count",
                "operator_claim_scope": "durable",
                "promotion_requires_operator_assertion": True,
            },
            display_rules={
                "target_label": "{color} {object_type}@({x},{y})",
                "grounding_answer_header": "GROUNDING ANSWER",
                "coordinate_frame": "minigrid_cell",
            },
            environment_identity_fields=[
                "env_id",
                "seed",
                "grid_width",
                "grid_height",
                "mission",
                "task_family",
            ],
            procedure_hints={
                "go_to_object": [
                    "locate_object",
                    "navigate_to_object",
                    "verify_adjacent",
                    "done",
                ]
            },
            metadata={
                "env_id": env_id,
                "description": "MiniGrid GoToDoor operational context.",
            },
        )
