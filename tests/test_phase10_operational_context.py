from __future__ import annotations

import unittest

from jeenom.minigrid_operational_context import MiniGridOperationalContext
from jeenom.operator_station import OperatorStationSession
from jeenom.schemas import OperationalContext, SchemaValidationError


class TestPhase10OperationalContext(unittest.TestCase):
    def test_context_fingerprint_is_stable_and_content_sensitive(self):
        ctx = OperationalContext(
            context_id="unit.context",
            substrate_id="unit",
            version="1",
            object_vocabulary=["door"],
            attribute_vocabulary=["color"],
            task_families=[{"task_type": "go_to_object"}],
            reference_semantics={"closest": {"requires_metric": True}},
            grounding_semantics={"distance_metrics": ["manhattan"]},
        )
        same = OperationalContext.from_dict(ctx.as_dict())
        changed = OperationalContext(
            context_id="unit.context",
            substrate_id="unit",
            version="2",
            object_vocabulary=["door"],
            attribute_vocabulary=["color"],
            task_families=[{"task_type": "go_to_object"}],
            reference_semantics={"closest": {"requires_metric": True}},
            grounding_semantics={"distance_metrics": ["manhattan"]},
        )

        self.assertEqual(ctx.fingerprint(), same.fingerprint())
        self.assertNotEqual(ctx.fingerprint(), changed.fingerprint())

    def test_context_rejects_missing_identity(self):
        with self.assertRaises(SchemaValidationError):
            OperationalContext(
                context_id="",
                substrate_id="unit",
                version="1",
            )

    def test_minigrid_context_exposes_compact_slice(self):
        ctx = MiniGridOperationalContext.default(env_id="MiniGrid-GoToDoor-8x8-v0")
        compact = ctx.compact_slice("go to the closest red door")

        self.assertIsInstance(ctx, OperationalContext)
        self.assertIn("door", ctx.object_vocabulary)
        self.assertIn("color", ctx.attribute_vocabulary)
        self.assertIn("manhattan", compact["grounding_semantics"]["distance_metrics"])
        self.assertLess(len(repr(compact)), len(repr(ctx.as_dict())))
        self.assertNotIn("display_rules", compact)
        self.assertNotIn("procedure_hints", compact)

    def test_operator_station_has_operational_context(self):
        session = OperatorStationSession(
            compiler_name="smoke_test",
            render_mode="none",
        )

        self.assertIsInstance(session.operational_context, MiniGridOperationalContext)
        self.assertEqual(
            session.context_fingerprint,
            session.operational_context.fingerprint(),
        )


if __name__ == "__main__":
    unittest.main()
