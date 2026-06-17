"""Phase 13A.2.1 — stable-fingerprint consolidation.

Red-bar first: this test imports `jeenom.fingerprint` (which must become the single
home for canonical serialization + hashing) and asserts (a) the helper reproduces the
exact digests the call sites used to hand-roll, and (b) no site still hand-rolls
`hashlib.sha256(` — the anti-drift guard. Fails while the four sites (plan_cache,
plan_reuse, mission_cortex, schemas) compute fingerprints inline.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from jeenom import fingerprint
from jeenom.schemas import EnvironmentIdentity

ROOT = Path(__file__).resolve().parents[1] / "jeenom"

# Files whose fingerprinting moved into jeenom/fingerprint.py.
HASHING_SITES = ["plan_reuse.py", "mission_cortex.py", "schemas.py"]
HELPER_SITES = ["plan_cache.py", "plan_reuse.py", "mission_cortex.py", "schemas.py"]


def test_canonical_json_matches_old_inline_expression():
    obj = {"b": 1, "a": 2, "nested": {"y": 1, "x": 2}}
    # sort_keys=True (the common case)
    assert fingerprint.canonical_json(obj) == json.dumps(
        obj, sort_keys=True, separators=(",", ":")
    )
    # sort_keys=False (plan_semantic_key keeps step order significant)
    ordered = {"objective_type": "query", "steps": [("a", "b"), ("c", "d")]}
    assert fingerprint.canonical_json(ordered, sort_keys=False) == json.dumps(
        ordered, separators=(",", ":")
    )
    # default=str (schemas fingerprints coerce non-JSON values)
    weird = {"k": object()}
    assert fingerprint.canonical_json(weird, default=str) == json.dumps(
        weird, sort_keys=True, separators=(",", ":"), default=str
    )


def test_stable_hash_and_fingerprint_match_old_expressions():
    assert fingerprint.stable_hash("go to door|grounding.x") == hashlib.sha256(
        "go to door|grounding.x".encode("utf-8")
    ).hexdigest()
    # truncation (mission id uses [:12], plan_semantic_key uses [:16])
    full = hashlib.sha256("abc".encode("utf-8")).hexdigest()
    assert fingerprint.stable_hash("abc", length=12) == full[:12]
    # fingerprint = hash of canonical json
    payload = {"objective_type": "query", "steps": [("a", "b")]}
    expected = hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    assert fingerprint.fingerprint(payload, sort_keys=False, length=16) == expected


def test_environment_identity_fingerprint_is_deterministic_and_routed():
    ident = EnvironmentIdentity(env_id="MiniGrid-GoToDoor-8x8-v0", seed=42, grid_width=8)
    # behavior preserved: equals the canonical-json-then-sha256 of the stable view
    assert ident.fingerprint() == ident.fingerprint()
    assert len(ident.fingerprint()) == 64  # full sha256 hex


def test_no_site_hand_rolls_sha256():
    """Anti-drift: the hashing sites route through fingerprint, not hashlib directly."""
    offenders = []
    for name in HASHING_SITES:
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sha256"
            ):
                offenders.append(f"{name}:{node.lineno}")
    assert not offenders, f"hand-rolled hashlib.sha256 remains: {offenders}"


def test_all_sites_reference_the_fingerprint_helper():
    for name in HELPER_SITES:
        src = (ROOT / name).read_text(encoding="utf-8")
        assert "fingerprint" in src, f"{name} does not route through the fingerprint helper"
