"""Stable, content-addressed fingerprinting — the single home for canonical
serialization and hashing.

Two primitives were hand-rolled identically across `plan_cache`, `plan_reuse`,
`mission_cortex`, and `schemas`: compact deterministic JSON, and a SHA-256 of a
payload. Centralized here so the digest contract (cache keys, reuse keys, env
identity, context fingerprints) cannot silently drift between sites.

Pure stdlib; imports nothing from jeenom, so any module (incl. ``schemas``) may use
it without a cycle.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable


def canonical_json(
    obj: Any,
    *,
    sort_keys: bool = True,
    default: Callable[[Any], Any] | None = None,
) -> str:
    """Compact, deterministic JSON. ``sort_keys=True`` for order-insensitive payloads;
    pass ``sort_keys=False`` when list/insertion order is itself significant (e.g. an
    ordered step signature). ``default`` coerces non-JSON values (e.g. ``str``)."""
    return json.dumps(obj, sort_keys=sort_keys, separators=(",", ":"), default=default)


def stable_hash(text: str, *, length: int | None = None) -> str:
    """SHA-256 hex digest of ``text`` (utf-8). ``length`` truncates the hex digest."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length] if length is not None else digest


def fingerprint(
    obj: Any,
    *,
    sort_keys: bool = True,
    default: Callable[[Any], Any] | None = None,
    length: int | None = None,
) -> str:
    """SHA-256 of an object's canonical JSON — the hash-of-payload pattern."""
    return stable_hash(
        canonical_json(obj, sort_keys=sort_keys, default=default), length=length
    )
