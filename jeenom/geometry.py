"""Substrate-agnostic geometry — the single home for coordinate metrics.

MiniGrid is an integral 2D grid; a robotics substrate (e.g. AI2-THOR) is a
float 3D world. Both feed the same cognition layer, so coordinates are kept
numeric (``int | float``) and the distance metrics are N-dimensional: a 2D and a
3D point simply carry 2 or 3 components. Adding an axis is one edit here, not a
sweep across every call site that hand-rolled ``obj.x - agent_x``.

Pure: depends only on the stdlib. No jeenom/substrate imports, so any module
(including ``schemas``) may import it without a cycle.
"""
from __future__ import annotations

import math
from typing import Sequence

Number = int | float
Point = Sequence[Number]


def as_coord(value: Number) -> Number:
    """Coerce a raw coordinate: ``int`` when integral, ``float`` otherwise.

    Keeps MiniGrid grid coords rendering and comparing as ``3`` while preserving
    a genuine ``3.5`` from a continuous substrate. Also normalizes numpy scalars.
    """
    f = float(value)
    return int(f) if f.is_integer() else f


def manhattan(a: Point, b: Point) -> Number:
    """L1 distance. ``zip`` pairs only shared axes, so mixed 2D/3D never throws;
    same-dimension points (the contract) sum every axis."""
    return sum(abs(p - q) for p, q in zip(a, b))


def euclidean(a: Point, b: Point) -> float:
    """L2 distance, always a float."""
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b)))
