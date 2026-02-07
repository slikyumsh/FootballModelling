"""Geometric helpers used by passing and interception logic."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Line:
    """Line in form y = m*x + b."""

    m: float
    b: float


def line_through_points(x1: float, y1: float, x2: float, y2: float) -> Line:
    """Build a line through two points."""
    dx = x2 - x1
    if abs(dx) < 1e-9:
        m = (y2 - y1) / 1e-9
    else:
        m = (y2 - y1) / dx
    b = y1 - m * x1
    return Line(m=m, b=b)


def distance_point_to_line(x: float, y: float, line: Line) -> float:
    """Compute perpendicular distance from a point to a line."""
    return abs(y - line.m * x - line.b) / math.sqrt(line.m**2 + 1.0)
