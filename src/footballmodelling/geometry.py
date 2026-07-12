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


def distance_point_to_segment(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Compute the distance from a point to a finite line segment."""
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / length_sq
    t = min(max(t, 0.0), 1.0)
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(px - closest_x, py - closest_y)
