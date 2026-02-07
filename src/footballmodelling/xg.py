"""Expected goals (xG) models."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .domain import Position


@dataclass(frozen=True)
class Goal:
    """Goal location definition."""

    center: Position


class SimpleLogisticXG:
    """A simple logistic xG approximation used in the original research code.

    This function intentionally mirrors the scale and transformation from the notebook
    so that refactoring does not silently change outcomes.
    """

    def __init__(self, goal: Goal) -> None:
        self.goal = goal

    def __call__(self, shooter: Position) -> float:
        """Compute xG for a shot from `shooter` towards the goal center."""
        x1 = shooter.x / 1500.0
        x2 = self.goal.center.x / 1500.0
        y1 = shooter.y / 1000.0
        y2 = self.goal.center.y / 1000.0

        x = abs(x1 - x2) * 100.0
        y = abs(y2 - y1) * 100.0

        z = -12.1032 + 0.103273 * y - 0.0585 * abs(x)
        ez = math.exp(z)
        return ez / (1.0 + ez)
