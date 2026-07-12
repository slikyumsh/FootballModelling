"""Defensive behaviours and constraints (interceptions, offside, pressing)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .domain import FIELD_HEIGHT, FIELD_WIDTH, Pitch, PlayerOnPitch
from .geometry import distance_point_to_segment


@dataclass(frozen=True)
class DefenseParams:
    """Defense parameters controlling interceptions and movement constraints."""

    interception_radius: float
    min_horizontal_gap: float = 50.0
    compression_threshold: float | None = None
    compression_factor: float = 1.0


def is_offside(
    receiving: PlayerOnPitch,
    pitch: Pitch,
    ball_holder: PlayerOnPitch,
    defending_team: int,
) -> bool:
    """Check a simplified offside rule for an attacker.

    The simplified rule matches the original research prototype: the receiver is
    offside if they are ahead of the second-last defender and ahead of the ball.

    Args:
        receiving: Potential receiver.
        pitch: Current pitch state.
        ball_holder: Current ball holder.
        defending_team: Team id of defending side.

    Returns:
        True if receiver is offside, otherwise False.
    """
    if receiving.player.team_id == defending_team:
        return False

    defenders = pitch.defenders(defending_team)
    if not defenders:
        return False

    x_positions = sorted(p.position.x for p in defenders)
    second_last_x = x_positions[-1] if len(x_positions) == 1 else x_positions[-2]

    return receiving.position.x > second_last_x and receiving.position.x > ball_holder.position.x


def pass_is_intercepted(
    passer: PlayerOnPitch,
    receiver: PlayerOnPitch,
    pitch: Pitch,
    defending_team: int,
    params: DefenseParams,
) -> bool:
    """Return True if the pass segment is within interception radius of a defender."""
    for defender in pitch.defenders(defending_team):
        d = distance_point_to_segment(
            defender.position.x,
            defender.position.y,
            passer.position.x,
            passer.position.y,
            receiver.position.x,
            receiver.position.y,
        )
        if d < params.interception_radius:
            return True
    return False


def pressing_step(
    pitch: Pitch,
    attacking_team: int,
    defending_team: int,
    ball_holder: PlayerOnPitch,
    params: DefenseParams,
) -> None:
    """Move defenders towards nearest non-offside attackers.

    This keeps the behavior of the original `defense_pressing()` idea but with clearer
    naming and explicit constraints.
    """
    defenders = pitch.defenders(defending_team)
    attackers = [
        p
        for p in pitch.attackers(attacking_team)
        if not is_offside(p, pitch, ball_holder, defending_team)
    ]

    if not defenders or not attackers:
        return

    defenders.sort(key=lambda p: p.position.x)

    for idx, defender in enumerate(defenders):
        closest = min(
            attackers,
            key=lambda a: math.hypot(
                defender.position.x - a.position.x,
                defender.position.y - a.position.y,
            ),
        )

        dx = closest.position.x - defender.position.x
        dy = closest.position.y - defender.position.y
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            continue

        step = min(defender.player.max_speed, dist)
        defender.position.x += (dx / dist) * step
        defender.position.y += (dy / dist) * step

        # Maintain ordering / spacing in x direction.
        if idx > 0:
            left = defenders[idx - 1]
            defender.position.x = max(
                defender.position.x, left.position.x + params.min_horizontal_gap
            )
        if idx < len(defenders) - 1:
            right = defenders[idx + 1]
            defender.position.x = min(
                defender.position.x, right.position.x - params.min_horizontal_gap
            )

        if (
            params.compression_threshold is not None
            and ball_holder.position.x >= params.compression_threshold
            and params.compression_factor < 1.0
        ):
            kappa = max(params.compression_factor, 0.0)
            defender.position.x = FIELD_WIDTH - kappa * (FIELD_WIDTH - defender.position.x)
            defender.position.y = FIELD_HEIGHT / 2.0 + kappa * (
                defender.position.y - FIELD_HEIGHT / 2.0
            )

        defender.position.clamp_to_field()
