"""Metropolis–Hastings simulation of an attacking sequence."""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .defense import DefenseParams, is_offside, pass_is_intercepted, pressing_step
from .domain import Pitch, PlayerOnPitch, Position
from .exceptions import SimulationError
from .xg import Goal, SimpleLogisticXG

logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Outputs of a single simulation run."""

    final_xg: float
    passes: int
    iterations: int
    unique_ball_holders: int
    ball_holders: list[int]


class MetropolisAttackSimulator:
    """Simulate an attacking possession using a Metropolis–Hastings acceptance rule."""

    def __init__(
        self,
        pitch: Pitch,
        ball_holder: PlayerOnPitch,
        epsilon: float,
        beta: float,
        max_steps: int,
        xg_early_stop: float,
        attack_shift_x: float = 200.0,
        attacking_team: int = 1,
        defending_team: int = 2,
        goal: Position | None = None,
        rng_seed: int | None = None,
    ) -> None:
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)

        self.pitch = pitch
        self.ball_holder = ball_holder
        self.attacking_team = attacking_team
        self.defending_team = defending_team

        self.defense = DefenseParams(interception_radius=epsilon)
        self.beta = float(beta)
        self.max_steps = int(max_steps)
        self.xg_early_stop = float(xg_early_stop)
        self.attack_shift_x = float(attack_shift_x)

        goal_pos = goal or Position(1500.0, 500.0)
        self.xg_model: Callable[[Position], float] = SimpleLogisticXG(Goal(center=goal_pos))

        self.current_xg = self.xg_model(self.ball_holder.position)
        self.passes = 0

        self.ball_holders: list[int] = [self.ball_holder.player.player_id]
        self.snapshots: list[Pitch] = [copy.deepcopy(self.pitch)]

    def _candidate_receivers(self) -> list[PlayerOnPitch]:
        """Collect valid passing options for the current ball holder."""
        receivers: list[PlayerOnPitch] = []
        for p in self.pitch.attackers(self.attacking_team):
            if p.player.player_id == self.ball_holder.player.player_id:
                continue
            if is_offside(p, self.pitch, self.ball_holder, self.defending_team):
                continue
            if pass_is_intercepted(self.ball_holder, p, self.pitch, self.defending_team, self.defense):
                continue
            receivers.append(p)
        return receivers

    def _metropolis_accept(self, new_xg: float) -> bool:
        """Metropolis acceptance decision.

        Passes that improve xG are always accepted.
        Passes that worsen xG are accepted with probability exp(beta*(new-current)).
        """
        if new_xg >= self.current_xg:
            return True

        prob = math.exp((new_xg - self.current_xg) * self.beta)
        return prob > random.random()

    def _attack_shift(self) -> None:
        """Deterministic team shift to mimic pushing the block forward."""
        for p in self.pitch.attackers(self.attacking_team):
            p.position.x += self.attack_shift_x
            p.position.clamp_to_field()

    def _random_attacker_movement(self) -> None:
        """Apply bounded random movement to attackers based on their max speed."""
        for p in self.pitch.attackers(self.attacking_team):
            v = float(p.player.max_speed)

            shift_x = random.uniform(-v / 1.5, v)
            remaining = max(v * v - shift_x * shift_x, 0.0)
            shift_y = random.uniform(-math.sqrt(remaining), math.sqrt(remaining))

            p.position.x += shift_x
            p.position.y += shift_y
            p.position.clamp_to_field()

    def step(self) -> bool:
        """Perform one simulation step.

        Returns:
            True if a pass was accepted (ball holder changed), False otherwise.
        """
        receivers = self._candidate_receivers()
        if not receivers:
            logger.info("No valid passing options available.")
            return False

        trials = 2 * len(receivers)
        for _ in range(trials):
            chosen = random.choice(receivers)
            new_xg = self.xg_model(chosen.position)

            if self._metropolis_accept(new_xg):
                self.ball_holder = chosen
                self.current_xg = new_xg
                self.passes += 1
                return True

        logger.info("No candidate pass accepted by Metropolis rule.")
        return False

    def run(self) -> SimulationResult:
        """Run the simulation until max steps or early stop condition.

        Raises:
            SimulationError: If initial ball holder is invalid.
        """
        if self.ball_holder.player.team_id != self.attacking_team:
            raise SimulationError("Ball holder must belong to the attacking team.")

        self._attack_shift()

        for _ in range(self.max_steps):
            if self.current_xg >= self.xg_early_stop:
                break

            _ = self.step()

            pressing_step(
                self.pitch,
                attacking_team=self.attacking_team,
                defending_team=self.defending_team,
                ball_holder=self.ball_holder,
                params=self.defense,
            )

            self._random_attacker_movement()

            self.ball_holders.append(self.ball_holder.player.player_id)
            self.snapshots.append(copy.deepcopy(self.pitch))

        unique = len(set(self.ball_holders))
        return SimulationResult(
            final_xg=self.current_xg,
            passes=self.passes,
            iterations=len(self.snapshots),
            unique_ball_holders=unique,
            ball_holders=self.ball_holders,
        )
