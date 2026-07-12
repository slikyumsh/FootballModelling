"""Metropolis-inspired simulation of an attacking sequence."""

from __future__ import annotations

import copy
import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .defense import DefenseParams, is_offside, pass_is_intercepted, pressing_step
from .domain import Pitch, PlayerOnPitch, Position
from .exceptions import SimulationError
from .xg import Goal, SimpleLogisticXG

class BetaPolicy:
    """Callable policy returning beta_t at a given simulation step.

    We keep this intentionally minimal: the policy can depend on the step index,
    current ball-holder position, and current xG if needed.
    """

    def __call__(self, step_idx: int, ball_x: float, current_xg: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class PiecewiseStepBeta(BetaPolicy):
    """Piecewise-constant beta over time (simulation steps).

    Example: betas=[30, 100, 300], cutoffs=[20, 60]
      steps 0..19   -> 30
      steps 20..59  -> 100
      steps >=60    -> 300
    """
    betas: Sequence[float]
    cutoffs: Sequence[int]

    def __post_init__(self) -> None:
        if len(self.betas) != len(self.cutoffs) + 1:
            raise ValueError("cutoffs must have length len(betas) - 1")
        if any(c < 0 for c in self.cutoffs):
            raise ValueError("cutoffs must be non-negative")
        if list(self.cutoffs) != sorted(self.cutoffs):
            raise ValueError("cutoffs must be sorted increasingly")

    def __call__(self, step_idx: int, ball_x: float, current_xg: float) -> float:
        for i, c in enumerate(self.cutoffs):
            if step_idx < c:
                return float(self.betas[i])
        return float(self.betas[-1])


@dataclass(frozen=True)
class DiscreteBetaSampler(BetaPolicy):
    """Sample beta_t i.i.d. from a discrete distribution each step."""
    values: Sequence[float]
    probs: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("values must be non-empty")
        if self.probs is not None:
            if len(self.probs) != len(self.values):
                raise ValueError("probs must match values length")
            if any(p < 0 for p in self.probs):
                raise ValueError("probs must be non-negative")
            s = sum(self.probs)
            if s <= 0:
                raise ValueError("sum(probs) must be > 0")

    def __call__(self, step_idx: int, ball_x: float, current_xg: float) -> float:
        # Uses Python's random module; reproducibility is handled by rng_seed in the simulator.
        return float(random.choices(self.values, weights=self.probs, k=1)[0])


logger = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    """Outputs of a single simulation run."""

    final_xg: float
    passes: int
    iterations: int
    unique_ball_holders: int
    ball_holders: list[int]
    screened_candidates: int
    candidate_options: int

    @property
    def screened_candidate_rate(self) -> float:
        """Fraction of post-offside-filter candidates screened out geometrically."""
        if self.candidate_options == 0:
            return 0.0
        return self.screened_candidates / self.candidate_options


class MetropolisAttackSimulator:
    """Simulate an attacking possession using a Metropolis-inspired acceptance rule."""

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
        beta_policy: Callable[[int, float, float], float] | None = None,
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
        self.beta_policy = beta_policy

        self.max_steps = int(max_steps)
        self.xg_early_stop = float(xg_early_stop)
        self.attack_shift_x = float(attack_shift_x)

        goal_pos = goal or Position(1500.0, 500.0)
        self.xg_model: Callable[[Position], float] = SimpleLogisticXG(Goal(center=goal_pos))

        self.current_xg = self.xg_model(self.ball_holder.position)
        self.passes = 0

        self.ball_holders: list[int] = [self.ball_holder.player.player_id]
        self.snapshots: list[Pitch] = [copy.deepcopy(self.pitch)]

        # Traces for analysis/plots (optional but useful for the paper)
        self.beta_trace: list[float] = []
        self.xg_trace: list[float] = [self.current_xg]
        self._beta_last: float = self.beta
        self.screened_candidates = 0
        self.candidate_options = 0


    def _candidate_receivers(self) -> list[PlayerOnPitch]:
        """Collect valid passing options for the current ball holder."""
        receivers: list[PlayerOnPitch] = []
        for p in self.pitch.attackers(self.attacking_team):
            if p.player.player_id == self.ball_holder.player.player_id:
                continue
            if is_offside(p, self.pitch, self.ball_holder, self.defending_team):
                continue
            self.candidate_options += 1
            if pass_is_intercepted(self.ball_holder, p, self.pitch, self.defending_team, self.defense):
                self.screened_candidates += 1
                continue
            receivers.append(p)
        return receivers
    
    def _metropolis_accept(self, new_xg: float, beta_t: float) -> bool:
        """Metropolis acceptance decision with possibly time-varying beta_t.

        Passes that improve xG are always accepted.
        Passes that worsen xG are accepted with probability exp(beta_t*(new-current)).
        """
        if new_xg >= self.current_xg:
            return True
        
        prob = math.exp((new_xg - self.current_xg) * beta_t)
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
    
    def step(self, step_idx: int) -> bool:
        """Perform one simulation step.

        Args:
            step_idx: current step index (0-based)

        Returns:
            True if a pass was accepted (ball holder changed), False otherwise.
        """
        receivers = self._candidate_receivers()
        if not receivers:
            logger.info("No valid passing options available.")
            return False

        # Determine beta_t for this step (constant by default).
        beta_t = self.beta
        if self.beta_policy is not None:
            beta_t = float(self.beta_policy(step_idx, self.ball_holder.position.x, self.current_xg))

        trials = 2 * len(receivers)
        for _ in range(trials):
            chosen = random.choice(receivers)
            new_xg = self.xg_model(chosen.position)

            if self._metropolis_accept(new_xg, beta_t):
                self.ball_holder = chosen
                self.current_xg = new_xg
                self.passes += 1
                return True

        logger.info("No candidate pass accepted by Metropolis rule.")
        return False

    def run(self) -> SimulationResult:
        """Run the simulation until max steps or early stop condition.

        Supports both:
        - constant beta (self.beta), and
        - time-varying beta via an optional callable self.beta_policy(step_idx, ball_x, current_xg).

        The beta policy affects only the Metropolis acceptance probability; the rest of
        the simulation (pressing + bounded random movement) is unchanged.

        Raises:
            SimulationError: If initial ball holder is invalid.
        """
        if self.ball_holder.player.team_id != self.attacking_team:
            raise SimulationError("Ball holder must belong to the attacking team.")

        # Ensure optional traces exist (useful for plotting/analysis).
        if not hasattr(self, "beta_trace"):
            self.beta_trace = []
        if not hasattr(self, "xg_trace"):
            self.xg_trace = [self.current_xg]

        self._attack_shift()

        for step_idx in range(self.max_steps):
            if self.current_xg >= self.xg_early_stop:
                break

            # Compute beta_t once per step for logging/analysis.
            beta_t = self.beta
            beta_policy = getattr(self, "beta_policy", None)
            if beta_policy is not None:
                beta_t = float(beta_policy(step_idx, self.ball_holder.position.x, self.current_xg))

             
            _ = self.step(step_idx)


            # Apply defensive pressing (bounded by each defender's max_speed).
            pressing_step(
                self.pitch,
                attacking_team=self.attacking_team,
                defending_team=self.defending_team,
                ball_holder=self.ball_holder,
                params=self.defense,
            )

            # Apply bounded random movement to attackers (bounded by each attacker's max_speed).
            self._random_attacker_movement()

            # Record traces after the step (beta used for acceptance; xG after possible pass acceptance).
            self.beta_trace.append(beta_t)
            self.xg_trace.append(self.current_xg)

            # Store artifacts as before (align ball_holders with snapshots).
            self.ball_holders.append(self.ball_holder.player.player_id)
            self.snapshots.append(copy.deepcopy(self.pitch))

        unique = len(set(self.ball_holders))
        return SimulationResult(
            final_xg=self.current_xg,
            passes=self.passes,
            iterations=len(self.snapshots),
            unique_ball_holders=unique,
            ball_holders=self.ball_holders,
            screened_candidates=self.screened_candidates,
            candidate_options=self.candidate_options,
        )
