"""Pitch/roster initialization from configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .domain import Pitch, Player, PlayerOnPitch, Position
from .exceptions import ConfigError, SimulationError


@dataclass(frozen=True)
class PlayerInit:
    """Player initialization payload."""
    number: int
    max_speed: float
    is_goalkeeper: bool
    x: float
    y: float
    player_id: int | None = None


@dataclass(frozen=True)
class PitchInit:
    """Pitch initialization payload."""
    attacking_team_id: int
    defending_team_id: int
    ball_holder_id: int
    teams: dict[int, list[PlayerInit]]


def _resolve_player_id(team_id: int, p: PlayerInit) -> int:
    """Compute player_id if not explicitly set in config."""
    if p.player_id is not None:
        return p.player_id
    # Mirror original research prototype IDs: number + team_id * 100
    return int(p.number + team_id * 100)


def build_pitch(pitch_init: PitchInit) -> tuple[Pitch, PlayerOnPitch]:
    """Build a Pitch and locate the initial ball holder.

    Raises:
        ConfigError: If teams/rosters are invalid.
        SimulationError: If the specified ball holder cannot be found.
    """
    if set(pitch_init.teams.keys()) != {pitch_init.attacking_team_id, pitch_init.defending_team_id}:
        raise ConfigError(
            "pitch.teams must contain exactly two teams: attacking_team_id and defending_team_id."
        )

    players: list[PlayerOnPitch] = []
    for team_id, roster in pitch_init.teams.items():
        if len(roster) != 11:
            raise ConfigError(f"Team {team_id} must have 11 players, got {len(roster)}")
        gk_count = sum(1 for p in roster if p.is_goalkeeper)
        if gk_count != 1:
            raise ConfigError(f"Team {team_id} must have exactly 1 goalkeeper, got {gk_count}")

        for p in roster:
            pid = _resolve_player_id(team_id, p)
            player = Player(
                player_id=pid,
                team_id=int(team_id),
                max_speed=float(p.max_speed),
                is_goalkeeper=bool(p.is_goalkeeper),
            )
            pos = Position(x=float(p.x), y=float(p.y))
            players.append(PlayerOnPitch(player=player, position=pos))

    pitch = Pitch(players=players)

    # Validate uniqueness and roster rules at domain layer too
    Pitch.validate_two_teams(pitch.players)

    ball_holder = next((p for p in pitch.players if p.player.player_id == pitch_init.ball_holder_id), None)
    if ball_holder is None:
        raise SimulationError(
            f"ball_holder_id={pitch_init.ball_holder_id} not found on the pitch. "
            "Check pitch.ball_holder_id and player IDs."
        )

    if ball_holder.player.team_id != pitch_init.attacking_team_id:
        raise SimulationError("Initial ball holder must belong to the attacking team.")

    return pitch, ball_holder
