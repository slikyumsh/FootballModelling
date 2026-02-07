"""Domain objects: positions, players, teams, and the pitch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


FIELD_WIDTH = 1500.0
FIELD_HEIGHT = 1000.0


@dataclass
class Position:
    """2D position on the pitch in pixel coordinates."""

    x: float
    y: float

    def clamp_to_field(self) -> None:
        """Clamp position within field bounds."""
        self.x = min(max(self.x, 0.0), FIELD_WIDTH)
        self.y = min(max(self.y, 0.0), FIELD_HEIGHT)


@dataclass(frozen=True)
class Player:
    """A player identity and physical capability."""

    player_id: int
    team_id: int
    max_speed: float
    is_goalkeeper: bool = False


@dataclass
class PlayerOnPitch:
    """A player and their current position."""

    player: Player
    position: Position


@dataclass
class Pitch:
    """A pitch with players from both teams."""

    players: list[PlayerOnPitch]

    def attackers(self, team_id: int) -> list[PlayerOnPitch]:
        """Return outfield players from the attacking team."""
        return [
            p
            for p in self.players
            if p.player.team_id == team_id and not p.player.is_goalkeeper
        ]

    def defenders(self, team_id: int) -> list[PlayerOnPitch]:
        """Return outfield players from the defending team."""
        return [
            p
            for p in self.players
            if p.player.team_id == team_id and not p.player.is_goalkeeper
        ]

    @staticmethod
    def validate_two_teams(players: Iterable[PlayerOnPitch]) -> None:
        """Validate that each team has exactly 11 players and exactly 1 goalkeeper.

        Args:
            players: Iterable of all players on the pitch.

        Raises:
            ValueError: If roster size or goalkeeper count is invalid.
        """
        players_list = list(players)

        by_team: dict[int, list[PlayerOnPitch]] = {}
        for p in players_list:
            by_team.setdefault(p.player.team_id, []).append(p)

        if len(by_team) != 2:
            raise ValueError(f"Expected 2 teams, got {len(by_team)}")

        for team_id, roster in by_team.items():
            if len(roster) != 11:
                raise ValueError(
                    f"Team {team_id} must have 11 players, got {len(roster)}"
                )
            gk_count = sum(1 for p in roster if p.player.is_goalkeeper)
            if gk_count != 1:
                raise ValueError(
                    f"Team {team_id} must have exactly 1 goalkeeper, got {gk_count}"
                )

        # Ensure player_id uniqueness
        ids = [p.player.player_id for p in players_list]
        if len(set(ids)) != len(ids):
            raise ValueError("player_id must be unique across the pitch")
