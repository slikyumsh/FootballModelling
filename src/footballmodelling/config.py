"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json

from .exceptions import ConfigError
from .initialization import PitchInit, PlayerInit


@dataclass(frozen=True)
class PathsConfig:
    """I/O paths for the project."""

    background_image: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    """Model hyperparameters."""

    epsilon: float
    beta: float
    max_steps: int
    xg_early_stop: float
    attack_shift_x: float = 200.0


@dataclass(frozen=True)
class RenderConfig:
    """Rendering parameters for artifacts."""

    gif_frame_ms: int = 120
    heatmap_radius: int = 20
    font_path: Path | None = None


@dataclass(frozen=True)
class RandomConfig:
    """Randomness control."""

    seed: int | None = None


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level config for a single simulation run."""

    paths: PathsConfig
    model: ModelConfig
    render: RenderConfig
    random: RandomConfig
    pitch: PitchInit


@dataclass(frozen=True)
class SweepConfig:
    """Parameter sweep configuration for experiments."""

    param: str
    start: float
    step: float
    count: int


@dataclass(frozen=True)
class ParallelConfig:
    """Parallel execution config."""

    processes: int = 0  # 0 => auto


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config for experiments."""

    paths: PathsConfig
    base_model: ModelConfig
    replications: int
    sweep: SweepConfig
    parallel: ParallelConfig
    random: RandomConfig
    pitch: PitchInit


def _require(d: dict[str, Any], key: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required config key: '{key}'")
    return d[key]


def _parse_player_init(raw: dict[str, Any]) -> PlayerInit:
    """Parse and validate a single player init dictionary."""
    try:
        return PlayerInit(
            number=int(_require(raw, "number")),
            max_speed=float(_require(raw, "max_speed")),
            is_goalkeeper=bool(_require(raw, "is_goalkeeper")),
            x=float(_require(raw, "x")),
            y=float(_require(raw, "y")),
            player_id=int(raw["player_id"]) if "player_id" in raw else None,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid player definition: {raw}") from exc


def _parse_pitch_init(raw: dict[str, Any]) -> PitchInit:
    """Parse pitch/team initialization."""
    attacking_team_id = int(_require(raw, "attacking_team_id"))
    defending_team_id = int(_require(raw, "defending_team_id"))
    ball_holder_id = int(_require(raw, "ball_holder_id"))
    teams_raw = _require(raw, "teams")

    if not isinstance(teams_raw, dict):
        raise ConfigError("pitch.teams must be a dictionary keyed by team id (as string or int).")

    teams: dict[int, list[PlayerInit]] = {}
    for k, roster in teams_raw.items():
        team_id = int(k)
        if not isinstance(roster, list):
            raise ConfigError(f"pitch.teams[{k}] must be a list.")
        teams[team_id] = [_parse_player_init(p) for p in roster]

    return PitchInit(
        attacking_team_id=attacking_team_id,
        defending_team_id=defending_team_id,
        ball_holder_id=ball_holder_id,
        teams=teams,
    )


def load_simulation_config(path: str | Path) -> SimulationConfig:
    """Load and validate simulation config from JSON."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    paths = _require(raw, "paths")
    model = _require(raw, "model")
    render = raw.get("render", {})
    rnd = raw.get("random", {})
    pitch = _require(raw, "pitch")

    return SimulationConfig(
        paths=PathsConfig(
            background_image=Path(_require(paths, "background_image")),
            artifacts_dir=Path(_require(paths, "artifacts_dir")),
        ),
        model=ModelConfig(
            epsilon=float(_require(model, "epsilon")),
            beta=float(_require(model, "beta")),
            max_steps=int(_require(model, "max_steps")),
            xg_early_stop=float(_require(model, "xg_early_stop")),
            attack_shift_x=float(model.get("attack_shift_x", 200.0)),
        ),
        render=RenderConfig(
            gif_frame_ms=int(render.get("gif_frame_ms", 120)),
            heatmap_radius=int(render.get("heatmap_radius", 20)),
            font_path=Path(render["font_path"]) if render.get("font_path") else None,
        ),
        random=RandomConfig(seed=rnd.get("seed")),
        pitch=_parse_pitch_init(pitch),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load and validate experiment config from JSON."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    raw = json.loads(path.read_text(encoding="utf-8"))

    paths = _require(raw, "paths")
    base = _require(raw, "base_model")
    sweep = _require(raw, "sweep")
    parallel = raw.get("parallel", {})
    rnd = raw.get("random", {})
    pitch = _require(raw, "pitch")

    return ExperimentConfig(
        paths=PathsConfig(
            background_image=Path(_require(paths, "background_image")),
            artifacts_dir=Path(_require(paths, "artifacts_dir")),
        ),
        base_model=ModelConfig(
            epsilon=float(_require(base, "epsilon")),
            beta=float(_require(base, "beta")),
            max_steps=int(_require(base, "max_steps")),
            xg_early_stop=float(_require(base, "xg_early_stop")),
            attack_shift_x=float(base.get("attack_shift_x", 200.0)),
        ),
        replications=int(_require(raw, "replications")),
        sweep=SweepConfig(
            param=str(_require(sweep, "param")),
            start=float(_require(sweep, "start")),
            step=float(_require(sweep, "step")),
            count=int(_require(sweep, "count")),
        ),
        parallel=ParallelConfig(processes=int(parallel.get("processes", 0))),
        random=RandomConfig(seed=rnd.get("seed")),
        pitch=_parse_pitch_init(pitch),
    )
