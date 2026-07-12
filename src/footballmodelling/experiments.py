"""Experiment runner: replication + parameter sweeps + CSV export."""

from __future__ import annotations

import copy
import csv
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .simulation import MetropolisAttackSimulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentRow:
    """A single row to be exported to CSV."""

    input_information: dict[str, Any]
    unique_players: int
    passes: int
    iterations: int
    final_xg: float
    screened_candidate_rate: float


def _run_one(sim: MetropolisAttackSimulator) -> ExperimentRow:
    """Run one simulation and return an ExperimentRow."""
    res = sim.run()
    info = {
        "epsilon": sim.defense.interception_radius,
        "beta": sim.beta,
        "max_steps": sim.max_steps,
        "xg_early_stop": sim.xg_early_stop,
        "attack_shift_x": sim.attack_shift_x,
    }
    return ExperimentRow(
        input_information=info,
        unique_players=res.unique_ball_holders,
        passes=res.passes,
        iterations=res.iterations,
        final_xg=res.final_xg,
        screened_candidate_rate=res.screened_candidate_rate,
    )


def save_rows_csv(rows: list[ExperimentRow], out_dir: Path, experiment_id: str) -> Path:
    """Save experiment results to CSV and return file path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = out_dir / f"football_{experiment_id}_{now}.csv"

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "input_information",
                "unique_players",
                "passes",
                "iterations",
                "final_xg",
                "screened_candidate_rate",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "input_information": json.dumps(r.input_information),
                    "unique_players": r.unique_players,
                    "passes": r.passes,
                    "iterations": r.iterations,
                    "final_xg": r.final_xg,
                    "screened_candidate_rate": r.screened_candidate_rate,
                }
            )

    logger.info("Experiment results saved to %s", path)
    return path


def build_sims_for_sweep(
    base_sim: MetropolisAttackSimulator,
    param: str,
    start: float,
    step: float,
    count: int,
    replications: int,
) -> list[MetropolisAttackSimulator]:
    """Create simulator copies for a parameter sweep.

    Raises:
        AttributeError: If the simulator does not expose the swept parameter.
    """
    sims: list[MetropolisAttackSimulator] = []

    for i in range(count):
        value = start + i * step
        for _ in range(replications):
            sim = copy.deepcopy(base_sim)
            if not hasattr(sim, param):
                raise AttributeError(f"Simulator has no attribute '{param}' to sweep.")
            setattr(sim, param, value)
            sims.append(sim)

    return sims


def run_experiment(
    sims: list[MetropolisAttackSimulator],
    out_dir: Path,
    experiment_id: str,
    processes: int = 0,
) -> Path:
    """Run an experiment, optionally in parallel."""
    if processes == 0:
        processes = max(mp.cpu_count() // 2, 1)

    logger.info("Running %d simulations with %d processes...", len(sims), processes)

    if processes == 1:
        rows = [_run_one(s) for s in sims]
    else:
        with mp.Pool(processes) as pool:
            rows = pool.map(_run_one, sims)

    return save_rows_csv(rows, out_dir=out_dir, experiment_id=experiment_id)
