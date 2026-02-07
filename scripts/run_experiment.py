#!/usr/bin/env python
"""Run a parameter sweep experiment and export CSV results."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from footballmodelling.config import load_experiment_config
from footballmodelling.experiments import build_sims_for_sweep, run_experiment
from footballmodelling.initialization import build_pitch
from footballmodelling.simulation import MetropolisAttackSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_experiment")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.experiment.json",
        help="Path to an experiment JSON config.",
    )
    parser.add_argument("--id", type=str, default="sweep", help="Experiment id prefix.")
    args = parser.parse_args()

    cfg = load_experiment_config(args.config)
    pitch, ball_holder = build_pitch(cfg.pitch)

    base = MetropolisAttackSimulator(
        pitch=pitch,
        ball_holder=ball_holder,
        epsilon=cfg.base_model.epsilon,
        beta=cfg.base_model.beta,
        max_steps=cfg.base_model.max_steps,
        xg_early_stop=cfg.base_model.xg_early_stop,
        attack_shift_x=cfg.base_model.attack_shift_x,
        attacking_team=cfg.pitch.attacking_team_id,
        defending_team=cfg.pitch.defending_team_id,
        rng_seed=cfg.random.seed,
    )

    sims = build_sims_for_sweep(
        base_sim=base,
        param=cfg.sweep.param,
        start=cfg.sweep.start,
        step=cfg.sweep.step,
        count=cfg.sweep.count,
        replications=cfg.replications,
    )

    exp_id = f"{args.id}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    out_csv = run_experiment(
        sims=sims,
        out_dir=cfg.paths.artifacts_dir,
        experiment_id=exp_id,
        processes=cfg.parallel.processes,
    )

    logger.info("Experiment finished: %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
