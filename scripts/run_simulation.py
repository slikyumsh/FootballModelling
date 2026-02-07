#!/usr/bin/env python
"""Run a single Metropolis simulation and save artifacts.

This script is intended to replace the exploratory Jupyter notebook workflow with a
clean, reproducible CLI entrypoint.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from footballmodelling.artifacts import create_ball_focus_heatmap, create_gif, prepare_run_dir
from footballmodelling.config import load_simulation_config
from footballmodelling.initialization import build_pitch
from footballmodelling.simulation import MetropolisAttackSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_simulation")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.simulation.json",
        help="Path to a simulation JSON config.",
    )
    args = parser.parse_args()

    cfg = load_simulation_config(args.config)

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    paths = prepare_run_dir(cfg.paths.artifacts_dir, run_id=run_id)

    pitch, ball_holder = build_pitch(cfg.pitch)

    sim = MetropolisAttackSimulator(
        pitch=pitch,
        ball_holder=ball_holder,
        epsilon=cfg.model.epsilon,
        beta=cfg.model.beta,
        max_steps=cfg.model.max_steps,
        xg_early_stop=cfg.model.xg_early_stop,
        attack_shift_x=cfg.model.attack_shift_x,
        attacking_team=cfg.pitch.attacking_team_id,
        defending_team=cfg.pitch.defending_team_id,
        rng_seed=cfg.random.seed,
    )

    result = sim.run()
    logger.info(
        "Done. final_xg=%.4f passes=%d steps=%d",
        result.final_xg,
        result.passes,
        result.iterations,
    )

    create_gif(
        background_image=cfg.paths.background_image,
        snapshots=sim.snapshots,
        ball_holders=result.ball_holders,
        output_path=paths.gif_path,
        frame_duration_ms=cfg.render.gif_frame_ms,
        font_path=cfg.render.font_path,
    )

    create_ball_focus_heatmap(
        background_image=cfg.paths.background_image,
        snapshots=sim.snapshots,
        ball_holders=result.ball_holders,
        output_path=paths.heatmap_path,
        radius=cfg.render.heatmap_radius,
    )

    logger.info("Artifacts saved under: %s", paths.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
