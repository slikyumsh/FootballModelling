#!/usr/bin/env python
"""Run experiments comparing constant beta vs beta(t) policies.

Outputs:
- CSV with per-run metrics
- Boxplots: final_xg, passes
- Distribution plots (hist overlays): final_xg, passes
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from footballmodelling.config import load_simulation_config
from footballmodelling.initialization import build_pitch
from footballmodelling.simulation import (
    MetropolisAttackSimulator,
    PiecewiseStepBeta,
    DiscreteBetaSampler,
)


def run_replications(sim_factory, n: int, seed0: int):
    """Run n replications with different seeds and return rows."""
    rows = []
    for i in range(n):
        sim = sim_factory(seed0 + i)
        res = sim.run()
        rows.append(
            {
                "final_xg": res.final_xg,
                "passes": res.passes,
                "iterations": res.iterations,
                "unique_ball_holders": res.unique_ball_holders,
                "mean_beta_used": (sum(sim.beta_trace) / len(sim.beta_trace)) if getattr(sim, "beta_trace", None) else sim.beta,
            }
        )
    return rows


def plot_hist_overlay(values_by_label, labels, xlabel, out_path, bins=30):
    """Overlayed histogram distributions for multiple labels (matplotlib only)."""
    plt.figure()
    for lab in labels:
        vals = values_by_label[lab]
        plt.hist(vals, bins=bins, density=True, alpha=0.4, label=lab)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.simulation.json")
    parser.add_argument("--out_dir", type=str, default="artifacts/beta_policy")
    parser.add_argument("--replications", type=int, default=200)
    args = parser.parse_args()

    cfg = load_simulation_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pitch, ball_holder = build_pitch(cfg.pitch)

    # --- Define policies ---
    policies = [
        ("constant_beta_0.5", None),
        ("piecewise_1.5_0.5_0.15", PiecewiseStepBeta(betas=[1.5, 0.5, 0.15], cutoffs=[100, 300])),
        ("sampled_{1.5,0.5,0.15}", DiscreteBetaSampler(values=[1.5, 0.5, 0.15], probs=[0.4, 0.4, 0.2])),
    ]

    all_rows = []

    for name, policy in policies:

        def make_sim(seed: int):
            return MetropolisAttackSimulator(
                pitch=pitch,
                ball_holder=ball_holder,
                epsilon=cfg.model.epsilon,
                beta=cfg.model.beta if policy is None else 0.5,  # baseline beta retained for logging
                max_steps=cfg.model.max_steps,
                xg_early_stop=cfg.model.xg_early_stop,
                attack_shift_x=cfg.model.attack_shift_x,
                attacking_team=cfg.pitch.attacking_team_id,
                defending_team=cfg.pitch.defending_team_id,
                rng_seed=seed,
                beta_policy=policy,
            )

        rows = run_replications(make_sim, n=args.replications, seed0=(cfg.random.seed or 0) + 10_000)
        for r in rows:
            r["policy"] = name
        all_rows.extend(rows)

    # --- Save CSV ---
    csv_path = out_dir / "beta_policy_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["policy", "final_xg", "passes", "iterations", "unique_ball_holders", "mean_beta_used"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    labels = [p[0] for p in policies]

    # --- Distribution Plot: passes (overlayed histograms) ---
    passes_by_label = {lab: [r["passes"] for r in all_rows if r["policy"] == lab] for lab in labels}
    plot_hist_overlay(
        values_by_label=passes_by_label,
        labels=labels,
        xlabel="Number of accepted passes",
        out_path=out_dir / "passes_distribution_by_beta_policy.pdf",
        bins=30,
    )

    print(f"Saved: {csv_path}")
    print(f"Saved plots in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
