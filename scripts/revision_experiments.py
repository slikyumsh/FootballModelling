#!/usr/bin/env python
"""Generate revision figures and numerical summaries for the manuscript."""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import random
import statistics
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from footballmodelling.config import SimulationConfig, load_simulation_config
from footballmodelling.defense import DefenseParams, pressing_step
from footballmodelling.domain import Position
from footballmodelling.initialization import PitchInit, build_pitch
from footballmodelling.simulation import (
    DiscreteBetaSampler,
    MetropolisAttackSimulator,
    PiecewiseStepBeta,
    SimulationResult,
)


ARTICLE_DIR = Path("Metropolis_article_new_1804_1_not_anon")
BASE_BETA = 50_000.0
MAX_STEPS = 220
XG_EARLY_STOP = 5e-4
DISPLAY_XG_SCALE = 1000.0
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0


def set_outfield_speed(pitch: PitchInit, attacking_speed: float, defending_speed: float) -> PitchInit:
    """Return a pitch init with uniform outfield movement capacities."""
    teams = {}
    for team_id, roster in pitch.teams.items():
        speed = attacking_speed if team_id == pitch.attacking_team_id else defending_speed
        teams[team_id] = [
            replace(player, max_speed=0.0 if player.is_goalkeeper else speed)
            for player in roster
        ]
    return replace(pitch, teams=teams)


def run_one(
    cfg: SimulationConfig,
    seed: int,
    *,
    beta: float = BASE_BETA,
    pitch_init: PitchInit | None = None,
    defense: DefenseParams | None = None,
    beta_policy=None,
) -> tuple[SimulationResult, MetropolisAttackSimulator]:
    """Run one simulation with the article-revision settings."""
    pitch, ball_holder = build_pitch(pitch_init or cfg.pitch)
    sim = MetropolisAttackSimulator(
        pitch=pitch,
        ball_holder=ball_holder,
        epsilon=cfg.model.epsilon,
        beta=beta,
        max_steps=MAX_STEPS,
        xg_early_stop=XG_EARLY_STOP,
        attack_shift_x=cfg.model.attack_shift_x,
        attacking_team=cfg.pitch.attacking_team_id,
        defending_team=cfg.pitch.defending_team_id,
        rng_seed=seed,
        beta_policy=beta_policy,
    )
    if defense is not None:
        sim.defense = defense
    return sim.run(), sim


def summarize(results: list[SimulationResult]) -> dict[str, float]:
    """Summarize repeated simulations as means and sample standard deviations."""
    def mean(values: list[float]) -> float:
        return statistics.fmean(values)

    def sd(values: list[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    final_xg = [r.final_xg for r in results]
    passes = [float(r.passes) for r in results]
    iterations = [float(r.iterations) for r in results]
    screened = [r.screened_candidate_rate for r in results]
    return {
        "n": float(len(results)),
        "final_xg_mean": mean(final_xg),
        "final_xg_sd": sd(final_xg),
        "passes_mean": mean(passes),
        "passes_sd": sd(passes),
        "iterations_mean": mean(iterations),
        "iterations_sd": sd(iterations),
        "screened_rate_mean": mean(screened),
        "screened_rate_sd": sd(screened),
    }


def run_repeated(cfg: SimulationConfig, n: int, seed0: int, **kwargs) -> list[SimulationResult]:
    """Run n replications and return only the result objects."""
    return [run_one(cfg, seed0 + i, **kwargs)[0] for i in range(n)]


def _pass_length_m(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Convert an internal-coordinate pass segment to an approximate metre length."""
    dx = (end[0] - start[0]) * FIELD_LENGTH_M / 1500.0
    dy = (end[1] - start[1]) * FIELD_WIDTH_M / 1000.0
    return math.hypot(dx, dy)


def run_endpoint_noise_simulation(
    cfg: SimulationConfig,
    seed: int,
    endpoint_noise_sd_m: float,
    receiver_control_radius_m: float = 2.5,
) -> dict[str, float]:
    """Run a lightweight technical-error sensitivity simulation.

    Endpoint noise is treated as an additional feasibility screen. A sampled
    endpoint outside the receiver control radius is discarded as misplaced.
    """
    pitch, ball_holder = build_pitch(cfg.pitch)
    sim = MetropolisAttackSimulator(
        pitch=pitch,
        ball_holder=ball_holder,
        epsilon=cfg.model.epsilon,
        beta=BASE_BETA,
        max_steps=MAX_STEPS,
        xg_early_stop=XG_EARLY_STOP,
        attack_shift_x=cfg.model.attack_shift_x,
        attacking_team=cfg.pitch.attacking_team_id,
        defending_team=cfg.pitch.defending_team_id,
        rng_seed=seed,
    )
    sim._attack_shift()

    sx = FIELD_LENGTH_M / 1500.0
    sy = FIELD_WIDTH_M / 1000.0
    sd_x = endpoint_noise_sd_m / sx if endpoint_noise_sd_m > 0 else 0.0
    sd_y = endpoint_noise_sd_m / sy if endpoint_noise_sd_m > 0 else 0.0
    proposal_count = 0
    misplaced_count = 0
    pass_lengths: list[float] = []
    step_idx = 0

    for step_idx in range(sim.max_steps):
        if sim.current_xg >= sim.xg_early_stop:
            break

        receivers = sim._candidate_receivers()
        if receivers:
            trials = 2 * len(receivers)
            for _ in range(trials):
                chosen = random.choice(receivers)
                start = (sim.ball_holder.position.x, sim.ball_holder.position.y)
                intended = (chosen.position.x, chosen.position.y)
                proposal_count += 1

                endpoint = intended
                if endpoint_noise_sd_m > 0:
                    endpoint = (
                        min(max(random.gauss(intended[0], sd_x), 0.0), 1500.0),
                        min(max(random.gauss(intended[1], sd_y), 0.0), 1000.0),
                    )
                    error_m = _pass_length_m(intended, endpoint)
                    if error_m > receiver_control_radius_m:
                        misplaced_count += 1
                        continue

                new_xg = sim.xg_model(Position(endpoint[0], endpoint[1]))
                if sim._metropolis_accept(new_xg, sim.beta):
                    sim.ball_holder = chosen
                    sim.current_xg = new_xg
                    sim.passes += 1
                    pass_lengths.append(_pass_length_m(start, intended))
                    break

        pressing_step(
            sim.pitch,
            attacking_team=sim.attacking_team,
            defending_team=sim.defending_team,
            ball_holder=sim.ball_holder,
            params=sim.defense,
        )
        sim._random_attacker_movement()

    return {
        "final_xg": sim.current_xg,
        "passes": float(sim.passes),
        "iterations": float(step_idx + 1),
        "misplaced_rate": misplaced_count / proposal_count if proposal_count else 0.0,
        "mean_pass_length_m": statistics.fmean(pass_lengths) if pass_lengths else 0.0,
        "short_share": sum(length < 15.0 for length in pass_lengths) / len(pass_lengths)
        if pass_lengths
        else 0.0,
        "medium_share": sum(15.0 <= length < 30.0 for length in pass_lengths) / len(pass_lengths)
        if pass_lengths
        else 0.0,
        "long_share": sum(length >= 30.0 for length in pass_lengths) / len(pass_lengths)
        if pass_lengths
        else 0.0,
    }


def summarize_dict_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    """Summarize dictionaries of scalar metrics as means and sample SDs."""
    out: dict[str, float] = {"n": float(len(rows))}
    for key in rows[0]:
        values = [row[key] for row in rows]
        out[f"{key}_mean"] = statistics.fmean(values)
        out[f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return out


def errorbar(ax, x_values, results_by_x, metric: str, ylabel: str) -> None:
    means = [summarize(results_by_x[x])[f"{metric}_mean"] for x in x_values]
    sds = [summarize(results_by_x[x])[f"{metric}_sd"] for x in x_values]
    ax.errorbar(x_values, means, yerr=sds, marker="o", linewidth=1.6, capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\beta$ (inverse location-xG score)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_beta_figures(beta_results: dict[float, list[SimulationResult]], out_dir: Path) -> None:
    beta_values = list(beta_results)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    errorbar(ax, beta_values, beta_results, "passes", "Accepted passes")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1a.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    errorbar(ax, beta_values, beta_results, "iterations", "Simulation steps")
    fig.tight_layout()
    fig.savefig(out_dir / "fig1b.pdf")
    plt.close(fig)


def plot_speed_figures(
    speed_results: dict[str, list[SimulationResult]],
    grid_results: dict[tuple[int, int], list[SimulationResult]],
    out_dir: Path,
) -> None:
    labels = list(speed_results)
    values = [[r.final_xg * DISPLAY_XG_SCALE for r in speed_results[label]] for label in labels]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.violinplot(values, showmeans=True, showextrema=False)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Final location-xG score")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2a.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for (attack_speed, defense_speed), results in grid_results.items():
        summary = summarize(results)
        marker_size = 18 + summary["iterations_mean"] * 0.7
        sc = ax.scatter(
            attack_speed,
            defense_speed,
            c=summary["final_xg_mean"] * DISPLAY_XG_SCALE,
            s=marker_size,
            cmap="viridis",
            vmin=0,
            vmax=max(summarize(v)["final_xg_mean"] * DISPLAY_XG_SCALE for v in grid_results.values()),
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xlabel("Attacking movement capacity (units/step)")
    ax.set_ylabel("Defensive movement capacity (units/step)")
    ax.set_xticks([30, 50, 70])
    ax.set_yticks([30, 50, 70])
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Mean final xG score")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2b.pdf")
    plt.close(fig)


def plot_compactness_figure(compact_results: dict[str, list[SimulationResult]], out_dir: Path) -> None:
    labels = list(compact_results)
    values = [[r.screened_candidate_rate for r in compact_results[label]] for label in labels]
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.violinplot(values, showmeans=True, showextrema=False)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Screened candidate-pass rate")
    ax.set_ylim(0, max(max(v) for v in values) * 1.15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3.pdf")
    plt.close(fig)


def plot_policy_figure(policy_results: dict[str, list[SimulationResult]], out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    bins = range(0, MAX_STEPS + 20, 10)
    for label, results in policy_results.items():
        ax.hist([r.passes for r in results], bins=bins, density=True, alpha=0.42, label=label)
    ax.set_xlabel("Accepted passes")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "passes_distribution_by_beta_policy.pdf")
    plt.close(fig)


def plot_sequence_figure(cfg: SimulationConfig, out_dir: Path) -> None:
    result, sim, pass_events, carrier_moves = run_sequence_trace(
        cfg,
        seed=89_990,
        beta=1_000_000.0,
        shown_passes=8,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.set_facecolor("#eff5ec")
    ax.plot([0, 1500, 1500, 0, 0], [0, 0, 1000, 1000, 0], color="#57745b", linewidth=1.2)
    ax.axhline(500, color="#9fb29d", linewidth=0.8, linestyle="--")
    ax.axvline(750, color="#9fb29d", linewidth=0.8, linestyle="--")
    penalty_x = 1500 - 1500 * 16.5 / 105
    six_yard_x = 1500 - 1500 * 5.5 / 105
    penalty_spot_x = 1500 - 1500 * 11 / 105
    penalty_half_width = 1000 * 40.32 / 68 / 2
    six_yard_half_width = 1000 * 18.32 / 68 / 2
    ax.add_patch(
        Rectangle(
            (penalty_x, 500 - penalty_half_width),
            1500 - penalty_x,
            2 * penalty_half_width,
            fill=False,
            edgecolor="#57745b",
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Rectangle(
            (six_yard_x, 500 - six_yard_half_width),
            1500 - six_yard_x,
            2 * six_yard_half_width,
            fill=False,
            edgecolor="#779077",
            linewidth=0.8,
        )
    )
    ax.scatter([penalty_spot_x], [500], s=12, color="#57745b", zorder=2)

    first_pitch = sim.snapshots[0]
    for player in first_pitch.players:
        color = "#1f77b4" if player.player.team_id == cfg.pitch.attacking_team_id else "#c23b32"
        ax.scatter(player.position.x, player.position.y, s=35, color=color, alpha=0.35)

    for start, end, _player_id in carrier_moves:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=1.0,
                linestyle=(0, (3, 3)),
                color="#6f7782",
                alpha=0.75,
                zorder=3,
            )
        )

    for order, event in enumerate(pass_events, start=1):
        start = event["start"]
        end = event["end"]
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.45,
                color="#111111",
                alpha=0.88,
                connectionstyle="arc3,rad=0.035",
                zorder=4,
            )
        )
        ax.text(end[0], end[1], str(order), ha="center", va="center", fontsize=8, color="white",
                bbox={"boxstyle": "circle,pad=0.18", "fc": "#111111", "ec": "none", "alpha": 0.9})

    ax.set_xlim(0, 1500)
    ax.set_ylim(1000, 0)
    ax.set_xlabel("Attacking depth x")
    ax.set_ylabel("Lateral coordinate y")
    pass_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", color="#111111", linewidth=1.45)
    move_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", color="#6f7782",
                                  linewidth=1.0, linestyle=(0, (3, 3)))
    ax.legend(
        [pass_handle, move_handle],
        ["accepted pass", "ball-carrier movement"],
        loc="lower left",
        frameon=False,
        fontsize=8,
    )
    ax.set_title("Example accepted-pass sequence", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_sequence.pdf")
    plt.close(fig)


def run_sequence_trace(
    cfg: SimulationConfig,
    seed: int,
    beta: float,
    shown_passes: int = 8,
) -> tuple[SimulationResult, MetropolisAttackSimulator, list[dict[str, object]], list[tuple[tuple[float, float], tuple[float, float], int]]]:
    """Run one sequence and record pass endpoints before post-step movement."""
    pitch, ball_holder = build_pitch(cfg.pitch)
    sim = MetropolisAttackSimulator(
        pitch=pitch,
        ball_holder=ball_holder,
        epsilon=cfg.model.epsilon,
        beta=beta,
        max_steps=MAX_STEPS,
        xg_early_stop=XG_EARLY_STOP,
        attack_shift_x=cfg.model.attack_shift_x,
        attacking_team=cfg.pitch.attacking_team_id,
        defending_team=cfg.pitch.defending_team_id,
        rng_seed=seed,
    )
    sim._attack_shift()
    sim.snapshots = [copy.deepcopy(sim.pitch)]
    sim.ball_holders = [sim.ball_holder.player.player_id]

    pass_events: list[dict[str, object]] = []
    carrier_moves: list[tuple[tuple[float, float], tuple[float, float], int]] = []
    receive_position = (sim.ball_holder.position.x, sim.ball_holder.position.y)

    for step_idx in range(sim.max_steps):
        if sim.current_xg >= sim.xg_early_stop or len(pass_events) >= shown_passes:
            break

        old_holder = sim.ball_holder
        pass_start = (old_holder.position.x, old_holder.position.y)
        accepted = sim.step(step_idx)

        if accepted and sim.ball_holder.player.player_id != old_holder.player.player_id:
            pass_end = (sim.ball_holder.position.x, sim.ball_holder.position.y)
            if math.hypot(pass_start[0] - receive_position[0], pass_start[1] - receive_position[1]) > 20:
                carrier_moves.append((receive_position, pass_start, old_holder.player.player_id))
            pass_events.append(
                {
                    "step": step_idx,
                    "passer": old_holder.player.player_id,
                    "receiver": sim.ball_holder.player.player_id,
                    "start": pass_start,
                    "end": pass_end,
                    "xg": sim.current_xg,
                }
            )
            receive_position = pass_end

        pressing_step(
            sim.pitch,
            attacking_team=sim.attacking_team,
            defending_team=sim.defending_team,
            ball_holder=sim.ball_holder,
            params=sim.defense,
        )
        sim._random_attacker_movement()
        sim.beta_trace.append(sim.beta)
        sim.xg_trace.append(sim.current_xg)
        sim.ball_holders.append(sim.ball_holder.player.player_id)

    unique = len(set(sim.ball_holders))
    result = SimulationResult(
        final_xg=sim.current_xg,
        passes=sim.passes,
        iterations=len(sim.ball_holders),
        unique_ball_holders=unique,
        ball_holders=sim.ball_holders,
        screened_candidates=sim.screened_candidates,
        candidate_options=sim.candidate_options,
    )
    return result, sim, pass_events, carrier_moves


def write_summary_csv(summary_rows: list[dict[str, str | float]], out_dir: Path) -> None:
    path = out_dir / "revision_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def write_technical_error_csv(summary_rows: list[dict[str, str | float]], out_dir: Path) -> None:
    path = out_dir / "technical_error_sensitivity.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def add_summary_row(rows, experiment: str, regime: str, results: list[SimulationResult]) -> None:
    summary = summarize(results)
    rows.append({"experiment": experiment, "regime": regime, **summary})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/beta.simulation.json")
    parser.add_argument("--out-dir", default=str(ARTICLE_DIR))
    parser.add_argument("--replications", type=int, default=120)
    args = parser.parse_args()

    logging.getLogger("footballmodelling.simulation").setLevel(logging.WARNING)
    cfg = load_simulation_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | float]] = []

    beta_values = [100.0, 10_000.0, 50_000.0, 100_000.0, 500_000.0]
    beta_results = {
        beta: run_repeated(cfg, args.replications, 10_000 + int(beta), beta=beta)
        for beta in beta_values
    }
    plot_beta_figures(beta_results, out_dir)
    for beta in [100.0, 50_000.0, 500_000.0]:
        add_summary_row(rows, "constant beta", f"beta={beta:g}", beta_results[beta])

    speed_specs = {
        "Fast defence": (30, 70),
        "Balanced": (50, 50),
        "Fast attack": (70, 30),
    }
    speed_results = {}
    for label, (attack_speed, defense_speed) in speed_specs.items():
        speed_results[label] = run_repeated(
            cfg,
            args.replications,
            20_000 + attack_speed * 100 + defense_speed,
            pitch_init=set_outfield_speed(cfg.pitch, attack_speed, defense_speed),
        )
        add_summary_row(rows, "speed asymmetry", label, speed_results[label])

    grid_results = {}
    grid_n = max(25, args.replications // 4)
    for attack_speed in [30, 50, 70]:
        for defense_speed in [30, 50, 70]:
            grid_results[(attack_speed, defense_speed)] = run_repeated(
                cfg,
                grid_n,
                30_000 + attack_speed * 100 + defense_speed,
                pitch_init=set_outfield_speed(cfg.pitch, attack_speed, defense_speed),
            )
    plot_speed_figures(speed_results, grid_results, out_dir)

    compact_specs = {
        "Loose spacing": DefenseParams(cfg.model.epsilon, min_horizontal_gap=70, compression_threshold=1100, compression_factor=1.0),
        "Moderate": DefenseParams(cfg.model.epsilon, min_horizontal_gap=50, compression_threshold=1100, compression_factor=0.95),
        "Tight compression": DefenseParams(cfg.model.epsilon, min_horizontal_gap=30, compression_threshold=1100, compression_factor=0.85),
    }
    compact_results = {}
    for label, defense in compact_specs.items():
        compact_results[label] = run_repeated(
            cfg,
            args.replications,
            40_000 + int(defense.min_horizontal_gap),
            defense=defense,
        )
        add_summary_row(rows, "compactness", label, compact_results[label])
    plot_compactness_figure(compact_results, out_dir)

    policy_specs = {
        r"constant $\beta$": None,
        r"piecewise $\beta(t)$": PiecewiseStepBeta([100_000.0, 50_000.0, 10_000.0], [60, 140]),
        r"sampled $\beta(t)$": DiscreteBetaSampler([10_000.0, 50_000.0, 100_000.0], [0.25, 0.50, 0.25]),
    }
    policy_results = {}
    for idx, (label, policy) in enumerate(policy_specs.items()):
        policy_results[label] = run_repeated(
            cfg,
            args.replications,
            50_000 + idx * 1_000,
            beta_policy=policy,
        )
        add_summary_row(rows, "beta policy", label.replace("$", ""), policy_results[label])
    plot_policy_figure(policy_results, out_dir)

    technical_rows: list[dict[str, str | float]] = []
    for idx, noise_sd_m in enumerate([0.0, 1.0, 2.0, 3.0]):
        noise_results = [
            run_endpoint_noise_simulation(cfg, 70_000 + idx * 1_000 + i, noise_sd_m)
            for i in range(args.replications)
        ]
        technical_rows.append(
            {
                "endpoint_noise_sd_m": noise_sd_m,
                **summarize_dict_rows(noise_results),
            }
        )
    write_technical_error_csv(technical_rows, out_dir)

    plot_sequence_figure(cfg, out_dir)
    write_summary_csv(rows, out_dir)
    print(f"Saved revision figures and summaries to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
