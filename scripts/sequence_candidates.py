#!/usr/bin/env python
"""Search for alternative illustrative passing-sequence figures."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

from footballmodelling.config import load_simulation_config
import revision_experiments as rev


PITCH_LENGTH = 1500.0
PITCH_WIDTH = 1000.0
PENALTY_X = PITCH_LENGTH - PITCH_LENGTH * 16.5 / 105.0
PENALTY_SPOT_X = PITCH_LENGTH - PITCH_LENGTH * 11.0 / 105.0
PENALTY_HALF_WIDTH = PITCH_WIDTH * 40.32 / 68.0 / 2.0
SIX_YARD_X = PITCH_LENGTH - PITCH_LENGTH * 5.5 / 105.0
SIX_YARD_HALF_WIDTH = PITCH_WIDTH * 18.32 / 68.0 / 2.0


def visible_points(pass_events, carrier_moves):
    points = []
    for event in pass_events:
        points.extend([event["start"], event["end"]])
    for start, end, _player_id in carrier_moves:
        points.extend([start, end])
    return points


def metrics_for(seed, result, pass_events, carrier_moves):
    final = pass_events[-1]["end"]
    points = visible_points(pass_events, carrier_moves)
    pre_points = points[:-1]
    max_x = max(point[0] for point in points)
    pre_max_x = max(point[0] for point in pre_points) if pre_points else final[0]
    final_y_offset = abs(final[1] - 500.0)
    overshoot = max(0.0, pre_max_x - final[0])
    score = (
        abs(final[0] - (PENALTY_X + PENALTY_SPOT_X) / 2.0) / 90.0
        + final_y_offset / 220.0
        + overshoot / 35.0
        + max(0.0, max_x - 1365.0) / 30.0
        + abs(len(pass_events) - 7) / 5.0
    )
    return {
        "seed": seed,
        "score": score,
        "passes_shown": len(pass_events),
        "simulation_passes": result.passes,
        "final_xg": result.final_xg,
        "final_x": final[0],
        "final_y": final[1],
        "max_visible_x": max_x,
        "pre_final_max_x": pre_max_x,
        "pre_final_overshoot": overshoot,
    }


def is_good_candidate(metrics):
    if metrics["passes_shown"] < 5:
        return False
    if not (PENALTY_X - 35.0 <= metrics["final_x"] <= PENALTY_SPOT_X + 35.0):
        return False
    if not (500.0 - PENALTY_HALF_WIDTH - 35.0 <= metrics["final_y"] <= 500.0 + PENALTY_HALF_WIDTH + 35.0):
        return False
    if metrics["max_visible_x"] > PENALTY_SPOT_X + 55.0:
        return False
    if metrics["pre_final_overshoot"] > 70.0:
        return False
    return True


def select_diverse(candidates, limit):
    selected = []
    for metrics, payload in sorted(candidates, key=lambda item: item[0]["score"]):
        final = (metrics["final_x"], metrics["final_y"])
        if all(math.hypot(final[0] - old["final_x"], final[1] - old["final_y"]) > 45 for old, _ in selected):
            selected.append((metrics, payload))
        if len(selected) == limit:
            break
    return selected


def draw_candidate(cfg, sim, pass_events, carrier_moves, metrics, out_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.set_facecolor("#eff5ec")
    ax.plot([0, PITCH_LENGTH, PITCH_LENGTH, 0, 0], [0, 0, PITCH_WIDTH, PITCH_WIDTH, 0], color="#57745b", linewidth=1.2)
    ax.axhline(500, color="#9fb29d", linewidth=0.8, linestyle="--")
    ax.axvline(750, color="#9fb29d", linewidth=0.8, linestyle="--")
    ax.add_patch(
        Rectangle(
            (PENALTY_X, 500 - PENALTY_HALF_WIDTH),
            PITCH_LENGTH - PENALTY_X,
            2 * PENALTY_HALF_WIDTH,
            fill=False,
            edgecolor="#57745b",
            linewidth=1.0,
        )
    )
    ax.add_patch(
        Rectangle(
            (SIX_YARD_X, 500 - SIX_YARD_HALF_WIDTH),
            PITCH_LENGTH - SIX_YARD_X,
            2 * SIX_YARD_HALF_WIDTH,
            fill=False,
            edgecolor="#779077",
            linewidth=0.8,
        )
    )
    ax.scatter([PENALTY_SPOT_X], [500], s=12, color="#57745b", zorder=2)

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
        ax.text(
            end[0],
            end[1],
            str(order),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "circle,pad=0.18", "fc": "#111111", "ec": "none", "alpha": 0.9},
        )

    ax.set_xlim(0, PITCH_LENGTH)
    ax.set_ylim(PITCH_WIDTH, 0)
    ax.set_xlabel("Attacking depth x")
    ax.set_ylabel("Lateral coordinate y")
    pass_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", color="#111111", linewidth=1.45)
    move_handle = FancyArrowPatch((0, 0), (1, 0), arrowstyle="-|>", color="#6f7782", linewidth=1.0, linestyle=(0, (3, 3)))
    ax.legend([pass_handle, move_handle], ["accepted pass", "ball-carrier movement"], loc="lower left", frameon=False, fontsize=8)
    ax.set_title(
        f"Candidate seed {metrics['seed']} | final x={metrics['final_x']:.0f}, max x={metrics['max_visible_x']:.0f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/beta.simulation.json"))
    parser.add_argument("--out-dir", default=str(ROOT / "Metropolis_article_new_1804_1_not_anon/sequence_candidates"))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-stop", type=int, default=120_000)
    parser.add_argument("--beta", type=float, default=1_000_000.0)
    parser.add_argument("--shown-passes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--pool-size", type=int, default=120)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    cfg = load_simulation_config(args.config)
    rev.MAX_STEPS = args.max_steps
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    seed_iter = args.seeds if args.seeds else range(args.seed_start, args.seed_stop)
    for seed in seed_iter:
        result, sim, pass_events, carrier_moves = rev.run_sequence_trace(
            cfg,
            seed=seed,
            beta=args.beta,
            shown_passes=args.shown_passes,
        )
        if not pass_events:
            continue
        metrics = metrics_for(seed, result, pass_events, carrier_moves)
        if args.seeds or is_good_candidate(metrics):
            candidates.append((metrics, (sim, pass_events, carrier_moves)))
        if not args.seeds and len(candidates) >= args.pool_size:
            break

    selected = candidates[: args.limit] if args.seeds else select_diverse(candidates, args.limit)
    if len(selected) < args.limit:
        print(f"Only found {len(selected)} candidates; try increasing --seed-stop or passing --seeds.")

    csv_path = out_dir / "sequence_candidates.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "seed",
            "score",
            "passes_shown",
            "simulation_passes",
            "final_xg",
            "final_x",
            "final_y",
            "max_visible_x",
            "pre_final_max_x",
            "pre_final_overshoot",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, (metrics, payload) in enumerate(selected, start=1):
            label = chr(ord("A") + idx - 1)
            out_base = out_dir / f"fig_sequence_candidate_{label}"
            sim, pass_events, carrier_moves = payload
            draw_candidate(cfg, sim, pass_events, carrier_moves, metrics, out_base)
            writer.writerow({"label": label, **metrics})
            print(
                f"{label}: seed={metrics['seed']} final=({metrics['final_x']:.1f}, {metrics['final_y']:.1f}) "
                f"max_x={metrics['max_visible_x']:.1f} pre_max={metrics['pre_final_max_x']:.1f} "
                f"passes={metrics['passes_shown']}"
            )
    print(f"Saved candidates to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
