# FootballModelling

**Metropolis–Hastings simulation of football attacking combinations (xG-driven passing)**

This repository provides an **interpretable Monte Carlo simulator** for association football attacks based on the **Metropolis–Hastings (MH)** acceptance rule. We model an attacking possession as a stochastic search over team configurations: candidate passes are proposed and then **accepted/rejected** depending on how they change the **expected goals (xG)** of a potential shot, subject to defensive constraints (e.g., interceptions and positioning).

The code accompanies the paper:

> **“Metropolis algorithm for simulating football combinations” (2025)**  
> Dmitrii Evtyukhov, Daniil Fedotov

---

## Why this project

Football tactics often balance between:
- **patient possession** (safe passes, gradual progress), and
- **direct / vertical play** (riskier actions to create higher-quality chances).

Many data-driven models are powerful but hard to interpret; simplified models can be interpretable but unrealistic. This project aims for a **clean, configurable, and reproducible** middle ground: a compact simulator that captures key tactical trade-offs through a small set of parameters.

---

## Model overview

### State
The attacking team is represented by the 2D positions of the **10 outfield players** (the goalkeeper can be excluded from the state). A simulation step corresponds to a candidate pass (state transition).

### Defense and feasibility
A pass is considered **invalid / intercepted** if its straight-line trajectory violates defensive constraints (e.g., crosses a defender’s interception neighborhood). The codebase also supports defense updates between steps (e.g., “pressing” adjustments).

### Metropolis–Hastings acceptance
Given a current state `S` and a proposed next state `S_new`, define:

- `δ = xG(S) − xG(S_new)`
- `p = min(1, exp(−β · δ))`

Interpretation:
- If the proposal improves xG (`xG(S_new) ≥ xG(S)`), it is accepted.
- If it worsens xG, it can still be accepted with probability controlled by **β**.

`β` acts as a **risk-aversion / temperature** parameter:
- higher `β` → more conservative (rarely accepts worse moves),
- lower `β` → more exploratory (sometimes accepts worse moves).

---

## Key findings (paper summary)

Empirical Monte Carlo experiments in the paper emphasize:
1. **A sweet spot for β**: both overly conservative and overly exploratory regimes can reduce performance; an intermediate β often balances progress vs. safety.
2. **Speed asymmetry matters**: faster attackers (or slower defenders) tend to increase final xG; faster defense suppresses it.
3. **Defensive compactness has diminishing returns**: increased compactness raises interception rates, but improvements taper beyond basic organization.

---

## Repository structure

```text
.
├── backgrounds/               # pitch background images / assets
├── configs/                   # JSON configs (simulation + experiments)
├── scripts/                   # CLI entrypoints
├── src/footballmodelling/     # library code (simulation, xG, defense, artifacts, etc.)
├── artifacts/                 # generated outputs (gitignored)
├── pyproject.toml
└── requirements.txt
```

All generated outputs (GIFs, heatmaps, CSVs, logs) are written under `artifacts/` so the repo stays clean.

---

## Installation

```bash
git clone https://github.com/slikyumsh/FootballModelling.git
cd FootballModelling

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install -e .
```

> `pip install -e .` (editable install) is recommended so `footballmodelling` imports work from anywhere.

---

## Configuration-driven runs

This project is designed so that **initial player positions, player speeds, ball holder, and model parameters** come from JSON config files in `configs/`.

### Single simulation (saves artifacts)

```bash
python scripts/run_simulation.py --config configs/default.simulation.json
```

Typical outputs:

- `artifacts/runs/<run_id>/player_positions.gif`
- `artifacts/runs/<run_id>/ball_focus_heatmap.png`

### Parameter sweep experiment (exports CSV)

```bash
python scripts/run_experiment.py --config configs/default.experiment.json
```

Typical outputs:

- `artifacts/experiments/<experiment_id>.csv`

---

## Linting (ruff)

```bash
ruff check .
```

Optional auto-fix:

```bash
ruff check . --fix
```

---

## Citation

If you use this repository in academic work, please cite the accompanying paper.

### BibTeX

```bibtex
@article{evtyukhov2025metropolis,
  title   = {Metropolis algorithm for simulating football combinations},
  author  = {Evtyukhov, Dmitrii and Fedotov, Daniil},
  year    = {2025}
}
```

### Suggested text citation

Evtyukhov, D., & Fedotov, D. (2025). *Metropolis algorithm for simulating football combinations*.

---

## License

Add a LICENSE file if you plan to distribute or reuse this code broadly (MIT or Apache-2.0 are common choices for research code).
