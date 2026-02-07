# FootballModelling
**Metropolis–Hastings simulation of football attacking combinations (xG-driven passing)**

This repository contains an interpretable simulation framework for association football attacks based on the **Metropolis–Hastings** algorithm. We model an attacking sequence as a stochastic search over team configurations, where passes are accepted/rejected according to how they change the **expected goals (xG)** of a potential shot, under defensive constraints (interceptions + positioning). :contentReference[oaicite:1]{index=1}

The project accompanies the paper:

**“Metropolis algorithm for simulating football combinations” (2025)**  
Dmitrii Evtyukhov, Daniil Fedotov. :contentReference[oaicite:2]{index=2}

---

## Motivation
Football tactics often oscillate between **patient possession** and **rapid transitions/counterattacks**. While modern analytics offers powerful data-driven tools, they can be hard to interpret, and simplified toy models often miss key tactical trade-offs.

Our goal is a **minimalist but meaningful** simulator that:
- reproduces qualitative patterns familiar to coaches and analysts,
- remains transparent and controllable,
- enables controlled experiments on tactical factors (risk appetite, relative speed, defensive compactness). :contentReference[oaicite:3]{index=3}

---

## How it works (high-level)
### State
The attacking team is represented by the **(x, y) positions of 10 outfield players** — a 20-dimensional state vector. :contentReference[oaicite:4]{index=4}

### Defense
Each defender has an **interception radius ε**: any pass whose straight-line trajectory crosses a defender’s ε-neighborhood is automatically intercepted. The base model uses static defenders; the extended model adds dynamic defensive adjustments (shifting, pressure, role-dependent positioning). :contentReference[oaicite:5]{index=5}

### Metropolis acceptance rule
At each step we propose a candidate pass to a new state `S_new`. If it is not intercepted, we compute:

- `δ = xG(S) − xG(S_new)`
- `p = min(1, exp(−β · δ))`

Passes that **increase xG** are always accepted; passes that **decrease xG** may still be accepted depending on **β**, which acts as a *risk-aversion / “temperature”* parameter. :contentReference[oaicite:6]{index=6}

---

## Key findings (from the paper)
Our Monte Carlo experiments highlight three main insights:

1. **An optimal β exists**: both overly conservative (“cold”) and overly exploratory (“hot”) passing perform worse than an intermediate “sweet spot”, balancing progress and safety. :contentReference[oaicite:7]{index=7}  
2. **Relative speed is decisive**: faster attacking play markedly increases final xG, while faster defense suppresses it. :contentReference[oaicite:8]{index=8}  
3. **Defensive compactness shows diminishing returns**: tighter organization increases interception rates, but gains taper off beyond basic compactness. :contentReference[oaicite:9]{index=9}  

---

## Repository contents (current)
Typical important files/folders:
- `modelling.ipynb` — main research notebook (experiments + plots)
- `class_second_version.py` — core simulation logic (state, passing, defense)
- `requirements.txt` — dependencies
- `backgrounds/` — visuals/background assets
- `with_players.jpg`, `ball_focus_heatmap_with_ball_focus.png` — current artifacts

> Note: this code started as research code; the next steps are to modularize and make results reproducible via scripts/configs.

---

## Installation
```bash
git clone https://github.com/slikyumsh/FootballModelling.git
cd FootballModelling

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
