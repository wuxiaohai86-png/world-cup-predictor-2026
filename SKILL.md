---
name: world-cup-predictor
description: "Predict FIFA World Cup 2026 match outcomes using ML (RF + LR + Poisson ensemble). Features: W/D/L, exact scores, upset probability, group standings projection, data analysis reports. TRIGGER when: user asks about World Cup predictions, match forecasting, group standings, team analysis."
---

# World Cup Predictor 2026

Comprehensive FIFA World Cup match prediction system with 4 analysis dimensions.

## Project Location

```
~/world-cup-predictor-2026/
├── config.py                  # All constants & hyperparameters
├── data.py                    # Data loading, team stats, H2H, xG, manager features
├── models.py                  # RF/LR/Poisson models + ensemble calibration
├── analysis.py                # Upset metrics + data report generator
├── group_projection.py        # Group stage simulation
├── predict_match.py           # CLI: single match + --group flag
├── world_cup.py               # Original training script (kept for reference)
├── matches_1930_2022.csv      # 964 World Cup matches (1930-2022)
├── data/
│   ├── fifa_ranking_2022-10-06.csv
│   ├── world_cup.csv
│   └── tournament_2026.json   # 48 teams, 12 groups, qualifying data, injuries
├── correlation_heatmap.png
├── feature_importance.png
└── README.md
```

## Capabilities

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Win/Draw/Loss** | RF + LR + Poisson (70%) weighted ensemble with draw suppression & host boost |
| 2 | **Exact Score** | Poisson distribution model, top-12 scorelines, expected goals, Over/Under, BTTS |
| 3 | **Upset Analysis** | Upset probability, shock index (0-100), composite strength gap |
| 4 | **Data Report** | Team profiles, decade trends, H2H history, radar comparison (6 dimensions) |
| 5 | **Group Projection** | Predict all 6 matches in any group, simulated standings with tiebreakers |
| 6 | **Honest Accuracy** | Temporal-split evaluation (train <2018, test >=2018) via --accuracy flag |

## Features Used (18 total)

| # | Feature | Source |
|---|---------|--------|
| 1-2 | Team identity (encoded) | LabelEncoder |
| 3 | Year | Match data |
| 4-5 | Win rate (weighted) | Historical, exp decay |
| 6-7 | Goals per game | Historical, weighted |
| 8-9 | Goals conceded/game | Historical, weighted |
| 10-11 | Recent form (5yr) | Last 5 years unweighted |
| 12 | H2H advantage | Head-to-head record |
| 13-14 | Host status | Binary flag |
| 15-16 | Manager win rate | Historical manager record |
| 17-18 | xG difference | Expected goals (2018+) |

## How to Run

```bash
cd ~/world-cup-predictor-2026

# Single match prediction
python predict_match.py "Mexico" "South Africa" 2026

# With honest accuracy report
python predict_match.py "Brazil" "Germany" 2026 --accuracy

# Show full score probability matrix
python predict_match.py "France" "Argentina" 2026 --all-scores

# Group stage projection
python predict_match.py --group A

# Run all 12 groups
python predict_match.py --group all
```

Dependencies: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`

## Model Calibration

- **Poisson-dominant ensemble**: 70% Poisson, 12% RF, 18% LR
- **Draw suppression**: When strength gap >40%, reduce draw probability
- **Poisson-aware**: If Poisson disagrees with historical stats on favorite, discount gap by 65%
- **Host opener boost**: +22pp (Azteca), +15pp (other hosts)
- **Qualifying bonus**: Confederation strength + qualifying performance factored into group projections

## Subcommands

| Command | Action |
|---------|--------|
| `--group A` | Project Group A standings |
| `--accuracy` | Show temporal-split honest accuracy |
| `--all-scores` | Show full score probability matrix |

## 2026 Tournament Data

- 48 teams, 12 groups of 4
- Qualifying performance data in `data/tournament_2026.json`
- Key injuries tracked (e.g., Davies for Canada)
- Confederation strength factors applied
- Top 2 + 8 best 3rd-place teams advance to Round of 32
