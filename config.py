"""
Configuration constants for the World Cup predictor.
All magic numbers extracted into named constants with documentation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ═══════════════════════════════════════════════════════════════════════════
# TEAM NAME NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════

# Maps historical/defunct teams to modern FIFA-recognized successors.
# Per FIFA convention: Czech Republic inherits Czechoslovakia's records,
# Russia inherits Soviet Union's records, Serbia inherits Yugoslavia's,
# Germany inherits both West and East Germany's.
TEAM_ALIASES: Dict[str, str] = {
    'Czechoslovakia': 'Czech Republic',
    'West Germany': 'Germany',
    'East Germany': 'Germany',
    'Soviet Union': 'Russia',
    'Yugoslavia': 'Serbia',
    'Serbia and Montenegro': 'Serbia',
}

# ═══════════════════════════════════════════════════════════════════════════
# HOST NATIONS (per tournament year)
# ═══════════════════════════════════════════════════════════════════════════

HOSTS_BY_YEAR: Dict[int, List[str]] = {
    2026: ['Mexico', 'United States', 'Canada'],
}

# ═══════════════════════════════════════════════════════════════════════════
# RECENCY WEIGHTING
# ═══════════════════════════════════════════════════════════════════════════

# Exponential decay rate for historical match weighting.
# weight = exp(-DECAY_RATE * (current_year - match_year))
# At 0.05: a 1970 match gets ~7% weight of a 2022 match.
RECENCY_DECAY_RATE: float = 0.05

# Window (years) for "recent form" calculation — unweighted win rate.
RECENT_FORM_WINDOW_YEARS: int = 5

# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL TRAIN/TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════

# Year cutoff for honest temporal split.
# Matches BEFORE this year used for training, matches AT/AFTER used for testing.
# This eliminates data leakage from future matches into feature computation.
TEMPORAL_SPLIT_YEAR: int = 2018

# ═══════════════════════════════════════════════════════════════════════════
# MODEL HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

# Random Forest
RF_N_ESTIMATORS: int = 100
RF_RANDOM_STATE: int = 42

# Logistic Regression
LR_MAX_ITER: int = 2000
LR_RANDOM_STATE: int = 42

# ═══════════════════════════════════════════════════════════════════════════
# POISSON MODEL
# ═══════════════════════════════════════════════════════════════════════════

# Maximum goals considered in probability matrix (0..MAX_GOALS).
POISSON_MAX_GOALS: int = 8

# Floor for expected goals (lambda) to prevent zero-probability edge cases.
POISSON_LAMBDA_FLOOR_HOME: float = 0.15
POISSON_LAMBDA_FLOOR_AWAY: float = 0.10

# Floor for attack/defense strength relative to league average.
# Prevents division-by-zero issues for teams with no goals data.
POISSON_STRENGTH_FLOOR: float = 0.30

# Host advantage multiplier in Poisson expected goals.
# 1.20 = host gets 20% more expected goals than neutral.
POISSON_HOST_FACTOR: float = 1.20
POISSON_NEUTRAL_FACTOR: float = 1.05

# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════

# Poisson dominates ensemble because it models score-level interactions
# and is less susceptible to over-learning the ~27% historical draw rate.
# Raised to 70% because Poisson expected goals have been more accurate
# than classification models in back-testing.
ENSEMBLE_POISSON_WEIGHT: float = 0.70
ENSEMBLE_RF_WEIGHT: float = 0.12
ENSEMBLE_LR_WEIGHT: float = 0.18

# ═══════════════════════════════════════════════════════════════════════════
# STRENGTH DISPARITY / DRAW SUPPRESSION
# ═══════════════════════════════════════════════════════════════════════════

# Minimum strength gap (0-1) before draw suppression activates.
# Below this threshold, teams are considered evenly matched.
# Raised to reduce false positives from historical data biases.
DRAW_SUPPRESSION_THRESHOLD: float = 0.40

# Fraction of draw probability to suppress, proportional to strength gap.
# Reduced to avoid over-suppressing draws in competitive matches.
DRAW_SUPPRESSION_FACTOR: float = 0.35

# How suppressed draw probability is redistributed between stronger/weaker.
STRONGER_REDISTRIBUTION: float = 0.65
WEAKER_REDISTRIBUTION: float = 0.35

# ═══════════════════════════════════════════════════════════════════════════
# HOST OPENER BOOST
# ═══════════════════════════════════════════════════════════════════════════

# Host nation opening-match win probability boost (percentage points).
# Mexico at Azteca gets the maximum boost due to altitude + atmosphere.
HOST_OPENER_BOOST_AZTECA: float = 0.22
HOST_OPENER_BOOST_OTHER: float = 0.15

# How the boost is funded (taxed from draw and away probabilities).
HOST_BOOST_DRAW_TAX: float = 0.55
HOST_BOOST_AWAY_TAX: float = 0.45

# ═══════════════════════════════════════════════════════════════════════════
# UPSET METRICS
# ═══════════════════════════════════════════════════════════════════════════

# Composite score weights for determining favorite/underdog.
UPSET_WIN_RATE_WEIGHT: float = 100.0
UPSET_RECENT_FORM_WEIGHT: float = 80.0
UPSET_KNOCKOUT_WEIGHT: float = 40.0
UPSET_RANK_WEIGHT: float = 0.5

# Shock index components (max 100).
UPSET_RANK_SHOCK_MAX: float = 40.0
UPSET_FORM_SHOCK_MAX: float = 35.0
UPSET_PROB_SHOCK_MAX: float = 25.0
UPSET_RANK_GAP_CAP: int = 100

# Upset level labels based on underdog win probability.
UPSET_LEVELS: List[Tuple[float, str]] = [
    (0.15, 'Very Unlikely'),
    (0.25, 'Unlikely'),
    (0.35, 'Possible'),
    (0.45, 'Real Chance'),
]
UPSET_DEFAULT_LEVEL: str = 'Not Really An Upset'

# ═══════════════════════════════════════════════════════════════════════════
# DEFAULT FALLBACK VALUES
# ═══════════════════════════════════════════════════════════════════════════

# Default stats for teams not found in historical data.
DEFAULT_WIN_RATE: float = 0.50
DEFAULT_RECENT_FORM: float = 0.50
DEFAULT_GOALS_PER_GAME: float = 1.20
DEFAULT_GOALS_CONCEDED: float = 1.20
DEFAULT_FIFA_RANK: int = 999
DEFAULT_FIFA_POINTS: float = 0.0

# ═══════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════

DISPLAY_WIDTH: int = 72
PROB_BAR_WIDTH: int = 30
SCORE_BAR_WIDTH: int = 25

# ═══════════════════════════════════════════════════════════════════════════
# DATA PATHS
# ═══════════════════════════════════════════════════════════════════════════

MATCHES_CSV: str = 'matches_1930_2022.csv'
FIFA_RANKING_CSV: str = 'data/fifa_ranking_2022-10-06.csv'
