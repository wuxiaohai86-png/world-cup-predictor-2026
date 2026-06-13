"""
Win/Draw/Loss classification models, Poisson score prediction, and ensemble
calibration. All models use temporal splitting for honest accuracy evaluation.
"""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

from config import (
    RF_N_ESTIMATORS, RF_RANDOM_STATE, LR_MAX_ITER, LR_RANDOM_STATE,
    POISSON_MAX_GOALS, POISSON_LAMBDA_FLOOR_HOME, POISSON_LAMBDA_FLOOR_AWAY,
    POISSON_STRENGTH_FLOOR, POISSON_HOST_FACTOR, POISSON_NEUTRAL_FACTOR,
    ENSEMBLE_POISSON_WEIGHT, ENSEMBLE_RF_WEIGHT, ENSEMBLE_LR_WEIGHT,
    DRAW_SUPPRESSION_THRESHOLD, DRAW_SUPPRESSION_FACTOR,
    STRONGER_REDISTRIBUTION, WEAKER_REDISTRIBUTION,
    HOST_OPENER_BOOST_AZTECA, HOST_OPENER_BOOST_OTHER,
    HOST_BOOST_DRAW_TAX, HOST_BOOST_AWAY_TAX,
    TEMPORAL_SPLIT_YEAR, DEFAULT_WIN_RATE, DEFAULT_RECENT_FORM,
    DEFAULT_GOALS_PER_GAME, DEFAULT_GOALS_CONCEDED,
)
from data import (
    get_h2h_advantage, get_hosts_for_year,
    calculate_team_stats, calculate_head_to_head,
    calculate_manager_stats, get_manager_advantage,
    calculate_xg_stats, temporal_split,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FEATURE ENGINEERING                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

FEATURE_NAMES = [
    'home_team_encoded', 'away_team_encoded', 'Year',
    'home_win_rate', 'away_win_rate',
    'home_goals_per_game', 'away_goals_per_game',
    'home_goals_conceded', 'away_goals_conceded',
    'home_recent_form', 'away_recent_form',
    'h2h_advantage', 'home_is_host', 'away_is_host',
    # NEW: manager and xG features
    'home_mgr_winrate', 'away_mgr_winrate',
    'home_xg_diff', 'away_xg_diff',
]


def _is_host(team: str, host: Any, year: int) -> int:
    """Check if team was a host nation."""
    if pd.isna(host):
        return 0
    hosts = [h.strip() for h in str(host).split('/')]
    return 1 if team in hosts else 0


def build_feature_matrix(
    df: pd.DataFrame,
    team_stats: Dict[str, Dict[str, Any]],
    h2h_stats: Dict,
    manager_stats: Dict[str, Dict[str, float]],
    xg_stats: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Build feature matrix from match data and computed statistics.
    Returns (X, y, le_home, le_away).
    """
    df = df.copy()

    # Encode team names
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    df['home_team_encoded'] = le_home.fit_transform(df['home_team'])
    df['away_team_encoded'] = le_away.fit_transform(df['away_team'])

    # Map core team stats
    df['home_win_rate'] = df['home_team'].map(
        lambda x: team_stats.get(x, {}).get('win_rate', DEFAULT_WIN_RATE))
    df['away_win_rate'] = df['away_team'].map(
        lambda x: team_stats.get(x, {}).get('win_rate', DEFAULT_WIN_RATE))
    df['home_goals_per_game'] = df['home_team'].map(
        lambda x: team_stats.get(x, {}).get('goals_per_game', DEFAULT_GOALS_PER_GAME))
    df['away_goals_per_game'] = df['away_team'].map(
        lambda x: team_stats.get(x, {}).get('goals_per_game', DEFAULT_GOALS_PER_GAME))
    df['home_goals_conceded'] = df['home_team'].map(
        lambda x: team_stats.get(x, {}).get('goals_conceded_per_game', DEFAULT_GOALS_CONCEDED))
    df['away_goals_conceded'] = df['away_team'].map(
        lambda x: team_stats.get(x, {}).get('goals_conceded_per_game', DEFAULT_GOALS_CONCEDED))
    df['home_recent_form'] = df['home_team'].map(
        lambda x: team_stats.get(x, {}).get('recent_form', DEFAULT_RECENT_FORM))
    df['away_recent_form'] = df['away_team'].map(
        lambda x: team_stats.get(x, {}).get('recent_form', DEFAULT_RECENT_FORM))

    # H2H
    df['h2h_advantage'] = df.apply(
        lambda r: get_h2h_advantage(r['home_team'], r['away_team'], h2h_stats), axis=1)

    # Host status
    df['home_is_host'] = df.apply(
        lambda r: _is_host(r['home_team'], r['Host'], r['Year']), axis=1)
    df['away_is_host'] = df.apply(
        lambda r: _is_host(r['away_team'], r['Host'], r['Year']), axis=1)

    # Manager features
    df['home_mgr_winrate'] = df.apply(
        lambda r: _get_mgr_rate(r['home_team'], r['home_manager'], manager_stats), axis=1)
    df['away_mgr_winrate'] = df.apply(
        lambda r: _get_mgr_rate(r['away_team'], r['away_manager'], manager_stats), axis=1)

    # xG features (0 if no xG data available for that team)
    df['home_xg_diff'] = df['home_team'].map(
        lambda x: xg_stats.get(x, {}).get('xg_diff', 0.0))
    df['away_xg_diff'] = df['away_team'].map(
        lambda x: xg_stats.get(x, {}).get('xg_diff', 0.0))

    X = df[FEATURE_NAMES]
    y = df['winner']

    return X, y, le_home, le_away


def _get_mgr_rate(team: str, mgr: Any,
                  manager_stats: Dict[str, Dict[str, float]]) -> float:
    """Get manager win rate for a team's manager. Default 0.5 if unknown."""
    if pd.isna(mgr) or mgr not in manager_stats:
        return 0.5
    return manager_stats[mgr]['win_rate']


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  WDL MODEL BUILDER (with temporal split)                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_wdl_models_temporal(
    df: pd.DataFrame,
    split_year: int = TEMPORAL_SPLIT_YEAR,
) -> Dict[str, Any]:
    """
    Train WDL models with honest temporal split.
    - Stats computed ONLY from pre-split data (no leakage).
    - Models trained on pre-split data.
    - Accuracy evaluated on post-split data.
    """
    train_df, test_df = temporal_split(df, split_year)

    # Edge case: if split_year is too early (or too late), use all data
    if len(train_df) == 0:
        train_df = df.copy()
        test_df = pd.DataFrame()
    if len(test_df) == 0 and len(train_df) > 100:
        # Create a minimal test set from the last ~20% of data
        cut = int(len(train_df) * 0.8)
        test_df = train_df.iloc[cut:].copy()
        train_df = train_df.iloc[:cut].copy()

    # Compute all stats from training data ONLY
    team_stats = calculate_team_stats(train_df, current_year=split_year)
    h2h_stats = calculate_head_to_head(train_df)
    manager_stats = calculate_manager_stats(train_df)
    xg_stats = calculate_xg_stats(train_df)

    # Build features
    X_train, y_train, le_home, le_away = build_feature_matrix(
        train_df, team_stats, h2h_stats, manager_stats, xg_stats)

    # Train models
    rf = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                random_state=RF_RANDOM_STATE)
    rf.fit(X_train, y_train)

    # Scale features for logistic regression (helps convergence)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=LR_MAX_ITER, random_state=LR_RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)

    # ── Honest evaluation on test set ──
    eval_result = None
    if len(test_df) > 0:
        X_test, y_test, _, _ = build_feature_matrix(
            test_df, team_stats, h2h_stats, manager_stats, xg_stats)

        # Re-fit label encoders on test-set teams (some may be unseen)
        le_home_all = LabelEncoder()
        le_away_all = LabelEncoder()
        le_home_all.fit(pd.concat([train_df['home_team'], test_df['home_team']]))
        le_away_all.fit(pd.concat([train_df['away_team'], test_df['away_team']]))

        # Handle unseen teams in test set
        X_test['home_team_encoded'] = _safe_encode(
            test_df['home_team'], le_home_all, le_home)
        X_test['away_team_encoded'] = _safe_encode(
            test_df['away_team'], le_away_all, le_away)

        X_test_scaled = scaler.transform(X_test)

        rf_pred = rf.predict(X_test)
        lr_pred = lr.predict(X_test_scaled)

        eval_result = {
            'test_matches': len(test_df),
            'rf_accuracy': accuracy_score(y_test, rf_pred),
            'lr_accuracy': accuracy_score(y_test, lr_pred),
            'rf_f1_macro': f1_score(y_test, rf_pred, average='macro'),
            'lr_f1_macro': f1_score(y_test, lr_pred, average='macro'),
            'rf_f1_weighted': f1_score(y_test, rf_pred, average='weighted'),
            'lr_f1_weighted': f1_score(y_test, lr_pred, average='weighted'),
        }

    return {
        'rf': rf,
        'lr': lr,
        'scaler': scaler,
        'le_home': le_home,
        'le_away': le_away,
        'team_stats': team_stats,
        'h2h_stats': h2h_stats,
        'manager_stats': manager_stats,
        'xg_stats': xg_stats,
        'feature_names': FEATURE_NAMES,
        'eval': eval_result,
    }


def _safe_encode(series: pd.Series, full_encoder: LabelEncoder,
                 train_encoder: LabelEncoder) -> pd.Series:
    """Encode team names, using -1 for teams unseen in training."""
    result = []
    train_classes = set(train_encoder.classes_)
    for val in series:
        if val in train_classes:
            result.append(train_encoder.transform([val])[0])
        else:
            result.append(-1)
    return pd.Series(result, index=series.index)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  POISSON SCORE PREDICTION                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def calculate_poisson_lambdas(
    home_team: str,
    away_team: str,
    team_stats: Dict[str, Dict[str, Any]],
    df: pd.DataFrame,
    year: int = 2026,
) -> Tuple[float, float]:
    """
    Calculate expected goals (lambda) for each team.
    Uses team attack/defense strength relative to league average.
    """
    hs = team_stats.get(home_team, {})
    aws = team_stats.get(away_team, {})

    home_gpg = hs.get('goals_per_game', DEFAULT_GOALS_PER_GAME)
    home_gcpg = hs.get('goals_conceded_per_game', DEFAULT_GOALS_CONCEDED)
    away_gpg = aws.get('goals_per_game', DEFAULT_GOALS_PER_GAME)
    away_gcpg = aws.get('goals_conceded_per_game', DEFAULT_GOALS_CONCEDED)

    # League averages
    all_scores = df['home_score'].tolist() + df['away_score'].tolist()
    league_avg_goal = np.mean(all_scores) if all_scores else 1.5
    league_avg_home = df['home_score'].mean()
    league_avg_away = df['away_score'].mean()

    # Attack and defense strengths
    home_attack = max(home_gpg / league_avg_goal, POISSON_STRENGTH_FLOOR) if league_avg_goal > 0 else 1.0
    home_defense = max(home_gcpg / league_avg_goal, POISSON_STRENGTH_FLOOR) if league_avg_goal > 0 else 1.0
    away_attack = max(away_gpg / league_avg_goal, POISSON_STRENGTH_FLOOR) if league_avg_goal > 0 else 1.0
    away_defense = max(away_gcpg / league_avg_goal, POISSON_STRENGTH_FLOOR) if league_avg_goal > 0 else 1.0

    # Host advantage
    hosts = get_hosts_for_year(year)
    host_factor = POISSON_HOST_FACTOR if home_team in hosts else POISSON_NEUTRAL_FACTOR

    home_lambda = home_attack * away_defense * league_avg_home * host_factor
    away_lambda = away_attack * home_defense * league_avg_away

    home_lambda = max(home_lambda, POISSON_LAMBDA_FLOOR_HOME)
    away_lambda = max(away_lambda, POISSON_LAMBDA_FLOOR_AWAY)

    return home_lambda, away_lambda


def score_probability_matrix(
    home_lambda: float,
    away_lambda: float,
    max_goals: int = POISSON_MAX_GOALS,
) -> Tuple[Dict[Tuple[int, int], float], Dict[str, float]]:
    """
    Calculate probability for each exact score using Poisson distribution.
    Returns (score_probs, wdl_probs).
    """
    score_probs: Dict[Tuple[int, int], float] = {}
    total_prob = 0.0

    for i in range(max_goals + 1):
        p_home = poisson.pmf(i, home_lambda)
        for j in range(max_goals + 1):
            p_away = poisson.pmf(j, away_lambda)
            p_score = p_home * p_away
            score_probs[(i, j)] = p_score
            total_prob += p_score

    # Normalize
    score_probs = {k: v / total_prob for k, v in score_probs.items()}

    # Aggregate to W/D/L
    home_win = sum(p for (h, a), p in score_probs.items() if h > a)
    away_win = sum(p for (h, a), p in score_probs.items() if a > h)
    draw = sum(p for (h, a), p in score_probs.items() if h == a)

    return score_probs, {'home': home_win, 'away': away_win, 'draw': draw}


def get_top_scorelines(
    score_probs: Dict[Tuple[int, int], float],
    n: int = 12,
) -> List[Tuple[Tuple[int, int], float]]:
    """Get the N most likely scorelines."""
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)
    return [(score, prob) for score, prob in sorted_scores[:n]]


def expected_score_from_matrix(
    score_probs: Dict[Tuple[int, int], float],
) -> Tuple[float, float]:
    """Calculate expected score from probability matrix."""
    exp_home = sum(h * p for (h, a), p in score_probs.items())
    exp_away = sum(a * p for (h, a), p in score_probs.items())
    return exp_home, exp_away


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MATCH PREDICTION                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def predict_wdl(
    models: Dict[str, Any],
    home_team: str,
    away_team: str,
    year: int,
    team_stats: Dict[str, Dict[str, Any]],
    h2h_stats: Dict,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Predict WDL for a specific match using trained models."""
    rf = models['rf']
    lr = models['lr']
    scaler = models['scaler']
    le_home = models['le_home']
    le_away = models['le_away']

    try:
        home_enc = le_home.transform([home_team])[0]
    except ValueError:
        home_enc = -1
    try:
        away_enc = le_away.transform([away_team])[0]
    except ValueError:
        away_enc = -1

    hs = team_stats.get(home_team, {})
    aws = team_stats.get(away_team, {})

    hosts = get_hosts_for_year(year)
    home_is_host = 1 if home_team in hosts else 0
    away_is_host = 1 if away_team in hosts else 0

    features = np.array([[
        home_enc, away_enc, year,
        hs.get('win_rate', DEFAULT_WIN_RATE),
        aws.get('win_rate', DEFAULT_WIN_RATE),
        hs.get('goals_per_game', DEFAULT_GOALS_PER_GAME),
        aws.get('goals_per_game', DEFAULT_GOALS_PER_GAME),
        hs.get('goals_conceded_per_game', DEFAULT_GOALS_CONCEDED),
        aws.get('goals_conceded_per_game', DEFAULT_GOALS_CONCEDED),
        hs.get('recent_form', DEFAULT_RECENT_FORM),
        aws.get('recent_form', DEFAULT_RECENT_FORM),
        get_h2h_advantage(home_team, away_team, h2h_stats),
        home_is_host, away_is_host,
        0.5, 0.5,  # manager defaults (not used in match prediction context)
        0.0, 0.0,  # xG defaults
    ]])

    rf_proba = rf.predict_proba(features)[0]
    features_scaled = scaler.transform(features)
    lr_proba = lr.predict_proba(features_scaled)[0]

    rf_result = {str(c): float(rf_proba[i]) for i, c in enumerate(rf.classes_)}
    lr_result = {str(c): float(lr_proba[i]) for i, c in enumerate(lr.classes_)}

    return rf_result, lr_result


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENSEMBLE CALIBRATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def calculate_strength_disparity(
    home_team: str,
    away_team: str,
    team_stats: Dict[str, Dict[str, Any]],
    rank_dict: Dict[str, int],
) -> Tuple[float, str, str]:
    """
    Calculate strength gap (0-1) and identify stronger/weaker team.
    """
    hs = team_stats.get(home_team, {})
    aws = team_stats.get(away_team, {})
    home_rank = rank_dict.get(home_team, 999)
    away_rank = rank_dict.get(away_team, 999)

    def composite_strength(stats: Dict[str, Any], rank: int) -> float:
        return (
            stats.get('win_rate', 0) * 40 +
            stats.get('recent_form', 0) * 30 +
            stats.get('goals_per_game', 1) * 10 -
            stats.get('goals_conceded_per_game', 1) * 12 +
            max(0, (100 - min(rank, 100))) * 0.3
        )

    home_str = composite_strength(hs, home_rank)
    away_str = composite_strength(aws, away_rank)

    gap = abs(home_str - away_str)
    normalized_gap = min(gap / 18.0, 1.0)

    stronger = home_team if home_str >= away_str else away_team
    weaker = away_team if home_str >= away_str else home_team

    return normalized_gap, stronger, weaker


def get_host_opener_boost(home_team: str, year: int) -> float:
    """Calculate host nation opening-match win probability boost."""
    hosts = get_hosts_for_year(year)
    if home_team not in hosts:
        return 0.0
    if home_team == 'Mexico':
        return HOST_OPENER_BOOST_AZTECA
    return HOST_OPENER_BOOST_OTHER


def calibrate_ensemble(
    rf_probs: Dict[str, float],
    lr_probs: Dict[str, float],
    poisson_wdl: Dict[str, float],
    home_team: str,
    away_team: str,
    year: int,
    team_stats: Dict[str, Dict[str, Any]],
    rank_dict: Dict[str, int],
) -> Dict[str, float]:
    """
    Build calibrated ensemble prediction.
    1. Weighted average (Poisson-dominant: 70%).
    2. Strength disparity draw suppression (Poisson-aware).
    3. Host nation opening match boost.
    """
    # Step 1: Weighted ensemble
    ensemble = {}
    for k in ['home', 'draw', 'away']:
        ensemble[k] = (
            rf_probs.get(k, 0) * ENSEMBLE_RF_WEIGHT +
            lr_probs.get(k, 0) * ENSEMBLE_LR_WEIGHT +
            poisson_wdl.get(k, 0) * ENSEMBLE_POISSON_WEIGHT
        )

    # Step 2: Poisson-aware strength disparity
    # If Poisson disagrees with historical-based strength about who is stronger,
    # reduce the gap (trust Poisson's forward-looking assessment more).
    hist_gap, hist_stronger, hist_weaker = calculate_strength_disparity(
        home_team, away_team, team_stats, rank_dict)

    # Who does Poisson think is stronger?
    poisson_home = poisson_wdl.get('home', 0)
    poisson_away = poisson_wdl.get('away', 0)
    poisson_stronger = home_team if poisson_home >= poisson_away else away_team

    # If Poisson disagrees with historical stats, halve the gap
    if poisson_stronger != hist_stronger:
        strength_gap = hist_gap * 0.35  # heavily discount historical gap
        stronger_team = poisson_stronger  # trust Poisson
        weaker_team = away_team if poisson_stronger == home_team else home_team
    else:
        strength_gap = hist_gap
        stronger_team = hist_stronger
        weaker_team = hist_weaker

    draw_suppression_applied = False
    if strength_gap > DRAW_SUPPRESSION_THRESHOLD:
        draw_suppress = ensemble['draw'] * strength_gap * DRAW_SUPPRESSION_FACTOR
        ensemble['draw'] -= draw_suppress
        if stronger_team == home_team:
            ensemble['home'] += draw_suppress * STRONGER_REDISTRIBUTION
            ensemble['away'] += draw_suppress * WEAKER_REDISTRIBUTION
        else:
            ensemble['away'] += draw_suppress * STRONGER_REDISTRIBUTION
            ensemble['home'] += draw_suppress * WEAKER_REDISTRIBUTION
        draw_suppression_applied = True

    # Step 3: Host opener boost
    host_boost = get_host_opener_boost(home_team, year)
    away_host_boost = get_host_opener_boost(away_team, year)
    host_boost_applied = host_boost > 0 or away_host_boost > 0

    if host_boost > 0:
        ensemble['home'] += host_boost
        ensemble['draw'] -= host_boost * HOST_BOOST_DRAW_TAX
        ensemble['away'] -= host_boost * HOST_BOOST_AWAY_TAX
    if away_host_boost > 0:
        ensemble['away'] += away_host_boost
        ensemble['draw'] -= away_host_boost * HOST_BOOST_DRAW_TAX
        ensemble['home'] -= away_host_boost * HOST_BOOST_AWAY_TAX

    # Renormalize
    total = ensemble['home'] + ensemble['draw'] + ensemble['away']
    if total > 0:
        ensemble = {k: max(0.005, v / total) for k, v in ensemble.items()}
        total2 = sum(ensemble.values())
        ensemble = {k: v / total2 for k, v in ensemble.items()}

    ensemble['_meta'] = {
        'strength_gap': strength_gap,
        'draw_suppressed': draw_suppression_applied,
        'host_boosted': host_boost_applied,
        'host_boost_amount': host_boost + away_host_boost,
        'stronger_team': stronger_team,
        'weaker_team': weaker_team,
    }

    return ensemble
