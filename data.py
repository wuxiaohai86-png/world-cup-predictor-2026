"""
Data loading, feature engineering, team statistics, and head-to-head analysis.

Key design choice: all team stats can be computed from a temporal subset
of the data (pre-split) to prevent data leakage when evaluating accuracy.
"""
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from config import (
    TEAM_ALIASES, HOSTS_BY_YEAR,
    RECENCY_DECAY_RATE, RECENT_FORM_WINDOW_YEARS,
    DEFAULT_WIN_RATE, DEFAULT_RECENT_FORM, DEFAULT_GOALS_PER_GAME,
    DEFAULT_GOALS_CONCEDED, DEFAULT_FIFA_RANK, DEFAULT_FIFA_POINTS,
    MATCHES_CSV, FIFA_RANKING_CSV,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA LOADING                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_match_data(csv_path: str = MATCHES_CSV) -> pd.DataFrame:
    """Load raw match CSV, normalize team names, return cleaned DataFrame."""
    df = pd.read_csv(csv_path)
    columns_to_keep = [
        'home_team', 'away_team', 'home_score', 'away_score',
        'Year', 'Round', 'Host',
        'home_manager', 'away_manager',          # manager features
        'home_xg', 'away_xg',                     # expected goals (2018+)
    ]
    available_cols = [c for c in columns_to_keep if c in df.columns]
    df_clean = df[available_cols].copy()

    # Normalize historical team names to modern successors
    df_clean['home_team'] = df_clean['home_team'].replace(TEAM_ALIASES)
    df_clean['away_team'] = df_clean['away_team'].replace(TEAM_ALIASES)

    # Create winner column
    def determine_winner(row: pd.Series) -> str:
        if row['home_score'] > row['away_score']:
            return 'home'
        elif row['away_score'] > row['home_score']:
            return 'away'
        return 'draw'

    df_clean['winner'] = df_clean.apply(determine_winner, axis=1)
    df_clean['total_goals'] = df_clean['home_score'] + df_clean['away_score']

    return df_clean


def load_fifa_rankings(path: str = FIFA_RANKING_CSV) -> Tuple[Dict[str, int], Dict[str, float]]:
    """Load FIFA ranking data. Returns (rank_dict, points_dict)."""
    try:
        rankings = pd.read_csv(path)
        rank_dict = dict(zip(rankings['team'], rankings['rank']))
        points_dict = dict(zip(rankings['team'], rankings['points']))
    except FileNotFoundError:
        rank_dict, points_dict = {}, {}
    return rank_dict, points_dict


def get_hosts_for_year(year: int) -> List[str]:
    """Get host nations for a given tournament year."""
    return HOSTS_BY_YEAR.get(year, [])


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TEAM STATISTICS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def calculate_team_stats(
    df: pd.DataFrame,
    current_year: int = 2022,
) -> Dict[str, Dict[str, Any]]:
    """
    Calculate comprehensive team statistics with recency weighting.

    All stats are computed ONLY from the provided DataFrame (no look-ahead).
    For temporal split, pass only pre-split-year data.

    Returns dict: team_name -> stat_dict
    """
    team_stats: Dict[str, Dict[str, Any]] = {}
    all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

    for team in all_teams:
        home_matches = df[df['home_team'] == team].copy()
        away_matches = df[df['away_team'] == team].copy()

        # Recency weights
        home_matches['weight'] = home_matches['Year'].apply(
            lambda y: np.exp(-RECENCY_DECAY_RATE * (current_year - y)))
        away_matches['weight'] = away_matches['Year'].apply(
            lambda y: np.exp(-RECENCY_DECAY_RATE * (current_year - y)))

        # Weighted wins & draws
        home_wins = ((home_matches['winner'] == 'home') * home_matches['weight']).sum()
        away_wins = ((away_matches['winner'] == 'away') * away_matches['weight']).sum()
        total_wins = home_wins + away_wins

        home_draws = ((home_matches['winner'] == 'draw') * home_matches['weight']).sum()
        away_draws = ((away_matches['winner'] == 'draw') * away_matches['weight']).sum()
        total_draws = home_draws + away_draws

        total_weighted = home_matches['weight'].sum() + away_matches['weight'].sum()

        # Win rate & draw rate
        win_rate = total_wins / total_weighted if total_weighted > 0 else 0
        draw_rate = total_draws / total_weighted if total_weighted > 0 else 0

        # Goals scored
        home_gf = (home_matches['home_score'] * home_matches['weight']).sum()
        away_gf = (away_matches['away_score'] * away_matches['weight']).sum()
        goals_per_game = (home_gf + away_gf) / total_weighted if total_weighted > 0 else 0

        # Goals conceded
        home_ga = (home_matches['away_score'] * home_matches['weight']).sum()
        away_ga = (away_matches['home_score'] * away_matches['weight']).sum()
        goals_conceded = (home_ga + away_ga) / total_weighted if total_weighted > 0 else 0

        # Recent form (unweighted, last N years)
        recent_home = home_matches[home_matches['Year'] >= current_year - RECENT_FORM_WINDOW_YEARS]
        recent_away = away_matches[away_matches['Year'] >= current_year - RECENT_FORM_WINDOW_YEARS]
        recent_wins = (len(recent_home[recent_home['winner'] == 'home']) +
                       len(recent_away[recent_away['winner'] == 'away']))
        recent_total = len(recent_home) + len(recent_away)
        recent_form = recent_wins / recent_total if recent_total > 0 else win_rate

        # Clean sheets
        home_cs = len(home_matches[home_matches['away_score'] == 0])
        away_cs = len(away_matches[away_matches['home_score'] == 0])
        clean_sheets = home_cs + away_cs

        # Scoring consistency
        all_goals = list(home_matches['home_score']) + list(away_matches['away_score'])
        goals_std = np.std(all_goals) if len(all_goals) > 1 else 0.5
        goals_consistency = 1.0 - min(goals_std / 3.0, 1.0)

        # Knockout performance
        knockout_rounds = ['Round of 16', 'Quarter-finals', 'Semi-finals',
                           'Third-place match', 'Final']
        ko_matches = home_matches[home_matches['Round'].isin(knockout_rounds)]
        ko_away = away_matches[away_matches['Round'].isin(knockout_rounds)]
        ko_wins = (len(ko_matches[ko_matches['winner'] == 'home']) +
                   len(ko_away[ko_away['winner'] == 'away']))
        ko_total = len(ko_matches) + len(ko_away)
        ko_success_rate = ko_wins / ko_total if ko_total > 0 else 0

        # Decades for reporting
        matches_by_decade: Dict[int, int] = {}
        for _, row in pd.concat([home_matches, away_matches]).iterrows():
            decade = (int(row['Year']) // 10) * 10
            matches_by_decade[decade] = matches_by_decade.get(decade, 0) + 1

        team_stats[team] = {
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'goals_per_game': goals_per_game,
            'goals_conceded_per_game': goals_conceded,
            'recent_form': recent_form,
            'total_matches': len(home_matches) + len(away_matches),
            'clean_sheets': clean_sheets,
            'goals_consistency': goals_consistency,
            'knockout_success_rate': ko_success_rate,
            'ko_matches': ko_total,
            'matches_by_decade': matches_by_decade,
            'all_years': sorted(set(
                list(home_matches['Year']) + list(away_matches['Year']))),
        }

    return team_stats


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MANAGER FEATURES                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def calculate_manager_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-manager win/draw/loss rates from historical data.
    Only uses data from the provided DataFrame (no look-ahead).
    """
    manager_stats: Dict[str, Dict[str, float]] = {}

    for _, row in df.iterrows():
        hm = row.get('home_manager', None)
        aw = row.get('away_manager', None)
        if pd.isna(hm) or pd.isna(aw):
            continue

        for mgr, side in [(hm, 'home'), (aw, 'away')]:
            if mgr not in manager_stats:
                manager_stats[mgr] = {'matches': 0, 'wins': 0, 'draws': 0}
            manager_stats[mgr]['matches'] += 1
            if row['winner'] == side:
                manager_stats[mgr]['wins'] += 1
            elif row['winner'] == 'draw':
                manager_stats[mgr]['draws'] += 1

    # Convert to rates
    result = {}
    for mgr, stats in manager_stats.items():
        m = stats['matches']
        result[mgr] = {
            'win_rate': stats['wins'] / m if m > 0 else 0.5,
            'draw_rate': stats['draws'] / m if m > 0 else 0.25,
            'total_matches': m,
        }

    return result


def get_manager_advantage(
    home_team: str, away_team: str, year: int,
    manager_stats: Dict[str, Dict[str, float]],
    df_all: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Estimate manager win-rate advantage for each team in this matchup.
    Uses the most recent known manager for each team at the given year.
    Returns (home_mgr_winrate, away_mgr_winrate) or defaults.
    """
    # Find the last known manager for each team before the prediction year
    home_mgr_rate = 0.5
    away_mgr_rate = 0.5

    pre_matches = df_all[df_all['Year'] < year]

    # Home team's last manager
    home_rows = pre_matches[
        (pre_matches['home_team'] == home_team) | (pre_matches['away_team'] == home_team)
    ]
    if len(home_rows) > 0:
        last = home_rows.iloc[-1]
        mgr = last['home_manager'] if last['home_team'] == home_team else last.get('away_manager', None)
        if not pd.isna(mgr) and mgr in manager_stats:
            home_mgr_rate = manager_stats[mgr]['win_rate']

    # Away team's last manager
    away_rows = pre_matches[
        (pre_matches['home_team'] == away_team) | (pre_matches['away_team'] == away_team)
    ]
    if len(away_rows) > 0:
        last = away_rows.iloc[-1]
        mgr = last['home_manager'] if last['home_team'] == away_team else last.get('away_manager', None)
        if not pd.isna(mgr) and mgr in manager_stats:
            away_mgr_rate = manager_stats[mgr]['win_rate']

    return home_mgr_rate, away_mgr_rate


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  xG (EXPECTED GOALS) FEATURES                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def calculate_xg_stats(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Calculate team xG stats from matches where xG data is available (2018+).
    xG is a better measure of underlying performance than actual goals.
    """
    xg_stats: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {'xg_for': [], 'xg_against': []})

    xg_matches = df[df['home_xg'].notna() & df['away_xg'].notna()]

    for _, row in xg_matches.iterrows():
        xg_stats[row['home_team']]['xg_for'].append(float(row['home_xg']))
        xg_stats[row['home_team']]['xg_against'].append(float(row['away_xg']))
        xg_stats[row['away_team']]['xg_for'].append(float(row['away_xg']))
        xg_stats[row['away_team']]['xg_against'].append(float(row['home_xg']))

    result = {}
    for team, stats in xg_stats.items():
        result[team] = {
            'xg_per_game': np.mean(stats['xg_for']) if stats['xg_for'] else 0,
            'xga_per_game': np.mean(stats['xg_against']) if stats['xg_against'] else 0,
            'xg_diff': (np.mean(stats['xg_for']) - np.mean(stats['xg_against']))
                       if stats['xg_for'] else 0,
            'xg_matches': len(stats['xg_for']),
        }

    return result


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  HEAD-TO-HEAD ANALYSIS                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def calculate_head_to_head(df: pd.DataFrame) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Calculate detailed head-to-head statistics for all team pairs."""
    h2h: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for _, row in df.iterrows():
        home, away, winner = row['home_team'], row['away_team'], row['winner']
        key = tuple(sorted([home, away]))

        if key not in h2h:
            h2h[key] = {
                'team1': key[0], 'team2': key[1],
                'team1_wins': 0, 'team2_wins': 0, 'draws': 0,
                'total': 0, 'matches': [],
            }

        h2h[key]['total'] += 1
        entry = {
            'year': int(row['Year']),
            'round': row['Round'],
            'home_team': home, 'away_team': away,
            'home_score': int(row['home_score']),
            'away_score': int(row['away_score']),
            'winner': winner,
        }
        h2h[key]['matches'].append(entry)

        if winner == 'draw':
            h2h[key]['draws'] += 1
        elif winner == 'home':
            if home == key[0]:
                h2h[key]['team1_wins'] += 1
            else:
                h2h[key]['team2_wins'] += 1
        else:
            if away == key[0]:
                h2h[key]['team1_wins'] += 1
            else:
                h2h[key]['team2_wins'] += 1

    return h2h


def get_h2h_advantage(home_team: str, away_team: str,
                      h2h_stats: Dict) -> float:
    """Get head-to-head advantage for home team (-1 to +1 scale)."""
    key = tuple(sorted([home_team, away_team]))
    if key not in h2h_stats or h2h_stats[key]['total'] == 0:
        return 0.0

    s = h2h_stats[key]
    if home_team == s['team1']:
        return (s['team1_wins'] - s['team2_wins']) / s['total']
    else:
        return (s['team2_wins'] - s['team1_wins']) / s['total']


def get_h2h_report(home_team: str, away_team: str,
                   h2h_stats: Dict) -> Optional[Dict[str, Any]]:
    """Generate detailed head-to-head report for two teams."""
    key = tuple(sorted([home_team, away_team]))
    if key not in h2h_stats:
        return None

    stats = h2h_stats[key]
    if home_team == stats['team1']:
        home_w = stats['team1_wins']
        away_w = stats['team2_wins']
    else:
        home_w = stats['team2_wins']
        away_w = stats['team1_wins']

    return {
        'total': stats['total'],
        'home_wins': home_w,
        'away_wins': away_w,
        'draws': stats['draws'],
        'matches': stats['matches'],
        'win_rate': home_w / stats['total'] if stats['total'] > 0 else 0,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TEMPORAL SPLIT (for honest accuracy)                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def temporal_split(
    df: pd.DataFrame,
    split_year: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally to prevent leakage.
    - Training: matches before split_year
    - Testing: matches at or after split_year

    Team stats should be computed from training data only.
    """
    train = df[df['Year'] < split_year].copy()
    test = df[df['Year'] >= split_year].copy()
    return train, test
