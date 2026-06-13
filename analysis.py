"""
Upset probability, shock index, and data analysis report generation.
"""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from config import (
    UPSET_WIN_RATE_WEIGHT, UPSET_RECENT_FORM_WEIGHT, UPSET_KNOCKOUT_WEIGHT,
    UPSET_RANK_WEIGHT, UPSET_RANK_SHOCK_MAX, UPSET_FORM_SHOCK_MAX,
    UPSET_PROB_SHOCK_MAX, UPSET_RANK_GAP_CAP, UPSET_LEVELS, UPSET_DEFAULT_LEVEL,
    DISPLAY_WIDTH, DEFAULT_FIFA_RANK,
)
from data import get_h2h_report


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UPSET ANALYSIS                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def calculate_upset_metrics(
    home_team: str,
    away_team: str,
    team_stats: Dict[str, Dict[str, Any]],
    rank_dict: Dict[str, int],
    rf_probs: Dict[str, float],
    lr_probs: Dict[str, float],
    poisson_probs: Dict[str, float],
) -> Dict[str, Any]:
    """
    Calculate upset probability and shock index.

    Returns dict with: favorite, underdog, upset_probability, shock_index,
    upset_level, score_gap, rank_gap, favorite_margin.
    """
    hs = team_stats.get(home_team, {})
    aws = team_stats.get(away_team, {})
    home_rank = rank_dict.get(home_team, DEFAULT_FIFA_RANK)
    away_rank = rank_dict.get(away_team, DEFAULT_FIFA_RANK)

    # Composite strength score
    home_score = _composite_strength(hs, home_rank)
    away_score = _composite_strength(aws, away_rank)

    favorite = home_team if home_score >= away_score else away_team
    underdog = away_team if home_score >= away_score else home_team
    score_gap = abs(home_score - away_score)

    # Upset = underdog wins
    if underdog == home_team:
        upset_rf = rf_probs.get('home', 0)
        upset_lr = lr_probs.get('home', 0)
        upset_poisson = poisson_probs.get('home', 0)
    else:
        upset_rf = rf_probs.get('away', 0)
        upset_lr = lr_probs.get('away', 0)
        upset_poisson = poisson_probs.get('away', 0)

    avg_upset_prob = (upset_rf + upset_lr + upset_poisson) / 3

    # Shock index components
    rank_gap = min(abs(home_rank - away_rank), UPSET_RANK_GAP_CAP)
    rank_shock = (rank_gap / UPSET_RANK_GAP_CAP) * UPSET_RANK_SHOCK_MAX

    form_gap = abs(hs.get('win_rate', 0) - aws.get('win_rate', 0))
    form_shock = form_gap * UPSET_FORM_SHOCK_MAX

    prob_shock = (1 - avg_upset_prob) * UPSET_PROB_SHOCK_MAX

    shock_index = rank_shock + form_shock + prob_shock

    # Upset level label
    upset_level = UPSET_DEFAULT_LEVEL
    for threshold, label in UPSET_LEVELS:
        if avg_upset_prob < threshold:
            upset_level = label
            break

    return {
        'favorite': favorite,
        'underdog': underdog,
        'upset_probability': avg_upset_prob,
        'shock_index': shock_index,
        'upset_level': upset_level,
        'score_gap': score_gap,
        'rank_gap': rank_gap,
        'favorite_margin': form_gap,
    }


def _composite_strength(stats: Dict[str, Any], rank: int) -> float:
    return (
        stats.get('win_rate', 0) * UPSET_WIN_RATE_WEIGHT +
        stats.get('recent_form', 0) * UPSET_RECENT_FORM_WEIGHT +
        stats.get('knockout_success_rate', 0) * UPSET_KNOCKOUT_WEIGHT +
        (100 - min(rank, 100)) * UPSET_RANK_WEIGHT
    )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SHOCK INTERPRETATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def describe_shock(shock_index: float) -> str:
    """Human-readable interpretation of shock index."""
    if shock_index >= 80:
        return ("If the underdog wins, this would be a MASSIVE shock — "
                "one of the biggest upsets in World Cup history.")
    elif shock_index >= 60:
        return "If the underdog wins, this would be a SIGNIFICANT upset."
    elif shock_index >= 40:
        return "If the underdog wins, this would be a MODERATE surprise."
    elif shock_index >= 20:
        return "An underdog win would be a MILD surprise."
    else:
        return "These teams are closely matched — either winning is plausible."


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DATA ANALYSIS REPORT                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def generate_analysis_report(
    home_team: str,
    away_team: str,
    year: int,
    team_stats: Dict[str, Dict[str, Any]],
    h2h_stats: Dict,
    rank_dict: Dict[str, int],
    points_dict: Dict[str, float],
) -> str:
    """Generate a comprehensive text-based data analysis report."""
    hs = team_stats.get(home_team, {})
    aws = team_stats.get(away_team, {})
    h2h_report = get_h2h_report(home_team, away_team, h2h_stats)

    home_rank = rank_dict.get(home_team, 'N/A')
    away_rank = rank_dict.get(away_team, 'N/A')
    home_pts = points_dict.get(home_team, 'N/A')
    away_pts = points_dict.get(away_team, 'N/A')

    lines: List[str] = []
    bar = chr(0x2500)

    def section(title: str) -> None:
        lines.append('')
        lines.append(bar * DISPLAY_WIDTH)
        lines.append(f'  {title}')
        lines.append(bar * DISPLAY_WIDTH)

    # ── Section 1: Team Profiles ──
    section(f'TEAM PROFILES: {home_team} (Home) vs {away_team} (Away)')
    lines.append(f'  {"Metric":<32} {"<- " + home_team:>8}    {"-> " + away_team:>8}')
    lines.append(f'  {"-" * 32} {"-" * 8}    {"-" * 8}')
    _metric_row(lines, 'Win Rate', hs.get('win_rate', 0) * 100, aws.get('win_rate', 0) * 100)
    _metric_row(lines, 'Draw Rate', hs.get('draw_rate', 0) * 100, aws.get('draw_rate', 0) * 100)
    _metric_row(lines, 'Recent Form (5yr)', hs.get('recent_form', 0) * 100, aws.get('recent_form', 0) * 100)
    _metric_row(lines, 'Goals per Game', hs.get('goals_per_game', 0), aws.get('goals_per_game', 0))
    _metric_row(lines, 'Goals Conceded/Game', hs.get('goals_conceded_per_game', 0), aws.get('goals_conceded_per_game', 0))
    _metric_row(lines, 'Goal Diff/Game',
                hs.get('goals_per_game', 0) - hs.get('goals_conceded_per_game', 0),
                aws.get('goals_per_game', 0) - aws.get('goals_conceded_per_game', 0))
    _metric_row(lines, 'Clean Sheets', hs.get('clean_sheets', 0), aws.get('clean_sheets', 0))
    _metric_row(lines, 'Total WC Matches', hs.get('total_matches', 0), aws.get('total_matches', 0))
    _metric_row(lines, 'Knockout Success %',
                hs.get('knockout_success_rate', 0) * 100,
                aws.get('knockout_success_rate', 0) * 100)
    _metric_row(lines, 'Scoring Consistency',
                hs.get('goals_consistency', 0) * 100,
                aws.get('goals_consistency', 0) * 100)
    lines.append('')
    lines.append(f'  FIFA Ranking (Oct 2022):  {home_team}: #{home_rank} ({home_pts} pts)  |  '
                 f'{away_team}: #{away_rank} ({away_pts} pts)')

    # ── Section 2: Historical Trends ──
    section('HISTORICAL PERFORMANCE BY DECADE')
    home_decades = hs.get('matches_by_decade', {})
    away_decades = aws.get('matches_by_decade', {})
    all_decades = sorted(set(list(home_decades.keys()) + list(away_decades.keys())))
    if all_decades:
        lines.append(f'  {"Decade":<10} {"<- " + home_team:<15} {"-> " + away_team:<15} {"Bar (Home vs Away)":<30}')
        lines.append(f'  {"-" * 10} {"-" * 15} {"-" * 15} {"-" * 30}')
        all_counts = list(home_decades.values()) + list(away_decades.values())
        max_matches = max(all_counts) if all_counts else 1
        for dec in all_decades:
            h_count = home_decades.get(dec, 0)
            a_count = away_decades.get(dec, 0)
            h_bar = chr(0x2588) * max(1, int(h_count / max(max_matches, 1) * 15))
            a_bar = chr(0x2591) * max(1, int(a_count / max(max_matches, 1) * 15))
            lines.append(f'  {dec:<10} {h_count:>3} matches    {a_count:>3} matches    {h_bar}{a_bar}')

    # ── Section 3: Head-to-Head ──
    section('HEAD-TO-HEAD HISTORY')
    if h2h_report and h2h_report['total'] > 0:
        r = h2h_report
        lines.append(f'  Total meetings: {r["total"]}')
        lines.append(f'  {home_team} wins: {r["home_wins"]} ({r["home_wins"]/r["total"]*100:.1f}%)')
        lines.append(f'  {away_team} wins: {r["away_wins"]} ({r["away_wins"]/r["total"]*100:.1f}%)')
        lines.append(f'  Draws: {r["draws"]} ({r["draws"]/r["total"]*100:.1f}%)')
        lines.append('')
        lines.append('  Match History:')
        for m in sorted(r['matches'], key=lambda x: x['year']):
            lines.append(
                f'    {m["year"]} | {m["round"]:<20} | '
                f'{m["home_team"]} {m["home_score"]}-{m["away_score"]} {m["away_team"]}'
            )
    else:
        lines.append(f'  No previous World Cup meetings between {home_team} and {away_team}.')
        lines.append('  This would be their first-ever World Cup encounter!')

    # ── Section 4: Match Context ──
    section('MATCH CONTEXT')
    hosts_2026 = ['Mexico', 'United States', 'Canada']
    home_is_host = home_team in hosts_2026
    away_is_host = away_team in hosts_2026

    if home_is_host:
        lines.append(f'  [STADIUM] {home_team} is a HOST NATION - significant home advantage expected.')
        lines.append('      Historical host win rate boost: ~15-20%')
    if away_is_host:
        lines.append(f'  [STADIUM] {away_team} is a HOST NATION.')
    if home_is_host and away_is_host:
        lines.append('  [NEUTRAL] Both teams are host nations - neutral advantage.')
    if not home_is_host and not away_is_host:
        lines.append('  [NEUTRAL] Neutral venue - neither team is host.')

    home_ko = hs.get('ko_matches', 0)
    away_ko = aws.get('ko_matches', 0)
    lines.append(f'  Knockout-stage experience: {home_team} {home_ko} matches | '
                 f'{away_team} {away_ko} matches')

    if year > 2022:
        lines.append(f'  [WARNING] Predicting for {year} - extrapolating beyond training data (1930-2022).')
        lines.append('      Features use 2022-era stats; current squads may differ significantly.')

    # ── Section 5: Radar Comparison ──
    section('RADAR COMPARISON (0-100 scale)')
    all_attack = [s.get('goals_per_game', 0) for s in team_stats.values()]
    all_defense = [s.get('goals_conceded_per_game', 0) for s in team_stats.values()]
    all_form = [s.get('recent_form', 0) for s in team_stats.values()]
    all_ko = [s.get('knockout_success_rate', 0) for s in team_stats.values()]
    all_consistency = [s.get('goals_consistency', 0) for s in team_stats.values()]
    all_exp = [s.get('total_matches', 0) for s in team_stats.values()]

    def normalize(val: float, max_val: float, min_val: float = 0) -> float:
        if max_val == min_val:
            return 50
        return max(0, min(100, (val - min_val) / (max_val - min_val) * 100))

    cat_max = {
        'Attack': max(all_attack) if all_attack else 1,
        'Defense': max(all_defense) if all_defense else 1,
        'Form': max(all_form) if all_form else 1,
        'KO Stage': max(all_ko) if all_ko else 1,
        'Consistency': max(all_consistency) if all_consistency else 1,
        'Experience': max(all_exp) if all_exp else 1,
    }

    home_radar = {
        'Attack': normalize(hs.get('goals_per_game', 0), cat_max['Attack']),
        'Defense': 100 - normalize(hs.get('goals_conceded_per_game', 2), cat_max['Defense']),
        'Form': normalize(hs.get('recent_form', 0), cat_max['Form']),
        'KO Stage': normalize(hs.get('knockout_success_rate', 0), cat_max['KO Stage']),
        'Consistency': normalize(hs.get('goals_consistency', 0.5), cat_max['Consistency']),
        'Experience': normalize(hs.get('total_matches', 0), cat_max['Experience']),
    }
    away_radar = {
        'Attack': normalize(aws.get('goals_per_game', 0), cat_max['Attack']),
        'Defense': 100 - normalize(aws.get('goals_conceded_per_game', 2), cat_max['Defense']),
        'Form': normalize(aws.get('recent_form', 0), cat_max['Form']),
        'KO Stage': normalize(aws.get('knockout_success_rate', 0), cat_max['KO Stage']),
        'Consistency': normalize(aws.get('goals_consistency', 0.5), cat_max['Consistency']),
        'Experience': normalize(aws.get('total_matches', 0), cat_max['Experience']),
    }

    lines.append(f'  {"Category":<16} {"<- " + home_team:<26} {"-> " + away_team:<26}')
    lines.append(f'  {"-" * 16} {"-" * 26} {"-" * 26}')
    for cat in ['Attack', 'Defense', 'Form', 'KO Stage', 'Consistency', 'Experience']:
        h_val = home_radar[cat]
        a_val = away_radar[cat]
        h_bar_str = chr(0x2588) * max(1, int(h_val / 5))
        a_bar_str = chr(0x2591) * max(1, int(a_val / 5))
        lines.append(f'  {cat:<16} {h_bar_str:<26} {a_bar_str:<26}')
        lines.append(f'  {"":>16} {h_val:>3.0f}%{"":22} {a_val:>3.0f}%')

    return '\n'.join(lines)


def _metric_row(lines: List[str], label: str, home_val: Any, away_val: Any) -> None:
    """Add a formatted metric row to the report."""
    if isinstance(home_val, float):
        h_str = f'{home_val:>8.2f}'
        a_str = f'{away_val:>8.2f}'
    else:
        h_str = f'{str(home_val):>8}'
        a_str = f'{str(away_val):>8}'
    lines.append(f'  {label:<32} {h_str}    {a_str}')
