"""
Group stage projection: predict all matches in a group and calculate standings.
Integrates qualifying-performance bonus into team strength.
"""
import json
import sys
import os
import io
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

from predict_match import predict_match

TOURNAMENT_FILE = "data/tournament_2026.json"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIG                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Confederation strength factors (relative to UEFA baseline)
# Based on historical World Cup performance by confederation
CONFED_STRENGTH: Dict[str, float] = {
    'UEFA': 1.00,
    'CONMEBOL': 0.95,
    'CONCACAF': 0.72,
    'AFC': 0.70,
    'CAF': 0.68,
    'OFC': 0.40,
}

# Qualifying performance bonus (added to ensemble home win probability)
# Teams that dominated qualifiers get a boost; playoff winners get less.
QUALIFYING_BONUS: Dict[str, float] = {
    '1st': 0.04,           # Won group comfortably
    'playoff': 0.00,       # Barely qualified — no bonus
    'co-host': 0.06,       # Host nations (extra preparation)
    'intercontinental': -0.02,  # Needed intercontinental playoff
}

# Performance notes that override the generic bonus
PERFORMANCE_OVERRIDES: Dict[str, float] = {
    'perfect record, 0 goals conceded': 0.08,
    'perfect record': 0.07,
    'first World Cup': -0.03,
    'first direct OFC berth': 0.01,
    'knocked out Italy': 0.04,
}


def load_tournament(path: str = TOURNAMENT_FILE) -> Dict[str, Any]:
    """Load tournament configuration."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_qualifying_bonus(team: str, tournament: Dict) -> float:
    """Calculate qualifying-performance bonus for a team (in probability points)."""
    qp = tournament.get('qualifying_performance', {}).get(team, {})
    if not qp:
        return 0.0

    status = qp.get('status', '')
    position = qp.get('position', '')
    note = qp.get('note', '')

    # Determine base bonus
    if status == 'co-host':
        bonus = QUALIFYING_BONUS['co-host']
    elif 'playoff' in position.lower():
        bonus = QUALIFYING_BONUS['playoff']
    elif 'intercontinental' in position.lower():
        bonus = QUALIFYING_BONUS['intercontinental']
    elif position:
        bonus = QUALIFYING_BONUS['1st']  # generic group winner
    else:
        bonus = 0.0

    # Check for performance note overrides
    for keyword, override_bonus in PERFORMANCE_OVERRIDES.items():
        if keyword.lower() in note.lower():
            bonus = override_bonus
            break

    # Confederation strength adjustment
    confed = qp.get('confederation', 'UEFA')
    confed_factor = CONFED_STRENGTH.get(confed, 0.70)

    # Combine: base bonus adjusted by confederation
    return bonus * confed_factor


def predict_match_silent(home_team: str, away_team: str, year: int) -> Dict[str, Any]:
    """Run predict_match without printing to stdout."""
    import os
    old_stdout = sys.stdout
    # Use a real file to avoid StringIO issues with reconfigure()
    devnull = open(os.devnull, 'w', encoding='utf-8')
    sys.stdout = devnull
    try:
        result = predict_match(home_team, away_team, year)
    finally:
        sys.stdout = old_stdout
        devnull.close()
    return result


def predict_group(
    group_name: str,
    teams: List[str],
    year: int = 2026,
    silent: bool = True,
) -> Dict[str, Any]:
    """
    Predict all matches in a group and compute standings.

    Returns dict with:
      - matches: list of match result dicts
      - standings: sorted list of (team, pts, gf, ga, gd)
    """
    tournament = load_tournament()
    standings: Dict[str, Dict[str, Any]] = {
        team: {'pts': 0, 'gf': 0, 'ga': 0, 'gd': 0}
        for team in teams
    }

    match_results = []

    # Generate all 6 matchups in a 4-team group
    matchups = [
        (0, 1), (2, 3),  # Round 1
        (0, 2), (3, 1),  # Round 2
        (3, 0), (1, 2),  # Round 3
    ]

    for home_idx, away_idx in matchups:
        home_team = teams[home_idx]
        away_team = teams[away_idx]

        # Predict match (silently)
        if silent:
            result = predict_match_silent(home_team, away_team, year)
        else:
            result = predict_match(home_team, away_team, year)

        # Get expected score and determine simulated outcome
        exp_home, exp_away = result['expected_score']

        # Apply qualifying bonus to ensemble
        home_q_bonus = get_qualifying_bonus(home_team, tournament)
        away_q_bonus = get_qualifying_bonus(away_team, tournament)

        ens = result['ensemble'].copy()
        ens['home'] = min(0.95, max(0.02, ens['home'] + home_q_bonus - away_q_bonus))
        ens['away'] = min(0.95, max(0.02, ens['away'] + away_q_bonus - home_q_bonus))
        # Renormalize
        total = ens['home'] + ens['draw'] + ens['away']
        if total > 0:
            ens = {k: v / total for k, v in ens.items()}

        # Determine simulated winner (most likely outcome, weighted by probability)
        winner = max(ens, key=ens.get)

        # Simulate score based on expected goals (rounded to nearest likely score)
        home_goals = round(exp_home)
        away_goals = round(exp_away)

        # Ensure the score matches the predicted winner
        if winner == 'home' and home_goals <= away_goals:
            home_goals = away_goals + 1
        elif winner == 'away' and away_goals <= home_goals:
            away_goals = home_goals + 1
        elif winner == 'draw' and home_goals != away_goals:
            avg = round((home_goals + away_goals) / 2)
            home_goals = avg
            away_goals = avg
        if home_goals < 0:
            home_goals = 0
        if away_goals < 0:
            away_goals = 0

        # Update standings
        if winner == 'home':
            standings[home_team]['pts'] += 3
        elif winner == 'away':
            standings[away_team]['pts'] += 3
        else:
            standings[home_team]['pts'] += 1
            standings[away_team]['pts'] += 1

        standings[home_team]['gf'] += home_goals
        standings[home_team]['ga'] += away_goals
        standings[away_team]['gf'] += away_goals
        standings[away_team]['ga'] += home_goals

        match_results.append({
            'home': home_team, 'away': away_team,
            'score': f"{home_goals}-{away_goals}",
            'winner': winner,
            'exp_home': exp_home, 'exp_away': exp_away,
            'home_prob': ens['home'], 'draw_prob': ens['draw'], 'away_prob': ens['away'],
        })

    # Calculate goal difference
    for team in teams:
        standings[team]['gd'] = standings[team]['gf'] - standings[team]['ga']

    # Sort: points, then goal difference, then goals for
    sorted_standings = sorted(
        standings.items(),
        key=lambda x: (x[1]['pts'], x[1]['gd'], x[1]['gf']),
        reverse=True,
    )

    return {
        'group': group_name,
        'matches': match_results,
        'standings': [(team, stats) for team, stats in sorted_standings],
    }


def print_group_table(result: Dict[str, Any]) -> None:
    """Print a formatted group standings table."""
    dbar = chr(0x2550)

    print()
    print(dbar * 72)
    print(f"  GROUP {result['group']} — PROJECTED STANDINGS")
    print(dbar * 72)
    print()

    # Table header
    print(f"  {'Pos':<4} {'Team':<28} {'Pts':>4} {'GF':>4} {'GA':>4} {'GD':>5}")
    print(f"  {'-'*4} {'-'*28} {'-'*4} {'-'*4} {'-'*4} {'-'*5}")

    for i, (team, stats) in enumerate(result['standings'], 1):
        pos_label = f"{i}." + (" ^" if i <= 2 else (" ~" if i == 3 else "  "))
        print(f"  {pos_label:<4} {team:<28} {stats['pts']:>4} {stats['gf']:>4} "
              f"{stats['ga']:>4} {stats['gd']:>+5}")

    print()
    print(f"  ^ Top 2 advance to Round of 32")
    print(f"  ~ 3rd place may advance as one of 8 best 3rd-place teams")
    print()

    # Match results
    print(f"  MATCH RESULTS:")
    print(f"  {'-'*68}")
    for m in result['matches']:
        prob_str = (f"(H:{m['home_prob']*100:.0f}% D:{m['draw_prob']*100:.0f}% "
                    f"A:{m['away_prob']*100:.0f}%)")
        print(f"  {m['home']:<24} {m['score']:>5}  {m['away']:<24}  {prob_str}")
    print()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    import sys

    tournament = load_tournament()
    groups = tournament['groups']

    if len(sys.argv) > 1:
        group_name = sys.argv[1].upper()
        if group_name in groups:
            result = predict_group(group_name, groups[group_name])
            print_group_table(result)
        else:
            print(f"Group {group_name} not found. Available: {list(groups.keys())}")
    else:
        # Print all groups
        for name, teams in sorted(groups.items()):
            result = predict_group(name, teams)
            print_group_table(result)

    # Show key injuries
    injuries = tournament.get('key_injuries_2026', {})
    if injuries:
        print(f"  KEY INJURIES / ABSENCES:")
        for team, issues in injuries.items():
            if issues:
                for issue in issues:
                    print(f"    {team}: {issue}")
        print()
