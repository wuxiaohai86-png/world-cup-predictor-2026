#!/usr/bin/env python3
"""
FIFA World Cup 2026 Comprehensive Match Predictor
==================================================
Features:
  1. Win/Draw/Loss (Random Forest + Logistic Regression + Poisson ensemble)
  2. Exact score prediction (Poisson model)
  3. Upset probability & shock index
  4. Data analysis report (team profiles, H2H, trends, radar)

Usage:
  python predict_match.py <home_team> <away_team> [year] [--all-scores] [--accuracy]

Refactored: config.py (constants), data.py (features), models.py (ML),
            analysis.py (reports).  Honest temporal-split accuracy.
"""
import sys
from typing import Dict, Any

from config import (
    DISPLAY_WIDTH, PROB_BAR_WIDTH, SCORE_BAR_WIDTH,
    TEMPORAL_SPLIT_YEAR,
)
from data import (
    load_match_data, load_fifa_rankings,
    calculate_team_stats, calculate_head_to_head,
    calculate_manager_stats, calculate_xg_stats,
    get_hosts_for_year,
)
from models import (
    build_wdl_models_temporal, predict_wdl,
    calculate_poisson_lambdas, score_probability_matrix,
    get_top_scorelines, expected_score_from_matrix,
    calibrate_ensemble,
)
from analysis import (
    calculate_upset_metrics, describe_shock,
    generate_analysis_report,
)


def prob_bar(p: float, width: int = PROB_BAR_WIDTH) -> str:
    """Visual probability bar."""
    filled = int(p * width)
    return chr(0x2588) * filled + chr(0x2591) * (width - filled)


def predict_match(
    home_team: str,
    away_team: str,
    year: int = 2026,
    show_all_scores: bool = False,
    show_accuracy: bool = False,
) -> Dict[str, Any]:
    """Run the full prediction pipeline and print results."""

    # ── Load data ──
    df = load_match_data()
    rank_dict, points_dict = load_fifa_rankings()

    # ── Build models on FULL data for best predictions ──
    models = build_wdl_models_temporal(df, split_year=0)

    # ── Compute team stats from ALL data (not just training split) ──
    team_stats = calculate_team_stats(df, current_year=2022)
    h2h_stats = calculate_head_to_head(df)

    # ── Predict WDL ──
    rf_probs, lr_probs = predict_wdl(
        models, home_team, away_team, year, team_stats, h2h_stats)

    # ── Poisson score prediction ──
    home_lambda, away_lambda = calculate_poisson_lambdas(
        home_team, away_team, team_stats, df, year)
    score_probs, poisson_wdl = score_probability_matrix(home_lambda, away_lambda)
    top_scores = get_top_scorelines(score_probs, n=12)
    exp_home, exp_away = expected_score_from_matrix(score_probs)

    # ── Calibrated ensemble ──
    ensemble = calibrate_ensemble(
        rf_probs, lr_probs, poisson_wdl,
        home_team, away_team, year, team_stats, rank_dict)
    meta = ensemble.pop('_meta', {})
    ensemble_winner = max(ensemble, key=ensemble.get)

    # ── Upset metrics ──
    upset = calculate_upset_metrics(
        home_team, away_team, team_stats, rank_dict,
        rf_probs, lr_probs, poisson_wdl)

    # ── Output ──
    sys.stdout.reconfigure(encoding='utf-8')
    dbar = chr(0x2550)
    w = DISPLAY_WIDTH

    label_map = {
        'home': f'{home_team} Win',
        'away': f'{away_team} Win',
        'draw': 'Draw',
    }

    print()
    print(dbar * w)
    print(f'  FIFA WORLD CUP {year} - MATCH PREDICTION & ANALYSIS')
    print(dbar * w)
    print(f'  {home_team} (Home)  vs  {away_team} (Away)')
    print(dbar * w)

    # ── Honest accuracy (temporal split for info only) ──
    if show_accuracy:
        models_eval = build_wdl_models_temporal(df, split_year=TEMPORAL_SPLIT_YEAR)
        ev = models_eval.get('eval')
        if ev:
            print()
            print(f'  HONEST ACCURACY (temporal split: train <{TEMPORAL_SPLIT_YEAR}, '
                  f'test >={TEMPORAL_SPLIT_YEAR})')
            print(f'  {"-" * 66}')
            print(f'  Test matches: {ev["test_matches"]}')
            print(f'  Random Forest:          '
                  f'Accuracy {ev["rf_accuracy"]*100:.1f}%  |  '
                  f'F1(macro) {ev["rf_f1_macro"]*100:.1f}%')
            print(f'  Logistic Regression:    '
                  f'Accuracy {ev["lr_accuracy"]*100:.1f}%  |  '
                  f'F1(macro) {ev["lr_f1_macro"]*100:.1f}%')
            print(f'  NOTE: Models used for prediction are trained on full data (better accuracy).')
            print(f'        These numbers are the LEAK-FREE baseline for comparison.')

    # ── [1] WDL ──
    print()
    print(f'  [1] WIN / DRAW / LOSS PREDICTION')
    print(f'  {"-" * 66}')

    # Individual models
    for model_name, probs, acc_label in [
        ('Random Forest', rf_probs, ''),
        ('Logistic Regression', lr_probs, ''),
    ]:
        print(f'  {model_name:<66}')
        for outcome in ['home', 'draw', 'away']:
            p = probs.get(outcome, 0)
            print(f'    {label_map[outcome]:<22} {p*100:>5.1f}%  {prob_bar(p)}')
        winner = max(probs, key=probs.get)
        print(f'    >>> Predicted: {label_map[winner]}')
        print()

    # Ensemble
    print(f'  {"WEIGHTED ENSEMBLE (Poisson 70% + RF 12% + LR 18%)":<66}')
    for outcome in ['home', 'draw', 'away']:
        p = ensemble[outcome]
        print(f'    {label_map[outcome]:<22} {p*100:>5.1f}%  {prob_bar(p)}')

    if meta.get('draw_suppressed'):
        gap_pct = meta['strength_gap'] * 100
        print(f'    [Draw suppressed due to {gap_pct:.0f}% strength gap]')
    if meta.get('host_boosted') and meta['host_boost_amount'] > 0:
        print(f'    [Host boost +{meta["host_boost_amount"]*100:.0f}pp applied]')
    print(f'    >>> Final Prediction: {label_map[ensemble_winner]}')

    # ── [2] Score Prediction ──
    print()
    print(f'  [2] EXACT SCORE PREDICTION (Poisson Model)')
    print(f'  {"-" * 66}')
    print(f'  Expected goals: {home_team} {exp_home:.2f} - {exp_away:.2f} {away_team}')
    print(f'  Most likely scorelines:')
    print()

    for i, ((h, a), p) in enumerate(top_scores):
        marker = '  >>>' if i == 0 else '    '
        print(f'{marker} {home_team} {h} - {a} {away_team}  |  '
              f'{p*100:>5.1f}%  {prob_bar(p, SCORE_BAR_WIDTH)}')

    print()
    print(f'  Poisson-derived W/D/L:')
    for outcome in ['home', 'draw', 'away']:
        p = poisson_wdl.get(outcome, 0)
        print(f'    {label_map[outcome]:<22} {p*100:>5.1f}%')

    # Betting markets
    over25 = sum(p for (h, a), p in score_probs.items() if h + a > 2.5)
    over15 = sum(p for (h, a), p in score_probs.items() if h + a > 1.5)
    btts = sum(p for (h, a), p in score_probs.items() if h > 0 and a > 0)
    print()
    print(f'  Betting markets (from Poisson):')
    print(f'    Over 1.5 goals:  {over15*100:>5.1f}%')
    print(f'    Over 2.5 goals:  {over25*100:>5.1f}%')
    print(f'    Both teams score: {btts*100:>5.1f}%')

    if show_all_scores:
        print()
        print(f'  Full score matrix (>0.1%):')
        for (h, a), p in sorted(score_probs.items(), key=lambda x: -x[1]):
            if p > 0.001:
                print(f'    {h} - {a}  |  {p*100:>5.2f}%  {prob_bar(p, 20)}')

    # ── [3] Upset ──
    print()
    print(f'  [3] UPSET ANALYSIS')
    print(f'  {"-" * 66}')
    print(f'  Favorite:       {upset["favorite"]}')
    print(f'  Underdog:       {upset["underdog"]}')
    print(f'  Score gap:      {upset["score_gap"]:.1f} points')
    print(f'  Rank gap:       {upset["rank_gap"]:.0f} positions')
    print(f'  Favorite margin:{upset["favorite_margin"]*100:.1f}% win-rate advantage')
    print()
    print(f'  Upset Probability: {upset["upset_probability"]*100:>5.1f}%')
    print(f'  Shock Index:       {upset["shock_index"]:>5.0f}/100')
    print(f'  Level:             {upset["upset_level"]}')
    print()
    print(f'  {describe_shock(upset["shock_index"])}')

    # ── [4] Report ──
    print()
    report = generate_analysis_report(
        home_team, away_team, year, team_stats, h2h_stats,
        rank_dict, points_dict)
    print(f'  [4] DATA ANALYSIS REPORT')
    print(report)

    print()
    print(dbar * w)
    print(f'  Prediction complete. Models trained on {len(df)} matches (1930-2022).')
    print(f'  NOTE: Does not account for 2026 squad changes or current injuries.')
    print(dbar * w)
    print()

    return {
        'ensemble': ensemble,
        'ensemble_winner': ensemble_winner,
        'expected_score': (exp_home, exp_away),
        'top_scorelines': top_scores,
        'upset': upset,
        'poisson_wdl': poisson_wdl,
        'rf_probs': rf_probs,
        'lr_probs': lr_probs,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI ENTRY POINT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    # --group flag: project entire group
    if '--group' in sys.argv:
        from group_projection import predict_group, print_group_table, load_tournament
        idx = sys.argv.index('--group')
        group_name = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else 'A'
        tournament = load_tournament()
        groups = tournament['groups']
        if group_name.upper() in groups:
            result = predict_group(group_name.upper(), groups[group_name.upper()])
            print_group_table(result)
        else:
            print(f"Group '{group_name}' not found. Available: {list(groups.keys())}")
    else:
        home_team = sys.argv[1] if len(sys.argv) > 1 else 'Mexico'
        away_team = sys.argv[2] if len(sys.argv) > 2 else 'South Africa'
        year = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
        show_all = '--all-scores' in sys.argv
        show_acc = '--accuracy' in sys.argv

        predict_match(home_team, away_team, year,
                      show_all_scores=show_all,
                      show_accuracy=show_acc)
