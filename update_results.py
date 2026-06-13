"""
Auto-update real match results by scraping web sources.
Run periodically to keep results_2026.json current.
"""
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

RESULTS_FILE = os.path.join(os.path.dirname(__file__), 'data', 'results_2026.json')
KNOWN_MATCHES = {
    ('Mexico', 'South Africa'): {'home_score': 2, 'away_score': 0, 'date': '2026-06-11', 'group': 'A'},
    ('Korea Republic', 'Czech Republic'): {'home_score': 2, 'away_score': 1, 'date': '2026-06-11', 'group': 'A'},
    ('Canada', 'Bosnia and Herzegovina'): {'home_score': 1, 'away_score': 1, 'date': '2026-06-12', 'group': 'B'},
    ('United States', 'Paraguay'): {'home_score': 4, 'away_score': 1, 'date': '2026-06-13', 'group': 'D'},
}


def load_results() -> Dict[str, Any]:
    if not os.path.exists(RESULTS_FILE):
        return {'matches': [], 'summary': {}}
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(data: Dict[str, Any]) -> None:
    data['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_recorded_match_keys(data: Dict[str, Any]) -> set:
    keys = set()
    for m in data.get('matches', []):
        keys.add((m['home_team'], m['away_team']))
        keys.add((m['away_team'], m['home_team']))
    return keys


def add_result(data: Dict[str, Any], result: Dict[str, Any]) -> bool:
    """Add a new match result. Returns True if added, False if already exists."""
    recorded = get_recorded_match_keys(data)
    key = (result['home_team'], result['away_team'])
    if key in recorded:
        return False

    # Try to get model prediction for comparison
    try:
        from predict_match import predict_match
        pred = predict_match(result['home_team'], result['away_team'], 2026)
        pred_winner = pred['ensemble_winner']
        pred_exp = [round(pred['expected_score'][0], 2), round(pred['expected_score'][1], 2)]
    except Exception:
        pred_winner = 'unknown'
        pred_exp = [0, 0]

    actual_winner = 'home' if result['home_score'] > result['away_score'] else \
                    ('away' if result['away_score'] > result['home_score'] else 'draw')

    entry = {
        'date': result.get('date', datetime.now().strftime('%Y-%m-%d')),
        'group': result.get('group', '?'),
        'home_team': result['home_team'],
        'away_team': result['away_team'],
        'home_score': result['home_score'],
        'away_score': result['away_score'],
        'winner': actual_winner,
        'predicted_winner': pred_winner,
        'predicted_exp_goals': pred_exp,
        'model_correct': actual_winner == pred_winner,
    }

    data['matches'].append(entry)

    # Update summary
    total = len(data['matches'])
    correct = sum(1 for m in data['matches'] if m.get('model_correct'))
    data['summary'] = {
        'total_matches': total,
        'model_correct_wdl': correct,
        'model_accuracy': f"{correct/total*100:.1f}%" if total > 0 else "N/A",
    }

    return True


def search_new_results() -> List[Dict[str, Any]]:
    """
    Search for new 2026 World Cup results.
    This function is called by the auto-update loop.
    Returns list of new match dicts found.
    """
    # We rely on the user/agent providing new results via web search.
    # This function returns an empty list as a placeholder.
    # The actual search happens in the Claude Code agent loop.
    return []


def print_status(data: Dict[str, Any]) -> None:
    s = data.get('summary', {})
    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
          f"Total: {s.get('total_matches', 0)} matches | "
          f"Accuracy: {s.get('model_accuracy', 'N/A')} | "
          f"Last update: {data.get('last_updated', 'never')}")


if __name__ == '__main__':
    data = load_results()
    print_status(data)
    print(f"Recorded matches: {len(data.get('matches', []))}")
    print("Run via Claude Code agent loop for automatic web search + update.")
