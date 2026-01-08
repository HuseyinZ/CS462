"""Utility helpers for Group5 agent."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence

from nenv import Bid, Preference
from nenv.Issue import Issue


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into [low, high]."""
    return max(low, min(high, value))


def time_concession(t: float, start: float, end: float, e: float) -> float:
    """Boulware-like concession curve from start to end."""
    t = clamp(t, 0.0, 1.0)
    concession = start - (start - end) * (t**e)
    return clamp(concession, min(start, end), max(start, end))


def rolling_deltas(seq: Sequence[float], k: int) -> List[float]:
    """Return the last k deltas for a numeric sequence."""
    if len(seq) < 2:
        return []
    k = max(1, min(k, len(seq) - 1))
    deltas = [seq[i] - seq[i - 1] for i in range(1, len(seq))]
    return deltas[-k:]


def detect_saturation(utilities: Sequence[float], k: int, eps: float, ratio: float) -> bool:
    """Detect concession saturation when improvements plateau."""
    deltas = rolling_deltas(utilities, k)
    if len(deltas) < k:
        return False
    avg_improvement = sum(max(0.0, d) for d in deltas) / len(deltas)
    spread = max(deltas) - min(deltas) if deltas else 0.0
    return avg_improvement <= eps and spread <= ratio * max(1e-6, abs(avg_improvement))


def select_decoy_issues(
    my_issue_weights: Dict[Issue, float],
    opp_issue_weights: Dict[Issue, float],
    top_m: int,
    top_n: int,
) -> List[Issue]:
    """Select issues that are important for the opponent and less important for us."""
    if not opp_issue_weights:
        return []
    sorted_opp = sorted(opp_issue_weights.items(), key=lambda item: item[1], reverse=True)
    top_opp = [issue for issue, _ in sorted_opp[:max(1, top_m)]]
    scored = []
    for issue in top_opp:
        my_weight = my_issue_weights.get(issue, 0.0)
        opp_weight = opp_issue_weights.get(issue, 0.0)
        scored.append((opp_weight - my_weight, issue))
    scored.sort(reverse=True)
    return [issue for _, issue in scored[:max(0, top_n)]]


def mutate_bid(
    bid: Bid,
    preference: Preference,
    allowed_issues: Iterable[Issue],
    value_selector: Callable[[Issue, Bid], str],
    rng,
) -> Bid:
    """Mutate a bid by changing values of selected issues."""
    new_bid = bid.copy()
    changed = False
    for issue in allowed_issues:
        if rng.random() < 0.5:
            new_value = value_selector(issue, new_bid)
            if new_bid[issue] != new_value:
                new_bid[issue] = new_value
                changed = True
    if not changed and allowed_issues:
        issue = rng.choice(list(allowed_issues))
        new_bid[issue] = value_selector(issue, new_bid)
    new_bid.utility = preference.get_utility(new_bid)
    return new_bid


def build_best_bid(preference: Preference) -> Bid:
    """Create a high-utility bid by selecting the best value per issue."""
    content = {}
    for issue in preference.issues:
        value_weights = preference.value_weights[issue]
        best_value = max(value_weights.items(), key=lambda item: item[1])[0]
        content[issue] = best_value
    bid = Bid(content)
    bid.utility = preference.get_utility(bid)
    return bid
