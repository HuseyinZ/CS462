"""Bidding strategy for Group5 agent."""

from __future__ import annotations

from typing import List, Optional

from nenv import Bid, Preference
from nenv.Issue import Issue

from agents.group5.opponent_model import FrequencyOpponentModelWithRecency
from agents.group5.utils import (
    build_best_bid,
    clamp,
    mutate_bid,
    select_decoy_issues,
    time_concession,
)


class SynergisticBiddingStrategy:
    """Time-based, opponent-aware bidding strategy with decoy issues."""

    def __init__(
        self,
        preference: Preference,
        opponent_model: FrequencyOpponentModelWithRecency,
        rng,
        candidate_count: int = 30,
        decoy_top_m: int = 3,
        decoy_top_n: int = 2,
    ):
        self.preference = preference
        self.opponent_model = opponent_model
        self.rng = rng
        self.candidate_count = candidate_count
        self.decoy_top_m = decoy_top_m
        self.decoy_top_n = decoy_top_n
        self._best_bid = build_best_bid(preference)

    def propose(self, t: float, last_received_bid: Optional[Bid], last_my_bid: Optional[Bid]) -> Bid:
        """Generate a candidate bid for the current time."""
        base_bid = last_my_bid or self._best_bid
        target_min = time_concession(
            t,
            start=0.98,
            end=max(self.preference.reservation_value, 0.65),
            e=3.0,
        )

        candidates: List[Bid] = []
        decoy_issues = self._select_decoys()
        allowed_issues = self.preference.issues

        for _ in range(self.candidate_count):
            if last_received_bid is not None and self.rng.random() < 0.25:
                base_bid = last_received_bid
            candidate = mutate_bid(
                base_bid,
                self.preference,
                allowed_issues,
                value_selector=lambda issue, bid: self._select_value(issue, bid, decoy_issues),
                rng=self.rng,
            )
            candidates.append(candidate)

        if not candidates:
            return self._best_bid

        filtered = [bid for bid in candidates if bid.utility >= target_min]
        if not filtered:
            filtered = candidates

        scored = sorted(filtered, key=lambda bid: self._score_bid(bid, t), reverse=True)
        return scored[0]

    def _score_bid(self, bid: Bid, t: float) -> float:
        alpha = clamp(1.0 - t, 0.2, 0.9)
        opp_utility = self.opponent_model.estimated_utility(bid) if self.opponent_model.has_data() else 0.5
        return alpha * bid.utility + (1.0 - alpha) * opp_utility

    def _select_decoys(self) -> List[Issue]:
        if not self.opponent_model.has_data():
            return []
        return select_decoy_issues(
            self.preference.issue_weights,
            self.opponent_model.get_issue_importance(),
            self.decoy_top_m,
            self.decoy_top_n,
        )

    def _select_value(self, issue: Issue, bid: Bid, decoy_issues: List[Issue]) -> str:
        if issue in decoy_issues and self.opponent_model.has_data():
            preferences = self.opponent_model.get_value_preferences(issue)
            return max(preferences.items(), key=lambda item: item[1])[0]

        my_value_weights = self.preference.value_weights[issue]
        best_value = max(my_value_weights.items(), key=lambda item: item[1])[0]

        if self.rng.random() < 0.2:
            return self.rng.choice(list(my_value_weights.keys()))
        return best_value
