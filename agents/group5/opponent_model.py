"""Opponent model with recency-weighted frequencies."""

from __future__ import annotations

from typing import Dict, Optional

from nenv import Bid, Preference
from nenv.Issue import Issue
from nenv.OpponentModel.AbstractOpponentModel import AbstractOpponentModel
from nenv.OpponentModel.EstimatedPreference import EstimatedPreference


class FrequencyOpponentModelWithRecency(AbstractOpponentModel):
    """Frequency-based opponent model with exponential recency weighting."""

    def __init__(self, reference: Preference, gamma: float = 0.9):
        super().__init__(reference)
        self.gamma = gamma
        self._value_counts: Dict[Issue, Dict[str, float]] = {}
        self._change_counts: Dict[Issue, int] = {}
        self._offer_count = 0
        self._last_bid: Optional[Bid] = None

        for issue in reference.issues:
            self._value_counts[issue] = {value: 0.0 for value in issue.values}
            self._change_counts[issue] = 0

    @property
    def name(self) -> str:
        return "Group5FrequencyOpponentModel"

    def update(self, bid: Bid, t: float):
        """Update frequency statistics using a new opponent bid."""
        self._offer_count += 1

        for issue in self._value_counts:
            for value in self._value_counts[issue]:
                self._value_counts[issue][value] *= self.gamma

            selected_value = bid[issue]
            self._value_counts[issue][selected_value] += 1.0

            if self._last_bid is not None and self._last_bid[issue] != selected_value:
                self._change_counts[issue] += 1

        self._last_bid = bid
        self._update_preference()

    def _update_preference(self):
        issue_weights = self._estimate_issue_importance()
        for issue, weight in issue_weights.items():
            self._pref.set_issue_weight(issue, weight)

        for issue, values in self._value_counts.items():
            total = sum(values.values())
            if total <= 0:
                for value in values:
                    self._pref.set_value_weight(issue, value, 1.0)
                continue
            for value, count in values.items():
                self._pref.set_value_weight(issue, value, count / total)

        self._pref.normalize()

    def _estimate_issue_importance(self) -> Dict[Issue, float]:
        if self._offer_count <= 1:
            uniform_weight = 1.0 / len(self._value_counts)
            return {issue: uniform_weight for issue in self._value_counts}

        importance = {}
        for issue, change_count in self._change_counts.items():
            change_rate = change_count / max(1, self._offer_count - 1)
            importance[issue] = max(0.05, 1.0 - change_rate)

        total = sum(importance.values())
        if total <= 0:
            uniform_weight = 1.0 / len(self._value_counts)
            return {issue: uniform_weight for issue in self._value_counts}

        return {issue: weight / total for issue, weight in importance.items()}

    @property
    def preference(self) -> EstimatedPreference:
        return self._pref

    def estimated_utility(self, bid: Bid) -> float:
        """Estimate opponent utility for a bid."""
        return self._pref.get_utility(bid)

    def get_issue_importance(self) -> Dict[Issue, float]:
        """Return current opponent issue importance estimates."""
        return {issue: self._pref.get_issue_weight(issue) for issue in self._value_counts}

    def get_value_preferences(self, issue: Issue) -> Dict[str, float]:
        """Return current opponent value preferences for an issue."""
        return self._pref.value_weights[issue]

    def has_data(self) -> bool:
        """Check whether enough data exists for meaningful estimates."""
        return self._offer_count > 0
