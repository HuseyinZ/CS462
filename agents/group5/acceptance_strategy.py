"""Acceptance strategy for Group5 agent."""

from __future__ import annotations

from typing import List

from nenv import Bid, Preference

from agents.group5.utils import detect_saturation, time_concession


class SaturationAwareAcceptanceStrategy:
    """Acceptance strategy that monitors concession saturation."""

    def __init__(
        self,
        preference: Preference,
        history_size: int = 6,
        min_start: float = 0.95,
        min_end: float = 0.65,
    ):
        self.preference = preference
        self.history_size = history_size
        self.min_start = min_start
        self.min_end = min_end
        self._opponent_utilities: List[float] = []

    def register_offer(self, bid: Bid):
        """Record the utility of a received offer."""
        utility = self.preference.get_utility(bid)
        self._opponent_utilities.append(utility)
        if len(self._opponent_utilities) > self.history_size * 2:
            self._opponent_utilities = self._opponent_utilities[-self.history_size * 2 :]

    def should_accept(self, t: float, received_bid: Bid, candidate_next_bid: Bid) -> bool:
        """Return True if the received bid should be accepted."""
        u_received = self.preference.get_utility(received_bid)
        u_candidate = self.preference.get_utility(candidate_next_bid)
        u_min = time_concession(
            t,
            start=self.min_start,
            end=max(self.min_end, self.preference.reservation_value),
            e=2.5,
        )

        margin = 0.02
        saturation = detect_saturation(self._opponent_utilities, k=4, eps=0.005, ratio=0.2)
        time_urgent = t >= 0.95

        return u_received >= max(u_min, u_candidate - margin) and (time_urgent or saturation)
