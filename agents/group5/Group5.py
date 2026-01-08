"""Group5 negotiation agent implementation."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np

import nenv
from nenv import Action, Bid, Offer
from nenv.Agent import AbstractAgent

from agents.group5.acceptance_strategy import SaturationAwareAcceptanceStrategy
from agents.group5.bidding_strategy import SynergisticBiddingStrategy
from agents.group5.opponent_model import FrequencyOpponentModelWithRecency


class Group5(AbstractAgent):
    """Synergistic negotiation agent for Group5."""

    def __init__(self, preference, session_time, estimators):
        super().__init__(preference, session_time, estimators)
        self.bidder: Optional[SynergisticBiddingStrategy] = None
        self.acceptor: Optional[SaturationAwareAcceptanceStrategy] = None
        self.opp_model: Optional[FrequencyOpponentModelWithRecency] = None
        self.last_my_bid: Optional[Bid] = None

    @property
    def name(self) -> str:
        """Return the unique agent name."""
        return "Group5"

    def initiate(self, opponent_name: Optional[str]):
        """Initialize components and random seeds."""
        random.seed(1234)
        np.random.seed(1234)
        self.opp_model = FrequencyOpponentModelWithRecency(self.preference)
        self.bidder = SynergisticBiddingStrategy(self.preference, self.opp_model, random)
        self.acceptor = SaturationAwareAcceptanceStrategy(self.preference)
        self.last_my_bid = None

    def receive_offer(self, bid: Bid, t: float):
        """Update opponent model and acceptance history."""
        if self.opp_model is not None:
            self.opp_model.update(bid, t)
        if self.acceptor is not None:
            self.acceptor.register_offer(bid)

    def act(self, t: float) -> Action:
        """Choose an action based on bidding and acceptance strategies."""
        if self.bidder is None or self.acceptor is None or self.opp_model is None:
            self.initiate(None)

        if not self.last_received_bids:
            opening_bid = self.bidder.propose(t, None, self.last_my_bid)
            self.last_my_bid = opening_bid
            return Offer(opening_bid)

        received_bid = self.last_received_bids[-1]
        candidate = self.bidder.propose(t, received_bid, self.last_my_bid)

        if self.can_accept() and self.acceptor.should_accept(t, received_bid, candidate):
            return self.accept_action

        self.last_my_bid = candidate
        return Offer(candidate)

    def terminate(self, t: float, **kwargs):
        """Finalize the negotiation session."""
        _ = t
