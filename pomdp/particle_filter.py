import random
from typing import List, Set

import numpy as np

from state_encoding.pomdp_state import WorldState
from euchre.deck import (
    ECard,
    ESuit,
    get_ecard_esuit,
)

# TODO: Normalize all these configs in one place so we don't have drift.
ALL_CARDS = list(ECard)
NUM_PLAYERS = 4
HAND_SIZE = 5


class ParticleFilter:
    def __init__(self, num_particles: int = 500):
        self.num_particles = num_particles
        self.particles: List[WorldState] = []
        self.weights: np.ndarray = np.ones(num_particles, dtype=float) / num_particles

    # Helper: weight normalization + collapse handling
    def _normalize_weights(self) -> None:
        total = float(self.weights.sum())
        if total == 0.0:
            # Every particle was ruled impossible by observations -> collapse.
            # For now, "reinflate" uniformly.
            print("WARNING: PF collapse! Reinflating uniformly.")
            self.weights[:] = 1.0 / self.num_particles
        else:
            self.weights /= total

    def effective_sample_size(self) -> float:
        """
        ESS = 1 / sum(w_i^2)
        With uniform weights, ESS = num_particles.
        """
        w = self.weights
        return 1.0 / float(np.sum(w * w))

    # Initialization
    def initialize_uniform(self, my_hand: Set[ECard]) -> None:
        """
        Creates particles with uniformly random assignments of all other cards.

        my_hand is the known hand of the agent using the filter (Player 0).

        NOTE: Right now this assumes a 20-card deck (no Aces):
        - 5 cards to Player 0
        - 5 cards each to players 1, 2, 3
        => 15 unseen cards.

        When you move to the full 24-card Euchre deck, you will need to
        adjust this logic/asserts.
        """
        self.particles = []
        self.weights[:] = 1.0 / self.num_particles

        known_my_hand = set(my_hand)
        base_unseen = [c for c in ALL_CARDS if c not in known_my_hand]

        assert len(known_my_hand) == HAND_SIZE, "Player 0 must have 5 cards at init."
        assert (
            len(base_unseen) == 3 * HAND_SIZE
        ), "There should be exactly 15 unseen cards in a 20-card deck." #TODO: Add Aces to the game

        for _ in range(self.num_particles):
            # Make a fresh shuffled copy for each particle to avoid correlation
            unseen = base_unseen[:]
            random.shuffle(unseen)

            ws = WorldState()

            ws.hands[0] = set(known_my_hand)

            # Distribute unseen cards to players 1, 2, 3 (each gets 5)
            ws.hands[1] = set(unseen[0:HAND_SIZE])
            ws.hands[2] = set(unseen[HAND_SIZE:2 * HAND_SIZE])
            ws.hands[3] = set(unseen[2 * HAND_SIZE:3 * HAND_SIZE])

            ws.played_cards = set()
            ws.trick = []
            ws.turn = 0

            self.particles.append(ws)

    # Observation update: someone failed to follow suit
    def observe_fail_follow(self, player: int, suit: ESuit) -> None:  # type: ignore
        """
        Observation: player *fails* to follow the given suit.

        Option A semantics:

        For each particle:
        - Compute the player's legal cards in this world (respecting follow-suit).
        - If ANY legal card has the observed suit, then in that world the player
          COULD have followed suit but didn't, contradicting the observation.
          => set weight = 0 for that particle.
        - Otherwise this world is consistent; leave it unchanged.

        We do NOT mutate hands or adjust hand sizes here; we only use weights.
        """
        for i, ws in enumerate(self.particles):
            if self.weights[i] == 0.0:
                continue

            legal = ws.legal_cards(player)
            if any(get_ecard_esuit(c) == suit for c in legal):
                # This world contradicts the observation
                self.weights[i] = 0.0

        self._normalize_weights()

    # Observation update: exact card played
    def observe_play(self, player: int, card: ECard) -> None:
        """
        Observation: player plays `card`.

        Option A semantics:

        For each particle:
        - If the specified player does NOT have that card, this world is
          impossible => weight = 0.
        - Otherwise:
            * Remove that card from all hands (for safety; only one should change).
            * Add it to played_cards.
            * Append (player, card) to the trick.
            * Advance ws.turn.
        """
        for i, ws in enumerate(self.particles):
            if self.weights[i] == 0.0:
                continue

            # If the player does not have this card, this world is impossible
            if card not in ws.hands[player]:
                self.weights[i] = 0.0
                continue

            # Remove from all hands (only one will actually contain it)
            for p in range(NUM_PLAYERS):
                ws.hands[p].discard(card)

            ws.played_cards.add(card)
            ws.trick.append((player, card))
            ws.turn = (player + 1) % NUM_PLAYERS

        self._normalize_weights()

    # Resampling
    def resample(self) -> None:
        """
        Standard multinomial resampling.

        - Sample new particle indices according to current weights.
        - Deep-copy them into a new particle set.
        - Reset weights to uniform.

        This keeps the *belief* the same in expectation, but reduces
        weight degeneracy (i.e., avoids a few particles carrying
        almost all of the weight).
        """
        # We assume weights are already normalized by the last observation.
        idxs = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights,
        )
        self.particles = [self.particles[i].copy() for i in idxs]
        self.weights[:] = 1.0 / self.num_particles

    # Rejuvenation (MCMC-style move on particle states, not weights)
    def rejuvenate(self, rate: float = 0.05) -> None:
        """
        MCMC-style rejuvenation that increases *state diversity* (not weight diversity).

        For each particle:
        - With probability `rate`, reshuffle all UNKNOWN cards among players 1, 2, 3,
          subject to these constraints:
            * Player 0's hand is fixed.
            * Played cards are never reintroduced to any hand.
            * The total pool of unknown cards is preserved.
            * Each opponent keeps the same hand size they currently have.

        This does NOT change weights. ESS therefore stays the same, but the
        number of *distinct world states* increases after resampling.
        """
        for ws in self.particles:
            if random.random() > rate:
                continue

            # Cards that are already "used": played or in our hand.
            used_cards = set(ws.played_cards) | set(ws.hands[0])

            # Free cards are everything else still in the deck
            free_cards = [c for c in ALL_CARDS if c not in used_cards]

            # Sanity check: these should exactly match the cards in opponents' hands
            opp_cards = set().union(*[ws.hands[p] for p in range(1, NUM_PLAYERS)])
            assert set(free_cards) == opp_cards, \
                "Rejuvenation invariant violated: opponent cards mismatch."

            random.shuffle(free_cards)

            # Keep opponent hand sizes, but reshuffle which specific cards they hold
            idx = 0
            for p in range(1, NUM_PLAYERS):
                k = len(ws.hands[p])
                ws.hands[p] = set(free_cards[idx:idx + k])
                idx += k

            # We should have consumed all free cards
            assert idx == len(free_cards), \
                "Rejuvenation invariant violated: leftover free cards."
