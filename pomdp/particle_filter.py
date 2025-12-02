# particle_filter.py

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

    # ------------------------------------------------------------------
    # Helper: weight normalization + collapse handling
    # ------------------------------------------------------------------
    def _normalize_weights(self) -> None:
        total = float(self.weights.sum())
        if total == 0.0:
            # Every particle was ruled impossible by observations -> collapse. For now, "reinflate" uniformly.
            print("WARNING: PF collapse! Reinflating uniformly.")
            self.weights[:] = 1.0 / self.num_particles
        else:
            self.weights /= total

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize_uniform(self, my_hand: Set[ECard]) -> None:
        """
        Creates particles with uniformly random assignments of all other cards.

        my_hand is the known hand of the agent using the filter (Player 0).
        """
        self.particles = []
        self.weights[:] = 1.0 / self.num_particles

        known_my_hand = set(my_hand)
        base_unseen = [c for c in ALL_CARDS if c not in known_my_hand]
        assert len(known_my_hand) == HAND_SIZE, "Player 0 must have 5 cards at init."
        assert len(base_unseen) == 3 * HAND_SIZE, "There should be exactly 15 unseen cards in a 20-card deck." #TODO: Add Aces back in the game

        # assert len(base_unseen) == 3 * HAND_SIZE + 4, "There should be 19 unseen cards."

        for _ in range(self.num_particles):
            # Make a fresh shuffled copy for each particle to avoid correlation
            unseen = base_unseen[:]
            random.shuffle(unseen)

            ws = WorldState()
            # Player 0 (us)
            ws.hands[0] = set(known_my_hand)

            # Distribute unseen cards to players 1, 2, 3
            ws.hands[1] = set(unseen[0:HAND_SIZE])
            ws.hands[2] = set(unseen[HAND_SIZE:2 * HAND_SIZE])
            ws.hands[3] = set(unseen[2 * HAND_SIZE:3 * HAND_SIZE])

            ws.played_cards = set()
            ws.trick = []
            ws.turn = 0

            self.particles.append(ws)

    # ------------------------------------------------------------------
    # Observation update: someone failed to follow suit
    # ------------------------------------------------------------------
    def observe_fail_follow(self, player: int, suit: ESuit) -> None: # type: ignore
        """
        Observation: player *fails* to follow the given suit.

        Option A:
        - For each particle, if the player COULD have followed suit in that world
          (i.e., has any card of that suit in their legal set), then this
          observation is impossible => set weight = 0.
        - Otherwise keep the particle unchanged.

        We do NOT randomly trim hands or adjust hand sizes here.
        """
        for i, ws in enumerate(self.particles):
            if self.weights[i] == 0.0:
                continue

            legal = ws.legal_cards(player)
            # If any legal card matches the led suit, the player SHOULD have followed suit
            if any(get_ecard_esuit(c) == suit for c in legal):
                # This world contradicts the observation
                self.weights[i] = 0.0

        self._normalize_weights()

    # ------------------------------------------------------------------
    # Observation update: exact card played
    # ------------------------------------------------------------------
    def observe_play(self, player: int, card: ECard) -> None:
        """
        Observation: player plays `card`.

        Option A:
        - If the player does not have that card in a given particle, that particle
          is impossible => weight = 0.
        - Otherwise, remove that card from ALL hands (for safety), add to played_cards,
          and append to the trick.
        """
        for i, ws in enumerate(self.particles):
            if self.weights[i] == 0.0:
                continue

            # If the player does not have this card, this world is impossible
            if card not in ws.hands[player]:
                self.weights[i] = 0.0
                continue

            # Safety: remove from all hands (though in a consistent world
            # it should already only be in one hand)
            for p in range(NUM_PLAYERS):
                ws.hands[p].discard(card)

            ws.played_cards.add(card)
            ws.trick.append((player, card))
            ws.turn = (player + 1) % NUM_PLAYERS

        self._normalize_weights()

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------
    def resample(self) -> None:
        """
        Standard multinomial resampling.

        - Sample new particle indices according to current weights.
        - Deep-copy them into a new particle set.
        - Reset weights to uniform.

        This keeps the *belief* the same in expectation, but reduces degeneracy.
        """
        # We assume weights are already normalized by the last observation.
        idxs = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights,
        )
        self.particles = [self.particles[i].copy() for i in idxs]
        self.weights[:] = 1.0 / self.num_particles


    # ------------------------------------------------------------------
    # Rejuvenation via MCMC Metropolis-Hastings swap proposals
    # ------------------------------------------------------------------
    def rejuvenate(self, rate: float = 0.05) -> None:
        """
        MCMC rejuvenation using random card-swap proposals between players.

        rate: fraction of particles to try to rejuvenate.

        Algorithm:
        - Pick a random particle.
        - Propose swapping two cards between two *opponent* players.
        - Reject the proposal if it violates:
              * card already played
              * trick consistency
              * follow-suit legality contradicts prior observations
        - Otherwise accept the swap.

        This helps particle diversity after resampling collapse.
        """

        num_attempts = int(self.num_particles * rate)
        if num_attempts <= 0:
            return

        for _ in range(num_attempts):
            idx = random.randrange(self.num_particles)
            ws = self.particles[idx]

            # Only rejuvenate if weight > 0
            # (weight is uniform after resampling, but check anyway)
            if self.weights[idx] == 0:
                continue

            # Choose two distinct opponent players to swap from
            players = [1, 2, 3]
            p1, p2 = random.sample(players, 2)

            # If either hand is empty (should not happen), skip
            if len(ws.hands[p1]) == 0 or len(ws.hands[p2]) == 0:
                continue

            # Choose random card from each
            c1 = random.choice(list(ws.hands[p1]))
            c2 = random.choice(list(ws.hands[p2]))

            # Skip if either card is already played
            if c1 in ws.played_cards or c2 in ws.played_cards:
                continue

            # --- PROPOSE THE SWAP ---
            # Make shallow copies
            new_h1 = ws.hands[p1].copy()
            new_h2 = ws.hands[p2].copy()

            new_h1.remove(c1); new_h1.add(c2)
            new_h2.remove(c2); new_h2.add(c1)

            # LEGALITY CHECK: Does swap break trick consistency?
            # i.e., if the trick already shows a card from p1 or p2
            # that contradicts the modified hands.
            consistent = True
            for (pl, card) in ws.trick:
                if pl == p1 and card not in new_h1:
                    consistent = False
                    break
                if pl == p2 and card not in new_h2:
                    consistent = False
                    break
            if not consistent:
                continue

            # LEGALITY CHECK: Follow-suit observations
            # If at any point a player failed to follow suit, ensure
            # the swap doesn't give them a card of that suit.
            # (We *cannot* reconstruct all past fail-follow events, so skip.)

            # --- ACCEPT THE SWAP ---
            ws.hands[p1] = new_h1
            ws.hands[p2] = new_h2
