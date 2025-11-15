import random
import numpy as np
from typing import List, Set
from state_encoding.pomdp_state import WorldState
from euchre.deck import (
    ECard,
    ESuit,
    get_ecard_esuit,
)

#TODO: Normalize all these configs in one place so we don't have drift.
ALL_CARDS = list(ECard)
NUM_PLAYERS = 4
HAND_SIZE = 5

class ParticleFilter:
    def __init__(self, num_particles: int = 500):
        self.num_particles = num_particles
        self.particles: List[WorldState] = []
        self.weights = np.ones(num_particles) / num_particles

    # Initialization
    def initialize_uniform(self, my_hand: Set[ECard]):
        """
        Creates particles with uniformly random assignments of all other cards.
        my_hand is the known hand of the agent using the filter.
        """
        self.particles = []
        self.weights = np.ones(self.num_particles) / self.num_particles

        for _ in range(self.num_particles):
            ws = WorldState()
            ws.hands[0] = set(my_hand)

            unseen = [c for c in ALL_CARDS if c not in my_hand]
            random.shuffle(unseen)

            ws.hands[1] = set(unseen[0:5]) #TODO: Get rid of this awful hardcoding for hands later if we get the time
            ws.hands[2] = set(unseen[5:10])
            ws.hands[3] = set(unseen[10:15])
            ws.played_cards = set()
            ws.turn = 0

            self.particles.append(ws)

    # Observation update: someone failed to follow suit (meaning an entire suite can be ruled out)
    def observe_fail_follow(self, player: int, suit: ESuit): # type: ignore # <-- ESuit is not recognized properly
        for ws in self.particles:
            legal = ws.legal_cards(player)
            if any(get_ecard_esuit(c) == suit for c in legal):
                ws.hands[player].clear()  # incompatible world state, will have a weight of 0 later


    # Observation update: exact play
    def observe_play(self, player: int, card: ECard):
        for ws in self.particles:
            # Remove the card from every player's hand if present
            for p in range(NUM_PLAYERS):
                ws.hands[p].discard(card)  # discard is safe even if card not present

            # Update played cards and trick
            ws.played_cards.add(card)
            ws.trick.append((player, card))
            ws.turn = (player + 1) % NUM_PLAYERS


    # Resampling --> Get rid of impossible worlds, and resample according to weights (all idx have equal probability)
    def resample(self):
        weights = np.array([1 if ws.hands[0] else 0 for ws in self.particles], dtype=float)
        total = weights.sum()
        if total == 0:
            weights = np.ones(self.num_particles) / self.num_particles
        else:
            weights /= total

        idxs = np.random.choice(self.num_particles, size=self.num_particles, p=weights)
        self.particles = [self.particles[i].copy() for i in idxs]
        self.weights = np.ones(self.num_particles) / self.num_particles

    # Markov Chain Monte Carlo (MCMC) rejuvenation --> REMOVED FOR NOW. It works, but I actually have come to the conclusion it is completely unecessary for euchre since the state space is so small and
    # there are such few legal swaps that can even be done. Leaving this here for now in case we want to revisit later.
    # def rejuvenate(self, rate: float = 0.05):
    #     """
    #     MCMC-style rejuvenation that respects constraints.
        
    #     For each particle:
    #     - With probability `rate`, reshuffle unknown cards among other players.
    #     - Player 0's hand is fixed.
    #     - Played cards and cards known impossible for a player are never reassigned to that player.
    #     """
    #     for ws in self.particles:
    #         if random.random() > rate:
    #             continue

    #         # Collect free cards: all cards not in your hand and not yet played
    #         used_cards = set(ws.played_cards) | set(ws.hands[0])
    #         free_cards = [c for c in ALL_CARDS if c not in used_cards]
    #         random.shuffle(free_cards)

    #         # Track how many cards each player already cannot have (from fail-follow)
    #         cannot_have = {p: set() for p in range(NUM_PLAYERS)}
    #         for p in range(1, NUM_PLAYERS):
    #             # Any cards that ws.hands[p] is currently missing are "impossible" for this player(i.e., fail-follow eliminated them)
    #             cannot_have[p] = set(c for c in ALL_CARDS if c not in ws.hands[p])

    #         # Reset hands for other players
    #         for p in range(1, NUM_PLAYERS):
    #             ws.hands[p] = set()

    #         # Assign cards respecting constraints
    #         idx = 0
    #         for _ in range(HAND_SIZE):
    #             for p in range(1, NUM_PLAYERS):
    #                 while idx < len(free_cards) and free_cards[idx] in cannot_have[p]:
    #                     idx += 1
    #                 if idx < len(free_cards):
    #                     ws.hands[p].add(free_cards[idx])
    #                     idx += 1


    # # Sample a particle index -- idk, could be useful later on but we can delete
    # def sample_particle_index(self) -> int:
    #     return np.random.choice(self.num_particles, p=self.weights)
