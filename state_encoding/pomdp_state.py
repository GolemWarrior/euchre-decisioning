from typing import List, Dict, Set
from copy import deepcopy

from euchre.deck import (
    ECard,
    ESuit,
    get_ecard_esuit,
)

#TODO: Normalize all these configs in one place so we don't have drift.
ALL_CARDS = list(ECard)
NUM_PLAYERS = 4
HAND_SIZE = 5


class WorldState:
    """
    Minimal world state for particle filtering.
    Tracks:
    - hands[player] = set of ECard
    - trump suit (ESuit or None)
    - trick = list[(player, card)]
    - turn = current player index
    - played_cards = set of all cards played
    """
    def __init__(self):
        self.hands: Dict[int, Set[ECard]] = {p: set() for p in range(NUM_PLAYERS)}
        self.trump: ESuit | None = None #type: ignore # <-- ESuit is not recognized properly
        self.trick: List[tuple[int, ECard]] = []
        self.turn: int = 0
        self.played_cards: Set[ECard] = set()

    def copy(self):
        return deepcopy(self)

    def legal_cards(self, player: int) -> Set[ECard]:
        """
        Euchre rule: must follow suit if possible.
        """
        hand = self.hands[player]
        if not self.trick:  # player is leading, so they can play anything
            return hand

        led_suit = get_ecard_esuit(self.trick[0][1])
        follow = {c for c in hand if get_ecard_esuit(c) == led_suit}
        return follow if follow else hand
