from enum import IntEnum
import numpy as np
from itertools import product

# Define IntEnums for each card, each suit, and each rank used in Euchre
SUITS = ["SPADES", "CLUBS", "HEARTS", "DIAMONDS"]  # Order is important for binary operations
RANKS = ["9", "10", "JACK", "QUEEN", "KING", "ACE"]  # Order is important for int operations
card_enum_strings = [(suit + "_" + rank, index) for index, (suit, rank) in enumerate(product(SUITS, RANKS))]
DECK_SIZE = len(card_enum_strings)
ECard = IntEnum('ECard', card_enum_strings)
ESuit = IntEnum('ESuit', [(suit, index) for index, suit in enumerate(SUITS)])
ERank = IntEnum('ERank', [(rank, index) for index, rank in enumerate(RANKS)])

def get_ecard_esuit(ecard):
    assert ecard is not None, "ecard is None!"
    return ESuit(int((ecard) / len(RANKS)))

def get_ecard_erank(ecard):
    assert ecard is not None, "ecard is None!"
    return ERank(ecard % len(RANKS))

def get_same_color_esuit(esuit):
    # Gets the other suit with the same color.
    # (Assumes 4 suits and same colors are next to each other)
    return ESuit(esuit ^ 1)

JACK = ERank["JACK"]
def get_ecard_score(ecard, trump_esuit, led_esuit):
    card_esuit = get_ecard_esuit(ecard)
    card_erank = get_ecard_erank(ecard)
    is_trump_suit = card_esuit == trump_esuit
    is_same_color_suit = card_esuit == get_same_color_esuit(trump_esuit)

    max_led_score = len(RANKS)
    max_trump_score = max_led_score + len(RANKS)

    # Score bowers
    if card_erank == JACK:
        if is_trump_suit:
            return max_trump_score + 2
        elif is_same_color_suit:
            return max_trump_score + 1

    # Score non-bower trump suit cards
    if is_trump_suit:
        return max_led_score + card_erank

    # Score non-bower, non-trump, led suit cards
    elif card_esuit == led_esuit:
        return int(card_erank) + 1

    # Non-bower, non-trump, and non-led can't win
    return 0


class Deck:
    def __init__(self):
        self.cards = np.ones((DECK_SIZE, 1))
        self.card_count = DECK_SIZE

    def draw_ecards(self, draw_count=1):
        random_cards = np.random.choice(np.nonzero(self.cards)[0], size=draw_count, replace=False)
        self.cards[random_cards, 0] = 0
        self.card_count -= draw_count
        return random_cards