from typing import List, Optional, Set
from euchre.deck import ECard, ESuit, ERank, get_ecard_esuit, get_ecard_erank, get_same_color_esuit
from euchre.round import EAction, CHOOSE_SPADES, CHOOSE_CLUBS, CHOOSE_HEARTS, CHOOSE_DIAMONDS
from euchre.players import EPlayer


# Card value rankings for trump suit
# Right bower (Jack of trump) is most valuable, left bower (Jack of same color) is second
TRUMP_CARD_VALUES = {
    ERank["JACK"]: 6,    # Right bower (if trump suit)
    ERank["ACE"]: 5,
    ERank["KING"]: 4,
    ERank["QUEEN"]: 3,
    ERank["10"]: 2,
    ERank["9"]: 1,
}

LEFT_BOWER_VALUE = 5.5  # Left bower (Jack of same color suit) value

# Thresholds for bidding decisions
ORDER_UP_THRESHOLD_DEALER = 7.0      # Dealer has advantage (gets to pick up card)
ORDER_UP_THRESHOLD_NON_DEALER = 10.0  # Non-dealer needs stronger hand
CALL_SUIT_THRESHOLD = 5.0            # Threshold for calling suit in second round
GO_ALONE_THRESHOLD = 20.0            # Threshold for going alone


def evaluate_hand_for_trump(hand: List[Optional[ECard]], trump_suit: ESuit, include_upcard: Optional[ECard] = None) -> float:
    """
    Evaluate the strength of a hand for a given trump suit.

    Args:
        hand: List of cards in hand (may contain None for played cards)
        trump_suit: The trump suit to evaluate for
        include_upcard: If dealer, include this card as part of hand evaluation

    Returns:
        Float score representing hand strength (higher is better)
    """
    score = 0.0
    same_color_suit = get_same_color_esuit(trump_suit)

    # Evaluate cards in hand
    for card in hand:
        if card is None:
            continue

        card_suit = get_ecard_esuit(card)
        card_rank = get_ecard_erank(card)

        # Right bower (Jack of trump suit)
        if card_suit == trump_suit and card_rank == ERank["JACK"]:
            score += TRUMP_CARD_VALUES[ERank["JACK"]]

        # Left bower (Jack of same color suit)
        elif card_suit == same_color_suit and card_rank == ERank["JACK"]:
            score += LEFT_BOWER_VALUE

        # Other trump cards
        elif card_suit == trump_suit:
            score += TRUMP_CARD_VALUES[card_rank]

        # Non-trump cards have value 0
    # Include upcard if dealer
    if include_upcard is not None:
        upcard_suit = get_ecard_esuit(include_upcard)
        upcard_rank = get_ecard_erank(include_upcard)

        # Right bower
        if upcard_suit == trump_suit and upcard_rank == ERank["JACK"]:
            score += TRUMP_CARD_VALUES[ERank["JACK"]]
        # Left bower
        elif upcard_suit == same_color_suit and upcard_rank == ERank["JACK"]:
            score += LEFT_BOWER_VALUE
        # Other trump
        elif upcard_suit == trump_suit:
            score += TRUMP_CARD_VALUES[upcard_rank]

    return score


def should_order_up(hand: List[Optional[ECard]], upcard: ECard, is_dealer: bool) -> bool:
    """
    Decide whether to order up the face-up card in first bidding round.

    Args:
        hand: Current hand
        upcard: The face-up card
        is_dealer: Whether the player is the dealer

    Returns:
        True if should order up, False if should pass
    """
    trump_suit = get_ecard_esuit(upcard)

    if is_dealer:
        # Dealer gets to pick up the card, so evaluate hand including upcard
        score = evaluate_hand_for_trump(hand, trump_suit, include_upcard=upcard)
        return score >= ORDER_UP_THRESHOLD_DEALER
    else:
        # Non-dealer doesn't get the upcard, so evaluate without it
        score = evaluate_hand_for_trump(hand, trump_suit, include_upcard=None)
        # Penalize slightly since dealer gets advantage
        adjusted_score = score - 1.0
        return adjusted_score >= ORDER_UP_THRESHOLD_NON_DEALER


def choose_best_suit(hand: List[Optional[ECard]], excluded_suit: Optional[ESuit] = None) -> tuple[ESuit, float]:
    """
    Choose the best suit to call in second bidding round.

    Args:
        hand: Current hand
        excluded_suit: Suit that cannot be called (typically upcard suit that was turned down)

    Returns:
        Tuple of (best_suit, score)
    """
    best_suit = None
    best_score = -1.0

    for suit in ESuit:
        # Skip excluded suit
        if excluded_suit is not None and suit == excluded_suit:
            continue

        score = evaluate_hand_for_trump(hand, suit, include_upcard=None)

        if score > best_score:
            best_score = score
            best_suit = suit

    return best_suit, best_score


def should_call_suit_second_round(hand: List[Optional[ECard]], upcard: ECard, is_dealer: bool) -> tuple[bool, Optional[ESuit]]:
    """
    Decide whether to call a suit in second bidding round.

    Args:
        hand: Current hand
        upcard: The face-up card that was turned down
        is_dealer: Whether the player is the dealer

    Returns:
        Tuple of (should_call, suit_to_call)
    """
    excluded_suit = get_ecard_esuit(upcard)
    best_suit, best_score = choose_best_suit(hand, excluded_suit=excluded_suit)

    if is_dealer:
        # Dealer is forced to call in second round, so return best suit
        # (In actual Euchre, dealer must pick if everyone passes)
        return True, best_suit
    else:
        # Non-dealer can still pass
        should_call = best_score >= CALL_SUIT_THRESHOLD
        return should_call, best_suit if should_call else None


def should_go_alone(hand: List[Optional[ECard]], trump_suit: ESuit) -> bool:
    score = evaluate_hand_for_trump(hand, trump_suit, include_upcard=None)
    return score >= GO_ALONE_THRESHOLD


def get_suit_choice_action(suit: ESuit) -> EAction:
    """Convert ESuit to corresponding CHOOSE_* action."""
    suit_to_action = {
        ESuit["SPADES"]: CHOOSE_SPADES,
        ESuit["CLUBS"]: CHOOSE_CLUBS,
        ESuit["HEARTS"]: CHOOSE_HEARTS,
        ESuit["DIAMONDS"]: CHOOSE_DIAMONDS,
    }
    return suit_to_action[suit]

if __name__ == "__main__":
    from euchre.round import Round

    # Create a test round
    round_obj = Round()

    # Get player 0's hand
    player0_hand = round_obj.hands[0]
    upcard = round_obj.upcard
    is_dealer = round_obj.dealer == EPlayer(0)

    print(f"Player 0 hand: {[card.name if card else None for card in player0_hand]}")
    print(f"Upcard: {upcard.name}")
    print(f"Is dealer: {is_dealer}")
    print()

    # Evaluate for upcard suit
    trump_suit = get_ecard_esuit(upcard)
    score = evaluate_hand_for_trump(player0_hand, trump_suit, include_upcard=upcard if is_dealer else None)
    print(f"Hand strength for {trump_suit.name} trump: {score:.1f}")

    # Make bidding decision
    should_order = should_order_up(player0_hand, upcard, is_dealer)
    print(f"Should order up: {should_order}")

    if not should_order:
        # Try second round
        should_call, suit = should_call_suit_second_round(player0_hand, upcard, is_dealer)
        print(f"Should call in second round: {should_call}")
        if should_call:
            print(f"Best suit to call: {suit.name}")
            score = evaluate_hand_for_trump(player0_hand, suit, include_upcard=None)
            print(f"Hand strength for {suit.name} trump: {score:.1f}")
