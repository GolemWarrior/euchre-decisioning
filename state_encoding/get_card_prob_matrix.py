import numpy as np

from euchre.deck import ECard, get_ecard_esuit, get_ecard_erank, get_same_color_esuit, JACK, DECK_SIZE
from euchre.players import EPlayer, PLAYER_COUNT
from euchre.round import PLAYING_STATE


def get_card_prob_matrix(round, player=None):
    '''
    Return matrix representing the probability of a card being in each player's hand given only the
    move history, rules of the game, and the information known to the given player.

    Specifically considers the cards played in the past, the upcard, the discarded card (if the dealer),
    and any times players failed to follow the led suit.

    The returned matrix has DECK_SIZE rows that sum to 1 and four columns indexed by player index
    '''
    if player == None:
        player = round.current_player

    all_played_ecard_lists = round.past_played_ecard_lists.copy()
    all_played_ecard_lists.append(round.played_ecards)

    seen_cards = set()
    for played_ecard_list in all_played_ecard_lists:
        for played_ecard in played_ecard_list:
            seen_cards.add(played_ecard)

    dealer_has_upcard = False
    if round.discarded_card is not None:
        dealer_has_upcard = True
        if player == round.dealer:
            seen_cards.add(round.discarded_card)
    else:
        seen_cards.add(round.upcard)
    
    hand_cards = set([ecard for ecard in round.hands[player] if ecard is not None])
    
    # A matrix where each [ecard, eplayer] represents whether it's possible for eplayer to have ecard in their hand
    possibilities = np.ones((DECK_SIZE, PLAYER_COUNT))

    # Consider played cards, cards in the players hand, and the upcard the dealer might have taken
    for i in range(DECK_SIZE):
        ecard = ECard(i)

        if ecard in seen_cards:
            possibilities[i, :] = 0
        
        if ecard in hand_cards:
            possibilities[i, :] = 0
            possibilities[i, player] = 1
        
        # If the dealer took the upcard, the current iteration is the upcard, and the dealer hasn't yet played the upcard
        if dealer_has_upcard and ecard == round.upcard and not (round.upcard in seen_cards):
            possibilities[i, :] = 0
            possibilities[i, round.dealer] = 1

    # Consider when/if players failed to follow suit
    if round.estate == PLAYING_STATE:
        esuit_constraints = {EPlayer(i): set() for i in range(PLAYER_COUNT)}
        for i, action_record in enumerate([action_record for action_record in round.past_actions if action_record[2] == PLAYING_STATE]):
            action_player, action, action_estate, played_ecard = action_record

            trick_index = int(i / 4)
            played_cards = all_played_ecard_lists[trick_index]
            led_card = played_cards[0]
            led_suit = get_ecard_esuit(led_card)

            is_led = led_suit == get_ecard_esuit(played_ecard)
            is_trump = round.trump_esuit == get_ecard_esuit(played_ecard)
            is_left_bower = get_ecard_erank(played_ecard) == JACK and get_ecard_esuit(played_ecard) == get_same_color_esuit(round.trump_esuit)

            if not (is_led or is_trump or is_left_bower):
                esuit_constraints[action_player].add(led_suit)

        for i in range(DECK_SIZE):
            ecard = ECard(i)
            esuit = get_ecard_esuit(ecard)
            for p in range(PLAYER_COUNT):
                if esuit in esuit_constraints[p]:
                    possibilities[i, p] = 0

    # Normalize the possibilities to get probabilities (divide rows by row sums, unless the sum is zero)
    row_sums = possibilities.sum(axis=1, keepdims=True)
    safe_row_sums = np.where(row_sums == 0, 1, row_sums)
    normalized_possibilities = possibilities / safe_row_sums
    probabilities = normalized_possibilities

    return probabilities
