import numpy as np

from euchre.deck import get_ecard_esuit
from euchre.players import get_teammate, eplayer_to_team_index, get_other_team_index, PLAYER_COUNT

from .get_card_prob_matrix import get_card_prob_matrix


def encode_state(round):
    '''
    Converts a round to a very minimal (lossy) state vector to use for NN-based RL (e.g. DQL)

    The state vector contains:
    round_state - 0-5 based on the round state
    trump_suit - 0 when unknown, 1-4 for suits
    led_suit - 0 when unknown, 1-4 for suits
    teammate_card - 0 when NA, 1-25 for cards
    opponent_card_1 - 0 when NA, 1-25 for cards
    opponent_card_2 - 0 when NA, 1-25 for cards
    is_team_maker - 0 when unknown, 1 for no, 2 for yes
    is_going_alone - 0 when unknown, 1 for no, 2 for yes (only yes when the player is the one going alone)
    hand_card_1 - 1-25 for cards
    hand_card_2 - 1-25 for cards
    hand_card_3 - 1-25 for cards
    hand_card_4 - 1-25 for cards
    team_tricks_won - 0-4
    opponent_tricks_won - 0-4
    '''

    player = round.current_player

    round_state = int(round.estate)

    trump_suit = 0
    if round.trump_esuit is not None:
        trump_suit = 1 + int(round.trump_esuit)
    
    led_suit = 0
    trick_card_count = len(round.played_ecards)
    teammate_card = 0
    opponent_cards = [0, 0]
    if trick_card_count > 0:
        led_suit = 1 + int(get_ecard_esuit(round.played_ecards[0]))

        opponent_index = 0
        for i in range(trick_card_count):
            back_index = -1 * (trick_card_count) - 1
            played_ecard = round.played_ecards[back_index]
            action_record = round.past_actions[back_index]
            action_player, action, action_estate = action_record

            if get_teammate(player) == action_player:
                teammate_card = 1 + int(played_ecard)
            if action_player != get_teammate(player) and action_player != player:
                opponent_cards[opponent_index] = 1 + int(played_ecard)
                opponent_index += 1
    
    is_team_maker = 0
    if round.maker is not None:
        is_team_maker = 1 + int(round.maker == player or round.maker == get_teammate(player))
    
    is_going_alone = 0
    if round.going_alone is not None:
        is_going_alone = 1 + int(round.going_alone and round.maker == player)
    
    hand_cards = []
    for ecard in round.hands[player]:
        if ecard is None:
            continue
        hand_cards.append(1 + int(ecard))
    
    team_index = eplayer_to_team_index(player)
    team_tricks_won = round.trick_wins[team_index]
    opponent_tricks_won = round.trick_wins[get_other_team_index(team_index)]

    return np.array([round_state, trump_suit, led_suit, teammate_card, *opponent_cards, is_team_maker, is_going_alone, *hand_cards, team_tricks_won, opponent_tricks_won])

def encode_state_with_card_prob(round):
    '''
    Returns a state encoding, specifically the minimal encoding with the flattened card probabilities concatenated to the end
    '''
    player = round.current_player

    card_prob_matrix = get_card_prob_matrix(round, player)
    # Move the columns such that the current player is in the first column
    card_prob_matrix = np.roll(card_prob_matrix, shift=-int(player), axis=1)

    minimal_encoding = encode_state(round)

    return np.concatenate((minimal_encoding, card_prob_matrix.flatten()))