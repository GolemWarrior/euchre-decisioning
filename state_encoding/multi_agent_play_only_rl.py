import numpy as np

from euchre.deck import ECard, ESuit, get_ecard_score, get_ecard_esuit, get_ecard_erank, get_same_color_esuit, DECK_SIZE, RANKS, SUITS, JACK
from euchre.round import EAction, ACTIONS, PLAY_CARD_ACTIONS, TRICK_COUNT, FIRST_BIDDING_STATE, DEALER_DISCARD_STATE, SECOND_BIDDING_STATE, CHOOSING_ESUIT_STATE, DECIDING_GOING_ALONE_STATE, PLAYING_STATE, PASS, ORDER_UP, GO_ALONE
from euchre.players import EPlayer, get_teammate, eplayer_to_team_index, get_other_team_index, PLAYER_COUNT

# For each possible current player, get a mapping from player to normalized player
player_normalization_mapping = {}
for i in range(PLAYER_COUNT):
    eplayer = EPlayer(i)
    teammate = get_teammate(eplayer)

    opponent_eplayers = []
    for j in range(PLAYER_COUNT): 
        index = (j + i) % PLAYER_COUNT  # Always get opponent in the same direction
        other_eplayer =  EPlayer(index)
        if other_eplayer != eplayer and other_eplayer != teammate:
            opponent_eplayers.append(other_eplayer)
    
    eplayers = [eplayer, opponent_eplayers[0], teammate, opponent_eplayers[1]]

    player_normalization_mapping[eplayer] = {eplayer: EPlayer(i) for i, eplayer in enumerate(eplayers)}

# Per Card Features:
def encode_in_hand(round, agent_player=None):
    agent_player = round.get_current_player() if agent_player is None else agent_player

    hand = round.hands[int(agent_player)]

    in_hand_encoding = np.zeros(DECK_SIZE)

    for card in hand:
        if card is not None:
            in_hand_encoding[int(card)] = 1
    
    return in_hand_encoding

def encode_seen(round, agent_player=None):
    agent_player = round.get_current_player() if agent_player is None else agent_player

    card_seen = np.zeros(DECK_SIZE)
    for i, action_record in enumerate([action_record for action_record in round.past_actions if action_record[2] == PLAYING_STATE]):
        action_player, action, action_estate, played_ecard = action_record

        # Don't include the cards of the non-participating player
        if round.going_alone and action_player == get_teammate(round.maker):
            continue

        card_seen[played_ecard] = 1
    
    if round.dealer == agent_player and round.discarded_card is not None:
        card_seen[round.discarded_card] = 1
    
    return card_seen

def encode_playable(round, agent_player=None):
    eplayer = round.get_current_player()
    hand = round.hands[eplayer]

    legal_actions = round.get_actions()

    playable_encoding = np.zeros(DECK_SIZE)

    # Only have non-zero playable vector when the agent is the player whose turn it is
    if agent_player is None or agent_player == eplayer:
        for play_card_action in legal_actions:
            co_index = PLAY_CARD_ACTIONS.index(play_card_action)

            co_card = hand[co_index]

            playable_encoding[int(co_card)] = 1
    
    return playable_encoding

def encode_win_if_played(round):
    played_ecards = round.played_ecards

    max_score = 0
    for ecard in played_ecards:
        score = get_ecard_score(ecard, round.trump_esuit, get_ecard_esuit(played_ecards[0]))
        max_score = max(score, max_score)
    
    win_if_played_encoding = np.zeros(DECK_SIZE)
    for i in range(DECK_SIZE):
        ecard = ECard(i)
        ecard_score = get_ecard_score(ecard, round.trump_esuit, get_ecard_esuit(played_ecards[0]))
        if ecard_score > max_score:
            win_if_played_encoding[i] = 1
    
    return win_if_played_encoding

MAX_CARD_SCORE = 0
for trump_index in range(len(SUITS)):
    trump_esuit = ESuit(trump_index)

    for led_index in range(len(SUITS)):
        led_esuit = ESuit(led_index)

        for card_index in range(DECK_SIZE):
            ecard = ECard(card_index)
            MAX_CARD_SCORE = max(MAX_CARD_SCORE, get_ecard_score(ecard, trump_esuit, led_esuit))

def encode_strength(round):
    strength_encoding = np.zeros(DECK_SIZE)

    for card_index in range(DECK_SIZE):
        ecard = ECard(card_index)
        card_score = get_ecard_score(ecard, trump_esuit, led_esuit)
        normalized_card_score = card_score / MAX_CARD_SCORE
        strength_encoding[card_index] = normalized_card_score

    return strength_encoding

# Other Agent Moves
def encode_trick_cards(round, agent_player=None):
    agent_player = round.get_current_player() if agent_player is None else agent_player

    played_ecards = set(round.played_ecards)
    if None in played_ecards:
        played_ecards.remove(None)

    trick_cards_encoding = np.zeros((PLAYER_COUNT-1, DECK_SIZE))
    
    for i, action_record in enumerate([action_record for action_record in round.past_actions if action_record[2] == PLAYING_STATE]):
        action_player, action, action_estate, played_ecard = action_record

        # Don't include the cards of the non-participating player
        if round.going_alone and action_player == get_teammate(round.maker):
            continue

        if played_ecard in played_ecards:
            normalized_action_player = player_normalization_mapping[agent_player][action_player]
            trick_cards_encoding[normalized_action_player-1, int(played_ecard)] = 1
    
    return trick_cards_encoding.flatten()

# Other features:
def encode_trump(round):
    trump_encoding = np.zeros(len(SUITS))
    trump_encoding[int(round.trump_esuit)] = 1

    return trump_encoding


def encode_led_suit(round):
    led_suit_encoding = np.zeros(len(SUITS))

    if len(round.played_ecards) > 0:
        led_suit_index = int(get_ecard_esuit(round.played_ecards[0]))
        led_suit_encoding[led_suit_index] = 1
    
    return led_suit_encoding

def encode_trick_wins(round, agent_player=None):
    agent_player = round.get_current_player() if agent_player is None else agent_player

    team_index = eplayer_to_team_index(agent_player)
    other_team_index = get_other_team_index(team_index)

    return np.array([round.trick_wins[team_index], round.trick_wins[other_team_index]])

MAX_HAND_SIZE = TRICK_COUNT
def encode_hand_sizes(round, agent_player=None):
    agent_player = round.get_current_player() if agent_player is None else agent_player

    hand_sizes_encoding = np.zeros(PLAYER_COUNT)

    for player_index in range(PLAYER_COUNT):
        hand = round.hands[player_index]
        hand_count = MAX_HAND_SIZE
        for card in hand:
            if card is None:
                card -= 1

        normalized_player_index = player_normalization_mapping[int(agent_player)][player_index]
        hand_sizes_encoding[normalized_player_index] = hand_count / MAX_HAND_SIZE

    return hand_sizes_encoding

def encode_trumps_played(round):
    trump_count = 0
    for i, action_record in enumerate([action_record for action_record in round.past_actions if action_record[2] == PLAYING_STATE]):
        action_player, action, action_estate, played_ecard = action_record

        # Don't include the cards of the non-participating player
        if round.going_alone and action_player == get_teammate(round.maker):
            continue

        if get_ecard_esuit(played_ecard) == round.trump_esuit:
            trump_count += 1
    
    normalized_trump_count = trump_count / len(RANKS)
    return np.array([normalized_trump_count])

def encode_is_going_alone(round, agent_player=None):
    agent_player = round.get_current_player() if agent_player is None else agent_player

    going_alone = np.zeros(4)

    for player_index in range(PLAYER_COUNT):
        normalized_player = player_normalization_mapping[int(agent_player)][player_index]
        if normalized_player == round.maker and round.going_alone:
            going_alone[int(normalized_player)] = 1
    
    return going_alone

# Combined encoding:
def encode_state(round, agent_player=None):
    assert round.estate == PLAYING_STATE, "Play only encoding can't be done outside of playing phase!"

    return np.concatenate([
        encode_in_hand(round, agent_player),
        encode_seen(round, agent_player),
        encode_playable(round, agent_player),
        encode_win_if_played(round),
        encode_strength(round),
        encode_trick_cards(round, agent_player),
        encode_trump(round),
        encode_led_suit(round),
        encode_trick_wins(round, agent_player),
        encode_hand_sizes(round, agent_player),
        encode_trumps_played(round),
        encode_is_going_alone(round, agent_player)
    ])
