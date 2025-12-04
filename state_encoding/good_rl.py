import numpy as np

from euchre.deck import ESuit, get_ecard_esuit, get_ecard_erank, get_same_color_esuit, DECK_SIZE, RANKS, SUITS, JACK
from euchre.round import EAction, ACTIONS, TRICK_COUNT, FIRST_BIDDING_STATE, DEALER_DISCARD_STATE, SECOND_BIDDING_STATE, CHOOSING_ESUIT_STATE, DECIDING_GOING_ALONE_STATE, PLAYING_STATE, PASS, ORDER_UP, GO_ALONE
from euchre.players import EPlayer, get_teammate, eplayer_to_team_index, get_other_team_index, PLAYER_COUNT

from .get_card_prob_matrix import get_card_prob_matrix

# For each possible trump suit, get a mapping from the suits to the normalized suits
suit_normalization_mapping = {}
for i in range(len(SUITS)):
    esuit = ESuit(i)
    same_color_esuit = get_same_color_esuit(esuit)

    other_suits = []
    for j in range(len(SUITS)):
        other_suit =  ESuit(j)
        if other_suit != esuit and other_suit != same_color_esuit:
            other_suits.append(other_suit)
    
    suits = [esuit, same_color_esuit, *other_suits]

    suit_normalization_mapping[esuit] = {esuit: ESuit(i) for i, esuit in enumerate(suits)}

# For each possible current player, get a mapping from player to normalized player
player_normalization_mapping = {}
for i in range(PLAYER_COUNT):
    eplayer = EPlayer(i)
    teammate = get_teammate(eplayer)

    opponent_eplayers = []
    for j in range(PLAYER_COUNT):
        other_eplayer =  EPlayer(j)
        if other_eplayer != eplayer and other_eplayer != teammate:
            opponent_eplayers.append(other_eplayer)
    
    eplayers = [eplayer, opponent_eplayers[0], teammate, opponent_eplayers[1]]

    player_normalization_mapping[eplayer] = {eplayer: EPlayer(i) for i, eplayer in enumerate(eplayers)}

def create_card(ecard, upcard, trump_esuit=None):
    '''
    Create a vector for a card based on the trump suit (or upcard suit if no trump suit)

    If ecard is None returns a zero vector (aka is_present indicator is 0, in addition to the other values)
    Should have an upcard if ecard is not None
    '''
    suit_one_hot = [0] * len(SUITS)
    normalized_rank = 0
    is_right_bower = 0
    is_left_bower = 0
    is_present = 0

    if ecard is not None:
        is_present = 1

        normalize_to_suit = trump_esuit if trump_esuit is not None else get_ecard_esuit(upcard)
        
        esuit = suit_normalization_mapping[normalize_to_suit][get_ecard_esuit(ecard)]
        suit_one_hot[esuit] = 1

        erank = get_ecard_erank(ecard)
        normalized_rank = erank / (len(RANKS) - 1)
        
        if trump_esuit is not None:
            if erank == JACK:
                if esuit == trump_esuit:
                    is_right_bower = 1
                elif esuit == get_same_color_esuit(trump_esuit):
                    is_left_bower = 1

    return np.array([*suit_one_hot, normalized_rank, is_right_bower, is_left_bower, is_present])

# Make a map from RoundEStates to representative integers
round_estate_to_int = {
    FIRST_BIDDING_STATE: 0,         # is_bidding
    SECOND_BIDDING_STATE: 0,        # is_bidding
    DEALER_DISCARD_STATE: 1,        # is_discard_or_choose
    CHOOSING_ESUIT_STATE: 1,        # is_discard_or_choose
    DECIDING_GOING_ALONE_STATE: 2,  # is_deciding_go_alone
    PLAYING_STATE: 3                # is_playing
}

def encode_tricks(round):
    '''
    Return a flattened vector encoding the trick history (including the current trick)
    
    TRICK HISTORY ENCODING:

    (Most recent game is in this slot, then the following games in the next, with unknown values filled with 0s)
    (The start_player also describes who won the previous round)

    start_player: 1 hot 4 vector
    led_suit: 1 hot 4 vector
    played_card: Card
    teammate_card: Card
    opponent0_card: Card
    opponent1_card: Card

    start_player: 1 hot 4 vector
    led_suit: 1 hot 4 vector
    played_card: Card
    teammate_card: Card
    opponent0_card: Card
    opponent1_card: Card

    start_player: 1 hot 4 vector
    led_suit: 1 hot 4 vector
    played_card: Card
    teammate_card: Card
    opponent0_card: Card
    opponent1_card: Card
    
    start_player: 1 hot 4 vector
    led_suit: 1 hot 4 vector
    played_card: Card
    teammate_card: Card
    opponent0_card: Card
    opponent1_card: Card
    '''
    player = round.current_player

    all_played_ecard_lists = round.past_played_ecard_lists.copy()
    all_played_ecard_lists.append(round.played_ecards)

    # Maps trick index to a list of action records for that trick, with the oldest action first
    trick_index_to_action_records = [[] for i in range(TRICK_COUNT)]
    for i, action_record in enumerate([action_record for action_record in round.past_actions if action_record[2] == PLAYING_STATE]):
        trick_index = int(i / 4)
        trick_index_to_action_records[trick_index].append(action_record)
    
    in_order_trick_encodings = []
    for trick_index in range(TRICK_COUNT):
        trick_action_records = trick_index_to_action_records[trick_index]
        trick_cards = all_played_ecard_lists[trick_index] if trick_index <= round.trick_number else []

        start_player_one_hot = [0] * PLAYER_COUNT
        if len(trick_action_records) > 0:
            # Get the player of the first action record
            start_player = trick_action_records[0][0]
            start_player_one_hot[player_normalization_mapping[player][start_player]] = 1
        elif trick_index <= round.trick_number:
            # If no cards have been played in the trick, then the current player is the one who is the starter
            start_player_one_hot[player_normalization_mapping[player][player]] = 1
        
        led_suit_one_hot = [0] * len(SUITS)
        if len(trick_cards) > 0:
            led_ecard = trick_cards[0]
            led_esuit = get_ecard_esuit(led_ecard)
            led_suit_one_hot[suit_normalization_mapping[round.trump_esuit][led_esuit]] = 1
        
        played_ecards = [create_card(None, round.upcard, trump_esuit=round.trump_esuit) for i in range(PLAYER_COUNT)]
        for action_record in trick_action_records:
            action_player, action, action_estate, played_ecard = action_record

            # Ignore the teammate of any player going alone
            if round.going_alone and get_teammate(round.maker) == action_player:
                continue

            normalized_action_player = player_normalization_mapping[player][action_player]

            played_ecards[normalized_action_player] = create_card(played_ecard, round.upcard, trump_esuit=round.trump_esuit)
        
        in_order_trick_encodings.append(np.concatenate([np.array(start_player_one_hot), np.array(led_suit_one_hot), np.concatenate(played_ecards)]))
    
    trick_number = round.trick_number  # 0 for the first trick, 1 for the second, etc
    right_shift = (TRICK_COUNT - trick_number) % TRICK_COUNT
    # Slide the elements to the right by right_shift
    most_recent_first_trick_encodings = in_order_trick_encodings[-right_shift:] + in_order_trick_encodings[:-right_shift]

    return np.concatenate(most_recent_first_trick_encodings)

def encode_seen_cards(round):
    '''
    Return a 20 vector for whether each card has already been played (or discarded if the current player is the dealer)
    '''
    card_seen = [0] * DECK_SIZE
    for i, action_record in enumerate([action_record for action_record in round.past_actions if action_record[2] == PLAYING_STATE]):
        action_player, action, action_estate, played_ecard = action_record

        # Ignore the teammate of any player going alone
        if round.going_alone and get_teammate(round.maker) == action_player:
            continue

        card_seen[played_ecard] = 1
    
    if round.dealer == round.current_player and round.discarded_card is not None:
        card_seen[round.discarded_card] = 1
    
    return np.array(card_seen)

def encode_state(round):
    '''
    Returns an encoded state vector for the given round's state (relative to round.current_player)

    SCHEMA DETAILS:

    Current player is mapped to 0
    Teammate is mapped to 2
    Opponent0 is mapped to 1
    Opponent1 is mapped to 3

    Trump suit is mapped to 0
    Left bower suit is mapped to 1
    Other suit is mapped to 2
    Other other suit is mapped to 3

    Rank is normalized to [0, 1] {9: 0.0, 10: 0.2, J: 0.4, Q: 0.6, K: 0.8, A: 1.0}

    class Card: [suit_one_hot, normalized_rank, is_right_bower, is_left_bower, is_present]


    SCHEMA:

    upcard: Card

    (Hand cards are set to all zeros when played, and card play info can be found in trick history encoding)
    hand0: Card
    hand1: Card
    hand2: Card
    hand3: Card

    round_state: [IntEnum(is_bidding, is_discard_or_choose, is_deciding_go_alone, is_playing), got_to_second_bidding]

    player_pass_last_bid_round: 1 hot 4 vector

    maker: 1 hot 4 vector
    dealer: 1 hot 4 vector
    did_dealer_discard
    is_maker_going_alone

    (After trump is chosen suits are remapped and left bower is marked, so trump is implied)

    trick_number_normalized: [0,3] (Then normalized to [0,1])
    
    team_trick_wins: [0,3] (Then normalized to [0,1])
    opponent_trick_wins: [0,3] (Then normalized to [0,1])

    [TRICK HISTORY ENCODING]

    [EITHER SEEN CARDS (including discarded card if dealer) OR CARD PROB MATRIX]

    legal_actions: 1 hot 9 vector (for whether each action is legal)
    '''
    upcard = round.upcard
    upcard_encoding = create_card(upcard, upcard=upcard, trump_esuit=round.trump_esuit)

    player = round.current_player

    # Encode the cards in the player's hand, with used cards as zero vectors (not sorted so encoding card to action correspondence is clear)
    # (A later extension might be to do sorting and remap the outputted action from the sorted cards to the unsorted cards)
    player_hand = round.hands[player]
    #def sort_card(ecard):
    #    # Sort by is_present, is_right_bower, is_left_bower, is_trump, rank, suit, and finally original ordering
    #    trump_esuit = round.trump_esuit if round.trump_esuit is not None else get_ecard_esuit(round.upcard)
    #    ecard_esuit = get_ecard_esuit(ecard) if ecard is not None else None
    #
    #    is_present = int(ecard is not None)
    #    is_right_bower = int(ecard_esuit == trump_esuit and get_ecard_erank(ecard) == JACK) if ecard is not None else 0
    #    is_left_bower = int(ecard_esuit == get_same_color_esuit(trump_esuit) and get_ecard_erank(ecard) == JACK) if ecard is not None else 0
    #    is_trump = int(ecard_esuit == trump_esuit) if ecard is not None else 0
    #    rank = int(get_ecard_erank(ecard)) if ecard is not None else 0
    #    suit = int(suit_normalization_mapping[trump_esuit][ecard_esuit]) if ecard is not None else 0
    #
    #    return (is_present, is_right_bower, is_left_bower, is_trump, len(RANKS) - rank, suit)
    #    
    #sorted_player_hand = sorted(player_hand, key=sort_card)
    #hand_card_encodings = [create_card(card, upcard=upcard, trump_esuit=round.trump_esuit) for card in sorted_player_hand]
    hand_card_encodings = [create_card(card, upcard=upcard, trump_esuit=round.trump_esuit) for card in player_hand]

    # Get whether the bidding went to the second round (as 0/1)
    got_to_second_bidding = 0
    for action_record in round.past_actions:
        action_player, action, action_estate, played_ecard = action_record
        if action_estate == SECOND_BIDDING_STATE:
            got_to_second_bidding = 1
            break
    
    # Map the RoundEState to a representative int (then normalize)
    round_state_int = round_estate_to_int[round.estate]
    normalized_round_state = round_state_int / max(list(round_estate_to_int.values()))
    
    # Iterate over the actions and get the passes for the last bidding round, the maker, whether the dealer discarded, and whether the maker is going alone
    last_bidding_round = FIRST_BIDDING_STATE if got_to_second_bidding == 0 else SECOND_BIDDING_STATE
    player_passes = [0] * PLAYER_COUNT
    maker = [0] * PLAYER_COUNT
    did_dealer_discard = 0
    is_maker_going_alone = 0
    for action_record in round.past_actions:
        action_player, action, action_estate, played_ecard = action_record
        if action_estate == last_bidding_round:
            if action == PASS:
                player_passes[player_normalization_mapping[player][action_player]] = 1
            if action == ORDER_UP:
                maker[player_normalization_mapping[player][action_player]] = 1
        if action_estate == DEALER_DISCARD_STATE:
            did_dealer_discard = 1
        if action_estate == DECIDING_GOING_ALONE_STATE:
            if action == GO_ALONE:
                is_maker_going_alone = 1
    
    # Get the dealer
    dealer = [0] * PLAYER_COUNT
    dealer[player_normalization_mapping[player][round.dealer]] = 1

    # Get a normalized value for which trick the round is on
    trick_number_normalized = round.trick_number / (TRICK_COUNT - 1)
    # Get a normalized value for how many team tricks have been won
    team_trick_wins = round.trick_wins[eplayer_to_team_index(eplayer)] / (TRICK_COUNT - 1)
    # Get a normalized value for how many opponent tricks have been won
    opponent_trick_wins = round.trick_wins[get_other_team_index(eplayer_to_team_index(eplayer))] / (TRICK_COUNT - 1)

    # Get an encoding for the tricks (e.g. cards played, who went first, etc)
    encoded_tricks = encode_tricks(round)

    seen_cards = encode_seen_cards(round)  # Seen cards is smaller, while card probability matrix has more "intuitive" information, since player suit constraints don't need to be learned
    # Get a matrix for the probability a player has a card based only on the seen cards and failures to follow suit
    #card_prob_matrix = get_card_prob_matrix(round)

    # Added to make selecting legal actions easier in training (outside of training the max legal action is used)
    legal_one_hot = [0] * len(ACTIONS)
    legal_actions = round.get_actions()
    for i in range(len(ACTIONS)):
        action = EAction(i)
        if action in legal_actions:
            legal_one_hot[i] = 1

    return np.concatenate([
        upcard_encoding,
        np.concatenate(hand_card_encodings),
        np.array([got_to_second_bidding]),
        np.array([normalized_round_state]),
        np.array(player_passes),
        np.array(maker),
        np.array([did_dealer_discard]),
        np.array([is_maker_going_alone]),
        np.array(dealer),
        np.array([trick_number_normalized]),
        np.array([team_trick_wins]),
        np.array([opponent_trick_wins]),
        encoded_tricks,
        seen_cards,
        #card_prob_matrix.flatten(),  # Larger vector but also very useful (difficult to learn) information
        np.array(legal_one_hot)
    ])


if __name__ == "__main__":
    # TODO: Add basic tests to confirm functionality 
    pass
