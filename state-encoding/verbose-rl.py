import numpy as np
from enum import IntEnum

from euchre.round import FIRST_BIDDING_STATE, SECOND_BIDDING_STATE, DEALER_DISCARD_STATE, CHOOSING_ESUIT_STATE, PLAYING_STATE, DECIDING_GOING_ALONE_STATE, PASS, ORDER_UP, GO_ALONE
from euchre.deck import get_ecard_erank, get_ecard_esuit

class ACTION_TYPE(IntEnum):
    NONE = 0
    BID = 1
    GO_ALONE = 2
    DISCARD = 3
    CHOOSE = 4
    PLAY = 5

def make_none_action_vector():
    np.array([0, ACTION_TYPE.NONE, 0, 0, 0, 0])

def make_action_vector(player, type, is_order, is_go_alone, esuit=None, erank=None):
    esuit_value = 0
    if esuit is not None:
        esuit_value = 1 + int(esuit)
    erank_value = 0
    if erank is not None:
        erank_value = 1 + int(erank)
    return np.array([1 + int(player), type, int(is_order), int(is_go_alone), esuit_value, erank_value])

ACTION_SECTION_COUNTS = {
    ACTION_TYPE.BID: 8,
    ACTION_TYPE.GO_ALONE: 1,
    ACTION_TYPE.CHOOSE: 1,
    ACTION_TYPE.DISCARD: 1,
    ACTION_TYPE.PLAY: 16
}
ACTION_SECTION_ORDER = [ACTION_TYPE.PLAY, ACTION_TYPE.CHOOSE, ACTION_TYPE.DISCARD, ACTION_TYPE.GO_ALONE, ACTION_TYPE.BID]

ACTION_VECTOR_COUNT = sum(list(ACTION_SECTION_COUNTS.values()))

def encode_past_actions(round, player):
    past_actions = round.past_actions

    action_section_indexes = {action_type: 0 for action_type in ACTION_SECTION_ORDER}
    start_index = 0
    for action_type in ACTION_SECTION_ORDER:
        action_section_indexes[action_type] = start_index
        start_index += ACTION_SECTION_COUNTS[action_type]

    action_vectors = [make_none_action_vector() for i in range(ACTION_VECTOR_COUNT)]

    for i in reversed(range(len(past_actions))):
        past_action_record = past_actions[i]
        action_player, action, action_estate, played_ecard = past_action_record

        if action_estate == FIRST_BIDDING_STATE or action_estate == SECOND_BIDDING_STATE:
            action_type = ACTION_TYPE.BID
            if action == PASS:
                action_vectors[action_section_indexes[action_type]] = make_action_vector(action_player, action_type, is_order=False, is_go_alone=False)
                action_vectors[action_section_indexes[action_type]] += 1
            elif action == ORDER_UP:
                action_vectors[action_section_indexes[action_type]] = make_action_vector(action_player, action_type, is_order=True, is_go_alone=False)
                action_vectors[action_section_indexes[action_type]] += 1
        
        if action == DEALER_DISCARD_STATE:
            action_type = ACTION_TYPE.DISCARD

            # Player only gets discard information if they're the dealer
            discarded_esuit = None
            discarded_erank = None
            if player == round.dealer:
                discarded_card = round.discarded_card
                discarded_esuit = get_ecard_esuit(discarded_card)
                discarded_erank = get_ecard_erank(discarded_card)
            
            action_vectors[action_section_indexes[action_type]] = make_action_vector(action_player, action_type, is_order=False, is_go_alone=False, esuit=discarded_esuit, erank=discarded_erank)
            action_vectors[action_section_indexes[action_type]] += 1
        
        if action == CHOOSING_ESUIT_STATE:
            chosen_esuit = round.trump_esuit

            action_type = ACTION_TYPE.CHOOSE
            action_vectors[action_section_indexes[action_type]] = make_action_vector(action_player, action_type, is_order=False, is_go_alone=False, esuit=chosen_esuit)
            action_vectors[action_section_indexes[action_type]] += 1

        if action_estate == DECIDING_GOING_ALONE_STATE:
            going_alone = action == GO_ALONE

            action_type = ACTION_TYPE.GO_ALONE
            action_vectors[action_section_indexes[action_type]] = make_action_vector(action_player, action_type, is_order=False, is_go_alone=going_alone)
            action_vectors[action_section_indexes[action_type]] += 1

        if action_estate == PLAYING_STATE:
            action_type = ACTION_TYPE.PLAY
            action_vectors[action_section_indexes[action_type]] = make_action_vector(action_player, action_type, is_order=False, is_go_alone=False, esuit=get_ecard_esuit(played_ecard), erank=get_ecard_erank(played_ecard))
            action_vectors[action_section_indexes[action_type]] += 1
    
    return action_vectors

def encode_state(round):
    '''
    Returns a vector represented the complete set of past actions in a round without any augmentation, fully determining the state.

    While this is a theoretically perfect representation, functionally it is might not be practical to train on such a large vector
    '''
    assert not round.finished

    past_action_vectors = encode_past_actions(round, round.current_player)

    return np.concatenate(past_action_vectors)