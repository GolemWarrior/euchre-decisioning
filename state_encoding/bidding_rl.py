import numpy as np

from euchre.deck import (
    DECK_SIZE,
    ECard,
    ESuit,
    get_ecard_esuit,
)
from euchre.round import (
    FIRST_BIDDING_STATE,
    SECOND_BIDDING_STATE,
    CHOOSING_ESUIT_STATE,
    DECIDING_GOING_ALONE_STATE,
    PLAYING_STATE,
    EAction,
    ACTIONS,
)
from euchre.players import (
    EPlayer,
)


def one_hot(n, idx):
    x = np.zeros(n, dtype=np.float32)
    if idx is not None:
        x[int(idx)] = 1.0
    return x


def encode_hand(round, agent_player: EPlayer):
    enc = np.zeros(DECK_SIZE, dtype=np.float32)
    hand = round.hands[int(agent_player)]
    for card in hand:
        if card is not None:
            enc[int(card)] = 1.0
    return enc


def encode_upcard(round):
    enc = np.zeros(DECK_SIZE, dtype=np.float32)
    up = round.upcard
    enc[int(up)] = 1.0
    return enc


def encode_phase(round):
    # 4 bidding-related phases + 'other'
    phase_vec = np.zeros(5, dtype=np.float32)
    if round.estate == FIRST_BIDDING_STATE:
        phase_vec[0] = 1.0
    elif round.estate == SECOND_BIDDING_STATE:
        phase_vec[1] = 1.0
    elif round.estate == CHOOSING_ESUIT_STATE:
        phase_vec[2] = 1.0
    elif round.estate == DECIDING_GOING_ALONE_STATE:
        phase_vec[3] = 1.0
    else:
        phase_vec[4] = 1.0  # not in bidding (shouldn’t be used for bidding net)
    return phase_vec


def encode_dealer_and_turn(round, agent_player: EPlayer):
    is_dealer = 1.0 if round.dealer == agent_player else 0.0
    is_turn = 1.0 if round.get_current_player() == agent_player else 0.0
    return np.array([is_dealer, is_turn], dtype=np.float32)


def encode_trump_info(round):
    # trump suit if already chosen (after CHOOSING_ESUIT_STATE)
    if round.trump_esuit is None:
        return np.zeros(len(ESuit), dtype=np.float32)
    return one_hot(len(ESuit), int(round.trump_esuit))


def encode_bidding_state(round, agent_player: EPlayer):
    """
    Encode state for bidding phases.
    Works in FIRST_BIDDING, SECOND_BIDDING, CHOOSING_ESUIT, DECIDING_GOING_ALONE.
    """
    return np.concatenate([
        encode_hand(round, agent_player),         # 24
        encode_upcard(round),                    # 24
        encode_phase(round),                     # 5
        encode_dealer_and_turn(round, agent_player),  # 2
        encode_trump_info(round),                # 4
    ]).astype(np.float32)


def encode_bidding_action_mask(round):
    """
    Mask over full EAction space (len(ACTIONS)).
    1 for legal actions, 0 otherwise, using round.get_actions().
    """
    mask = np.zeros(len(ACTIONS), dtype=np.float32)
    legal = round.get_actions()
    for a in legal:
        mask[int(a)] = 1.0
    return mask
