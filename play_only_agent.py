import os

import numpy as np

from state_encoding.multi_agent_play_only_rl import encode_state, encode_playable

from euchre.deck import ECard
from euchre.round import PLAYING_STATE, PLAY_CARD_ACTIONS, EAction

from play_only_env import order_weights, order_bias, order_alone_weights, order_alone_bias
from learn_bidding import get_best_bidding_action, learn_bidding

from play_only_train import model

AGENT_MODEL_PATH = "play_only_euchre_agent_model.zip"
BIDDING_LEARNING_SIMS = 200000
BIDDING_WEIGHTS_SAVE_PATH = f"play_only_euchre_agent_bidding_weights_{BIDDING_LEARNING_SIMS}.npz"


class PlayOnlyAgent:
    def __init__(self):
        self.order_weights, self.order_bias, self.order_alone_weights, self.order_alone_bias = order_weights, order_bias, order_alone_weights, order_alone_bias
        self.model = model.__class__.load(AGENT_MODEL_PATH)

        self.relearn_bidding_for_self()

    def relearn_bidding_for_self(self):
        if os.path.exists(BIDDING_WEIGHTS_SAVE_PATH):
            data = np.load(BIDDING_WEIGHTS_SAVE_PATH)
            self.order_weights = data["order_weights"]
            self.order_bias = data["order_bias"]
            self.order_alone_weights = data["order_alone_weights"]
            self.order_alone_bias = data["order_alone_bias"]
        else:
            self.order_weights, self.order_bias, self.order_alone_weights, self.order_alone_bias = learn_bidding(self, BIDDING_LEARNING_SIMS)
            np.savez(BIDDING_WEIGHTS_SAVE_PATH, order_weights=self.order_weights, order_bias=self.order_bias, order_alone_weights=self.order_alone_weights, order_alone_bias=self.order_alone_bias)

    def play(self, round):
        assert not round.finished, "Agent can't play finished round!"

        if round.estate != PLAYING_STATE:
            round.take_action(get_best_bidding_action(round, self.order_weights, self.order_bias, self.order_alone_weights, self.order_alone_bias))
        else:
            current_player = round.get_current_player()
            ecard_index, _ = self.model.predict(encode_state(round), action_masks=encode_playable(round), deterministic=True)
            ecard_played = ECard(ecard_index)
            player_hand = round.hands[current_player]
            hand_index = player_hand.index(ecard_played)
            action_index = PLAY_CARD_ACTIONS[hand_index]
            round_action = EAction(int(action_index))
            round.take_action(round_action)
