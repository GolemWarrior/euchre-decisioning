import random

from euchre.round import PLAYING_STATE

from play_only_env import order_weights, order_bias, order_alone_weights, order_alone_bias
from learn_bidding import get_best_bidding_action

class RandomCardAgent:
    def __init__(self):
        self.order_weights, self.order_bias, self.order_alone_weights, self.order_alone_bias = order_weights, order_bias, order_alone_weights, order_alone_bias 

    def play(self, round):
        assert not round.finished, "Agent can't play finished round!"

        if round.estate != PLAYING_STATE:
            round.take_action(get_best_bidding_action(round, self.order_weights, self.order_bias, self.order_alone_weights, self.order_alone_bias))
        else:
            round.take_action(random.choice(list(round.get_actions())))
