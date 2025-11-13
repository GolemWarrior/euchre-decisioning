import gym
from gym import spaces
import numpy as np

from euchre.round import Round, ACTIONS
from state_encoding.good_rl import encode_state

ENCODING_LENGTH = len(encode_state(Round()))
ACTION_COUNT = len(ACTIONS)

def play_round_until_player(round, player):
    pass

class EuchreEnvironment:
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(ENCODING_LENGTH,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(ACTION_COUNT)

        self.state = np.zeros(300, dtype=np.float32)
    
    def reset(self):
        self.round = Round()
        play_round_until_player(self.round)
        self.state = encode_state(self.round)
        return self.state
    
    def step(self, action):
        #self.state = ...
        reward = 0
        done = self.round.finished
        info = {}

        return self.state, reward, done, info

