import random
import gym
from gym import spaces
import numpy as np

from euchre.round import Round, EAction, ACTIONS
from euchre.players import EPlayer, eplayer_to_team_index, get_other_team_index, PLAYER_COUNT
from state_encoding.good_rl import encode_state

ENCODING_LENGTH = len(encode_state(Round()))
ACTION_COUNT = len(ACTIONS)

WIN_TRICK = 1
LOSE_TRICK = -1

PER_WON_POINT = 10
PER_LOST_POINT = -10

ILLEGAL_MOVE = -50

def play_round_until_player(round, player):
    # TODO: Train against better play than random
    while not round.finished and round.current_player != player:
        round.take_action(random.choice(list(round.get_actions())))

class EuchreEnvironment(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(ENCODING_LENGTH,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(ACTION_COUNT)

        # Will be set to meaningful values because reset is called before learning starts
        self.state = np.zeros(300, dtype=np.float32)
        self.player = 0
    
    def reset(self):
        self.player = EPlayer(random.randint(0, PLAYER_COUNT - 1))
        self.round = Round()
        play_round_until_player(self.round, self.player)
        if self.round.finished:
            while self.round.finished:
                self.round = Round()
                play_round_until_player(self.round, self.player)

        self.state = encode_state(self.round)
        return self.state
    
    def step(self, action):
        reward = 0
        round_action = EAction(action)
        legal_actions = self.round.get_actions()

        if round_action not in legal_actions:
            reward += ILLEGAL_MOVE
            round_action = random.choice(list(legal_actions))
        
        before_trick_wins = round.trick_wins.copy()

        self.round.take_action(round_action)

        team_index = eplayer_to_team_index(self.player)
        opposing_team_index = get_other_team_index(team_index)

        reward += (round.trick_wins[team_index] - before_trick_wins[team_index]) * WIN_TRICK
        reward += (round.trick_wins[opposing_team_index] - before_trick_wins[opposing_team_index]) * LOSE_TRICK

        self.state = encode_state(self.round)
        
        done = self.round.finished
        if done:
            round_points = self.round.round_points
            win_points = round_points[team_index]
            loss_points = round_points[opposing_team_index]
            reward += win_points * PER_WON_POINT + loss_points * PER_LOST_POINT
        
        info = {}

        return self.state, reward, done, info

