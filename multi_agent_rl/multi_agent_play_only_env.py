import numpy as np
import random

from pettingzoo import AECEnv
from gymnasium import spaces

from euchre.players import PLAYERS, EPlayer, eplayer_to_team_index, get_other_team_index, get_teammate
from euchre.deck import DECK_SIZE, ECard
from euchre.round import Round, PLAY_CARD_ACTIONS, PLAYING_STATE

from state_encoding.multi_agent_play_only_rl import encode_state

# Reward scales
WIN_TRICK = 1
PER_WON_POINT = 10
ILLEGAL_MOVE = -1

def do_bidding_phase(round):
    while round.estate != PLAYING_STATE:
        round.take_action(random.choice(list(round.get_actions())))

test_round = Round()
do_bidding_phase(test_round)
ENCODING_LENGTH = len(encode_state(test_round))

class EuchreMultiAgentEnv(AECEnv):
    def __init__(self):
        super().__init__()

        self.possible_agents = PLAYERS.copy()

        self.agents = self.possible_agents.copy()

        self.action_spaces = {agent: spaces.Discrete(DECK_SIZE) for agent in self.agents}
        self.observation_spaces = {
            agent: {
                "observation":  spaces.Box(low=0.0, high=1.0, shape=(ENCODING_LENGTH,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(DECK_SIZE,), dtype=np.int8)
            } for agent in self.agents}

    def observe(self, agent):
        player_index = PLAYERS.index(agent)
        if self.round.get_current_player() == EPlayer(player_index):
            return encode_state(self.round)
        
        # No observation when it's not the agent's turn. Should make an encoding that works for any agent if using a learning algorithm that uses this
        return np.zeros(ENCODING_LENGTH)

    def reset(self, seed=None):
        self.round = Round()
        do_bidding_phase(self.round)

        # Interface Stuff
        self.agent_selection = PLAYERS[int(self.round.get_current_player())]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, action):
        # Interface Stuff (step is called even when an agent is finished)
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_done_step(action)
            return

        self.rewards = {agent: 0 for agent in self.agents}

        reward = 0
        info = {}
        truncated = False
        terminated = False

        current_agent = self.agent_selection
        teammate_agent = PLAYERS[get_teammate(self.round.get_current_player())]

        ecard_played = ECard(int(action))

        current_player = self.round.get_current_player()
        player_hand = self.round.hands[current_player]

        # If the action is illegal, punish the agent and let it try again (don't change the state)
        if ecard_played not in player_hand:
            # Only the agent that makes the illegal move is punished
            self.rewards[current_agent] += ILLEGAL_MOVE

            observations = {agent: self.observe(agent) for agent in self.agents}
            self.terminations = {agent: terminated for agent in self.agents}
            self.truncations = {agent: truncated for agent in self.agents}
            infos = {agent: info for agent in self.agents}

            return observations, self.rewards, self.terminations, self.truncations, infos
        
        hand_index = player_hand.index(ecard_played)
        action = PLAY_CARD_ACTIONS[hand_index]

        before_trick_wins = self.round.trick_wins.copy()

        self.round.take_action(action)
        self.agent_selection = PLAYERS[int(self.round.get_current_player())]

        team_index = eplayer_to_team_index(self.round.get_current_player())
        opposing_team_index = get_other_team_index(team_index)

        reward += (self.round.trick_wins[team_index] - before_trick_wins[team_index]) * WIN_TRICK
        reward += (self.round.trick_wins[opposing_team_index] - before_trick_wins[opposing_team_index]) * -1 * WIN_TRICK

        terminated = self.round.finished
        if terminated:
            round_points = self.round.round_points
            win_points = round_points[team_index]
            loss_points = round_points[opposing_team_index]
            reward += win_points * PER_WON_POINT + loss_points * -1 * PER_WON_POINT
        
        observations = {agent: self.observe(agent) for agent in self.agents}
        # Teammate shares rewards with agent, opponents get flipped rewards
        for key in self.rewards.keys():
            self.rewards[key] = -reward
        self.rewards[current_agent] = reward
        self.rewards[teammate_agent] = reward
        self.terminations = {agent: terminated for agent in self.agents}
        self.truncations = {agent: truncated for agent in self.agents}
        infos = {agent: info for agent in self.agents}

        return observations, self.rewards, self.terminations, self.truncations, infos