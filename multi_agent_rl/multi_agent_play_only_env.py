import numpy as np
import random

from pettingzoo import AECEnv
from gymnasium import spaces

from euchre.players import (
    PLAYERS,
    EPlayer,
    eplayer_to_team_index,
    get_other_team_index,
    get_teammate,
)
from euchre.deck import DECK_SIZE
from euchre.round import Round, PLAY_CARD_ACTIONS, PLAYING_STATE

from state_encoding.multi_agent_play_only_rl import (
    encode_state,
    encode_playable,
    NUM_ACTIONS,
)

# Reward scales
WIN_TRICK = 3
PER_WON_POINT = 8
ILLEGAL_MOVE = -5


# ===============================================================
# Helper: safe random bidding phase
# ===============================================================
def do_bidding_phase(round_obj: Round):
    """
    Run through the bidding phase safely.

    If get_actions() ever returns empty (which shouldn't happen in a clean
    implementation), force-advance to PLAYING_STATE to avoid crashes.
    """
    safety_counter = 0
    MAX_STEPS = 50  # generous for bidding

    while round_obj.estate != PLAYING_STATE and safety_counter < MAX_STEPS:
        legal = list(round_obj.get_actions())
        if not legal:
            # Fallback: force into playing state to keep env usable for RL.
            round_obj.estate = PLAYING_STATE
            break

        round_obj.take_action(random.choice(legal))
        safety_counter += 1

    # If somehow still not in PLAYING_STATE, force it.
    if round_obj.estate != PLAYING_STATE:
        round_obj.estate = PLAYING_STATE


# ===============================================================
# Compute encoding length from play-only encode_state
# ===============================================================
test_round = Round()
do_bidding_phase(test_round)
ENCODING_LENGTH = len(encode_state(test_round, EPlayer(0)))


# ===============================================================
#                Euchre RL Environment (AEC)
# ===============================================================
class EuchreMultiAgentEnv(AECEnv):
    """
    AECEnv wrapper for play-only multi-agent Euchre.

    - Action space: indices 0..NUM_ACTIONS-1, corresponding to PLAY_CARD_ACTIONS.
    - Observation: encode_state(round, agent_player)
    - Action mask: encode_playable(round, agent_player) ∈ {0,1}^NUM_ACTIONS.
    """

    metadata = {"name": "euchre_multi_agent_play_only_v0"}

    def __init__(self):
        super().__init__()

        self.possible_agents = PLAYERS.copy()
        self.agents = self.possible_agents.copy()

        # Each agent chooses among PLAY_CARD_1..PLAY_CARD_5 (NUM_ACTIONS)
        self.action_spaces = {
            agent: spaces.Discrete(NUM_ACTIONS) for agent in self.agents
        }

        # Observation dict: state vector + action mask
        self.observation_spaces = {
            agent: {
                "observation": spaces.Box(
                    low=0.0, high=1.0, shape=(ENCODING_LENGTH,), dtype=np.float32
                ),
                "action_mask": spaces.Box(
                    low=0, high=1, shape=(NUM_ACTIONS,), dtype=np.int8
                ),
            }
            for agent in self.agents
        }

    # -----------------------------------------------------------
    # OBSERVE
    # -----------------------------------------------------------
    def observe(self, agent):
        """Return observation and action mask for a specific agent."""
        player_index = PLAYERS.index(agent)
        eplayer = EPlayer(player_index)
        return {
            "observation": encode_state(self.round, eplayer),
            "action_mask": encode_playable(self.round, eplayer),
        }

    # -----------------------------------------------------------
    # RESET
    # -----------------------------------------------------------
    def reset(self, seed=None):
        # Start a fresh round AFTER bidding
        self.round = Round()
        do_bidding_phase(self.round)

        # Required PettingZoo fields
        self.agents = self.possible_agents.copy()
        self.agent_selection = PLAYERS[int(self.round.get_current_player())]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    # -----------------------------------------------------------
    # STEP
    # -----------------------------------------------------------
    def step(self, action):
        # PettingZoo: step still called even if agent already done
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_done_step(action)
            return

        # Reset rewards for this step
        self.rewards = {agent: 0.0 for agent in self.agents}
        info = {}
        truncated = False
        terminated = False

        current_agent = self.agent_selection
        current_player_e = self.round.get_current_player()
        current_player_idx = int(current_player_e)
        teammate_agent = PLAYERS[get_teammate(current_player_e)]

        # Team indices
        team_index = eplayer_to_team_index(current_player_e)
        opposing_team_index = get_other_team_index(team_index)

        # -------------------------------------------------------
        # Interpret action index as index into PLAY_CARD_ACTIONS
        # -------------------------------------------------------
        legal_mask = encode_playable(self.round, current_player_e)
        legal_actions = set(self.round.get_actions())

        # Flag if action is illegal (mask says 0 or index out of range)
        illegal = False
        if not (0 <= action < NUM_ACTIONS):
            illegal = True
            real_action = None
        else:
            real_action = PLAY_CARD_ACTIONS[action]
            if legal_mask[action] < 0.5 or real_action not in legal_actions:
                illegal = True

        # -------------------------------------------------------
        # Save trick wins BEFORE any card is actually played
        # -------------------------------------------------------
        before_trick_wins = self.round.trick_wins.copy()

        # -------------------------------------------------------
        # Handle illegal vs legal
        # -------------------------------------------------------
        if illegal:
            # Punish only the acting agent
            self.rewards[current_agent] = ILLEGAL_MOVE

            # Auto-play a random legal card to keep the game moving
            play_legal = [a for a in PLAY_CARD_ACTIONS if a in legal_actions]
            if play_legal:
                fallback_action = random.choice(play_legal)
                self.round.take_action(fallback_action)
            else:
                # No legal play actions from PLAY_CARD_ACTIONS
                # Fallback: if there are *any* legal actions, take one
                if legal_actions:
                    self.round.take_action(random.choice(list(legal_actions)))
                # else: nothing to do; should not happen
        else:
            # LEGAL move: execute the chosen play action
            self.round.take_action(real_action)

        # -------------------------------------------------------
        # Skip unreachable teammate turns when going alone
        # -------------------------------------------------------
        while (
            not self.round.finished
            and self.round.going_alone
            and self.round.get_current_player() == get_teammate(self.round.maker)
        ):
            self.round.take_action(random.choice(list(self.round.get_actions())))

        # Advance to next agent
        self.agent_selection = PLAYERS[int(self.round.get_current_player())]

        # -------------------------------------------------------
        # Compute reward from trick wins and round points
        # -------------------------------------------------------
        reward_delta = 0.0
        terminated = self.round.finished

        # Only compute trick/round rewards for legal moves
        if not illegal:
            # Trick win contributions
            reward_delta += (
                self.round.trick_wins[team_index] - before_trick_wins[team_index]
            ) * WIN_TRICK
            reward_delta += (
                self.round.trick_wins[opposing_team_index]
                - before_trick_wins[opposing_team_index]
            ) * -WIN_TRICK

            # Round points if the round just ended
            if terminated:
                round_points = self.round.round_points
                win_pts = round_points[team_index]
                lose_pts = round_points[opposing_team_index]
                reward_delta += win_pts * PER_WON_POINT + lose_pts * -PER_WON_POINT

        # -------------------------------------------------------
        # Assign final rewards
        # -------------------------------------------------------
        if not illegal and abs(reward_delta) > 0.0:
            # Team-based reward shaping for tricks/round points
            for agent in self.agents:
                self.rewards[agent] = -reward_delta

            self.rewards[current_agent] = reward_delta
            self.rewards[teammate_agent] = reward_delta
        # else:
        #   - illegal: keep ILLEGAL_MOVE on current_agent, others 0
        #   - legal but no trick/round change: keep all 0

        self.terminations = {agent: terminated for agent in self.agents}
        self.truncations = {agent: truncated for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: info for agent in self.agents}

        return observations, self.rewards, self.terminations, self.truncations, infos
