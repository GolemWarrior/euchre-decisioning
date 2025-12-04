import random
from typing import Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from euchre.round import Round, EAction
from euchre.players import EPlayer, eplayer_to_team_index, get_other_team_index
from euchre.deck import DECK_SIZE
from state_encoding.multi_agent_play_only_rl import (
    encode_state,
    encode_playable,
    NUM_ACTIONS,
)

# Reward scales – tweak if you like
WIN_TRICK = 1          # (optional – not used in this minimal version)
PER_WON_POINT = 10     # per round point (win/loss)
ILLEGAL_MOVE = -1.0    # penalty for illegal move


class EuchreSelfPlayEnv(gym.Env):
    """
    Single-agent environment where ONE shared policy controls all four seats.

    On each call to step(action):
      - We figure out whose turn it is from Round
      - Interpret `action` as an EAction index
      - If illegal, give ILLEGAL_MOVE and replace with random legal action
      - Apply the action via Round.take_action
      - If the round ends, give reward from the perspective of the SEAT that just acted

    This is standard parameter-sharing self-play.
    """

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()

        # Build a dummy round just to infer obs_dim
        dummy_round = Round()
        dummy_player = dummy_round.get_current_player()
        dummy_obs = encode_state(dummy_round, dummy_player)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.round: Round | None = None
        self.current_player: EPlayer | None = None
        self.last_action_mask: np.ndarray | None = None

    def _get_obs_and_mask(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.round is not None
        eplayer = self.round.get_current_player()
        obs = encode_state(self.round, eplayer)
        mask = encode_playable(self.round, eplayer)
        return obs.astype(np.float32), mask.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.round = Round()
        self.current_player = self.round.get_current_player()
        obs, mask = self._get_obs_and_mask()
        self.last_action_mask = mask
        info = {"action_mask": mask}
        return obs, info

    def step(self, action: int):
        assert self.round is not None, "Call reset() before step()."

        # Who is acting now?
        acting_player: EPlayer = self.round.get_current_player()

        # Interpret action as EAction
        try:
            round_action = EAction(action)
        except ValueError:
            # Completely invalid index
            round_action = None

        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        legal_actions = self.round.get_actions()

        # Illegal move handling: penalize and fall back to a random legal action
        if round_action not in legal_actions:
            reward += ILLEGAL_MOVE
            round_action = random.choice(list(legal_actions))

        # Apply action
        self.round.take_action(round_action)

        # Round finished?
        if self.round.finished:
            terminated = True
            # Reward is from the perspective of the team of the acting player
            round_points = self.round.round_points  # [team0_points, team1_points]
            team_idx = eplayer_to_team_index(acting_player)
            opp_idx = get_other_team_index(team_idx)

            win_points = round_points[team_idx]
            loss_points = round_points[opp_idx]

            # Positive if this player's team did better, negative otherwise
            reward += (win_points - loss_points) * PER_WON_POINT

            # No further actions – we can just keep obs/mask zeros
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
            info["action_mask"] = mask
            self.last_action_mask = mask
            return obs, float(reward), terminated, truncated, info

        # Otherwise, continue the round; next player will act next step
        self.current_player = self.round.get_current_player()
        obs, mask = self._get_obs_and_mask()
        info["action_mask"] = mask
        self.last_action_mask = mask

        return obs, float(reward), terminated, truncated, info

    def render(self):
        # You can add a textual debug renderer if you want
        pass

    def close(self):
        pass
