# multi_agent_play_only_rl.py

import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Categorical

# Adjust these imports depending on your package layout
from euchre.deck import DECK_SIZE, ESuit
from euchre.round import EAction, RoundEState, ROUND_STATES
from euchre.players import EPlayer, eplayer_to_team_index, get_other_team_index, PLAYER_COUNT


# =========================
#   STATE ENCODING
# =========================

NUM_ACTIONS = len(EAction.__members__)          # All possible actions
NUM_STATES = len(ROUND_STATES)                  # FIRST_BIDDING, ..., PLAYING
NUM_SUITS = len(ESuit.__members__)


def _one_hot(index: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        v[index] = 1.0
    return v


def encode_in_hand(round_obj, agent_player: EPlayer) -> np.ndarray:
    """
    One-hot over cards the agent currently holds.
    Shape: (DECK_SIZE,)
    """
    vec = np.zeros(DECK_SIZE, dtype=np.float32)
    hand = round_obj.hands[int(agent_player)]
    for card in hand:
        if card is not None:
            vec[int(card)] = 1.0
    return vec


def encode_seen(round_obj, agent_player: EPlayer) -> np.ndarray:
    """
    Cards that are publicly visible to everyone or known to this player.
    - All played cards (completed/current tricks)
    - Upcard
    - Discarded card (only if this player is the dealer)
    Shape: (DECK_SIZE,)
    """
    vec = np.zeros(DECK_SIZE, dtype=np.float32)

    # Played cards
    for c in round_obj.played_ecards:
        if c is not None:
            vec[int(c)] = 1.0

    # Upcard (visible to all)
    if getattr(round_obj, "upcard", None) is not None:
        vec[int(round_obj.upcard)] = 1.0

    # Discarded card – only the dealer knows
    if getattr(round_obj, "discarded_card", None) is not None:
        if round_obj.dealer == agent_player:
            vec[int(round_obj.discarded_card)] = 1.0

    return vec


def encode_trump(round_obj) -> np.ndarray:
    """
    One-hot trump suit, or all zeros if not chosen yet.
    Shape: (NUM_SUITS,)
    """
    trump = getattr(round_obj, "trump_esuit", None)
    if trump is None:
        return np.zeros(NUM_SUITS, dtype=np.float32)
    return _one_hot(int(trump), NUM_SUITS)


def encode_round_state(round_obj) -> np.ndarray:
    """
    One-hot of Round state (FIRST_BIDDING, PLAYING, etc).
    Shape: (NUM_STATES,)
    """
    estate = getattr(round_obj, "estate", None)
    if estate is None:
        return np.zeros(NUM_STATES, dtype=np.float32)
    return _one_hot(int(estate), NUM_STATES)


def encode_trick_number(round_obj) -> np.ndarray:
    """
    Trick number / 5, so it's in [0,1].
    Shape: (1,)
    """
    tn = getattr(round_obj, "trick_number", 0)
    return np.array([tn / 5.0], dtype=np.float32)


def encode_trick_wins(round_obj, agent_player: EPlayer) -> np.ndarray:
    """
    Tricks won by:
      - agent's team
      - opponents
    both normalized by TRICK_COUNT (=5)
    Shape: (2,)
    """
    trick_wins = getattr(round_obj, "trick_wins", [0, 0])
    team_idx = eplayer_to_team_index(agent_player)
    opp_idx = get_other_team_index(team_idx)
    return np.array(
        [
            trick_wins[team_idx] / 5.0,
            trick_wins[opp_idx] / 5.0,
        ],
        dtype=np.float32,
    )


def encode_player_id(agent_player: EPlayer) -> np.ndarray:
    """
    One-hot for which player index (0..3).
    Shape: (PLAYER_COUNT,)
    """
    return _one_hot(int(agent_player), PLAYER_COUNT)


def encode_state(round_obj, agent_player: EPlayer) -> np.ndarray:
    """
    Full observation vector for the currently acting seat.

    You can absolutely add more features later (led suit, trump counts,
    particle-filter beliefs, etc.) – this is just a reasonably rich baseline.

    Final shape:
      in_hand          : DECK_SIZE
      seen             : DECK_SIZE
      trump suit       : NUM_SUITS
      round state      : NUM_STATES
      trick number     : 1
      trick wins       : 2
      player id        : PLAYER_COUNT

      total = 2*DECK_SIZE + NUM_SUITS + NUM_STATES + 1 + 2 + PLAYER_COUNT
    """
    return np.concatenate(
        [
            encode_in_hand(round_obj, agent_player),
            encode_seen(round_obj, agent_player),
            encode_trump(round_obj),
            encode_round_state(round_obj),
            encode_trick_number(round_obj),
            encode_trick_wins(round_obj, agent_player),
            encode_player_id(agent_player),
        ],
        axis=0,
    ).astype(np.float32)


def encode_playable(round_obj, agent_player: EPlayer) -> np.ndarray:
    """
    Action mask over ALL actions in EAction (by IntEnum value).
    1.0 where legal, 0.0 otherwise.
    Shape: (NUM_ACTIONS,)
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    legal = round_obj.get_actions()
    for a in legal:
        mask[int(a)] = 1.0
    return mask


# =========================
#   POLICY & VALUE NET
# =========================

class SharedPolicyNet(nn.Module):
    """
    Shared policy + value head that controls all 4 seats.
    Uses action masking by adding log(mask) to logits.
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor | None = None):
        """
        obs: (B, obs_dim)
        action_mask: (B, act_dim) in {0,1}, or None
        """
        x = self.backbone(obs)
        logits = self.policy_head(x)  # (B, act_dim)

        if action_mask is not None:
            # Avoid log(0) by eps shifting
            eps = 1e-8
            logits = logits + torch.log(action_mask + eps)

        value = self.value_head(x).squeeze(-1)  # (B,)
        return logits, value

    def act(self, obs: np.ndarray, mask: np.ndarray, device: torch.device):
        """
        Single-step act() helper for interaction with the env.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.forward(obs_t, mask_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        return (
            int(action.item()),
            float(logprob.item()),
            float(value.item()),
        )


# =========================
#   ROLLOUT BUFFER (PPO)
# =========================

@dataclass
class Transition:
    obs: np.ndarray
    mask: np.ndarray
    action: int
    logprob: float
    reward: float
    value: float
    done: bool


class RolloutBuffer:
    def __init__(self, max_size: int, obs_dim: int, act_dim: int, device: torch.device):
        self.max_size = max_size
        self.device = device
        self.reset(obs_dim, act_dim)

    def reset(self, obs_dim: int, act_dim: int):
        self.obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.masks = np.zeros((self.max_size, act_dim), dtype=np.float32)
        self.actions = np.zeros(self.max_size, dtype=np.int64)
        self.logprobs = np.zeros(self.max_size, dtype=np.float32)
        self.rewards = np.zeros(self.max_size, dtype=np.float32)
        self.values = np.zeros(self.max_size, dtype=np.float32)
        self.dones = np.zeros(self.max_size, dtype=np.float32)
        self.ptr = 0

    def store(self, tr: Transition):
        assert self.ptr < self.max_size, "RolloutBuffer overflow – increase max_size"
        self.obs[self.ptr] = tr.obs
        self.masks[self.ptr] = tr.mask
        self.actions[self.ptr] = tr.action
        self.logprobs[self.ptr] = tr.logprob
        self.rewards[self.ptr] = tr.reward
        self.values[self.ptr] = tr.value
        self.dones[self.ptr] = float(tr.done)
        self.ptr += 1

    def compute_returns_advantages(self, gamma: float, lam: float):
        size = self.ptr
        returns = np.zeros(size, dtype=np.float32)
        advantages = np.zeros(size, dtype=np.float32)
        last_adv = 0.0
        last_ret = 0.0

        for t in reversed(range(size)):
            mask = 1.0 - self.dones[t]
            last_ret = self.rewards[t] + gamma * last_ret * mask
            delta = self.rewards[t] + gamma * self.values[t + 1] * mask - self.values[t] if t + 1 < size else \
                self.rewards[t] - self.values[t]
            last_adv = delta + gamma * lam * last_adv * mask

            returns[t] = last_ret
            advantages[t] = last_adv

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return (
            torch.as_tensor(self.obs[:size], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.masks[:size], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.actions[:size], dtype=torch.int64, device=self.device),
            torch.as_tensor(self.logprobs[:size], dtype=torch.float32, device=self.device),
            torch.as_tensor(returns, dtype=torch.float32, device=self.device),
            torch.as_tensor(advantages, dtype=torch.float32, device=self.device),
        )


def ppo_update(
    policy: SharedPolicyNet,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    gamma: float,
    lam: float,
    clip_ratio: float = 0.2,
    policy_epochs: int = 4,
    batch_size: int = 256,
):
    obs, masks, acts, old_logps, returns, adv = buffer.compute_returns_advantages(gamma, lam)
    num_samples = obs.shape[0]
    indices = np.arange(num_samples)

    for _ in range(policy_epochs):
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            idx = indices[start:start + batch_size]
            batch_obs = obs[idx]
            batch_masks = masks[idx]
            batch_acts = acts[idx]
            batch_old_logps = old_logps[idx]
            batch_returns = returns[idx]
            batch_adv = adv[idx]

            logits, values = policy(batch_obs, batch_masks)
            dist = Categorical(logits=logits)
            logps = dist.log_prob(batch_acts)

            ratio = torch.exp(logps - batch_old_logps)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * (batch_returns - values).pow(2).mean()
            entropy_loss = -dist.entropy().mean() * 0.01

            loss = policy_loss + value_loss + entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
