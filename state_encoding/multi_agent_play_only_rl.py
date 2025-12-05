# ============================================================
# multi_agent_play_only_rl.py  (FINAL SIMPLIFIED VERSION)
# ============================================================

import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from euchre.deck import DECK_SIZE
from euchre.round import PLAY_CARD_ACTIONS
from euchre.players import EPlayer


# ============================================================
# CONFIG
# ============================================================

NUM_ACTIONS = len(PLAY_CARD_ACTIONS)  # should be 5


# ============================================================
# STATE ENCODING
# ============================================================

def encode_playable(round_obj, player: EPlayer) -> np.ndarray:
    """
    Action mask over PLAY_CARD_* actions only.
    Shape: (NUM_ACTIONS=5,)
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    legal = round_obj.get_actions()

    for idx, act in enumerate(PLAY_CARD_ACTIONS):
        if act in legal:
            mask[idx] = 1.0
    return mask


def encode_state(round_obj, player: EPlayer) -> np.ndarray:
    """
    Simple, compact play-only state encoding that:
    - encodes hand (onehot)
    - encodes visible tricks
    - encodes trump, dealer, maker, trick index
    - encodes player identity

    Final dim ~137–180 depending on deck size.
    """
    hand = round_obj.hands[player]

    # ----------------------------
    # 1. Hand encoding (one-hot)
    # ----------------------------
    hand_vec = np.zeros(DECK_SIZE, dtype=np.float32)
    for c in hand:
        if c is not None:
            hand_vec[int(c)] = 1.0

    # ----------------------------
    # 2. Current trick cards (4 × DECK_SIZE)
    # ----------------------------
    trick_cards = []
    for c in round_obj.played_ecards:
        one = np.zeros(DECK_SIZE, dtype=np.float32)
        if c is not None:
            one[int(c)] = 1.0
        trick_cards.append(one)

    # pad to 4 cards
    while len(trick_cards) < 4:
        trick_cards.append(np.zeros(DECK_SIZE, dtype=np.float32))

    trick_vec = np.concatenate(trick_cards, axis=0)

    # ----------------------------
    # 3. Meta information
    # ----------------------------
    trick_pos = np.zeros(4, dtype=np.float32)
    tp = len(round_obj.played_ecards)
    if 0 <= tp <= 3:
        trick_pos[tp] = 1.0

    trump_vec = np.zeros(4, dtype=np.float32)
    if round_obj.trump_esuit is not None:
        trump_vec[int(round_obj.trump_esuit)] = 1.0

    maker_vec = np.zeros(4, dtype=np.float32)
    if round_obj.maker is not None:
        maker_vec[int(round_obj.maker)] = 1.0

    dealer_vec = np.zeros(4, dtype=np.float32)
    dealer_vec[int(round_obj.dealer)] = 1.0

    player_vec = np.zeros(4, dtype=np.float32)
    player_vec[int(player)] = 1.0

    # ----------------------------
    # Concatenate everything
    # ----------------------------
    state = np.concatenate(
        [
            hand_vec,
            trick_vec,
            trick_pos,
            trump_vec,
            maker_vec,
            dealer_vec,
            player_vec,
        ],
        axis=0,
    )

    return state.astype(np.float32)


# ============================================================
# POLICY NETWORK (Shared across all 4 seats)
# ============================================================

class SharedPolicyNet(nn.Module):
    """
    Masked PPO policy + value network.
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

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        """
        obs:  (B, obs_dim)
        mask: (B, act_dim) values ∈ {0,1}
        """
        x = self.backbone(obs)
        logits = self.policy_head(x)

        # Masking: illegal → -1e9
        eps = 1e-8
        masked_logits = logits + torch.log(mask + eps)
        masked_logits = masked_logits.masked_fill(mask <= 0.0, -1e9)

        value = self.value_head(x)
        return masked_logits, value

    def act(self, obs_np: np.ndarray, mask_np: np.ndarray, device=torch.device("cpu")):
        """
        Choose an action from single observation.
        """
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        mask = torch.tensor(mask_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.forward(obs, mask)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        return (
            int(action.item()),
            float(logprob.item()),
            float(value.squeeze().item()),
        )


# ============================================================
# ROLLOUT BUFFER
# ============================================================

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
        assert self.ptr < self.max_size, "RolloutBuffer overflow"
        self.obs[self.ptr] = tr.obs
        self.masks[self.ptr] = tr.mask
        self.actions[self.ptr] = tr.action
        self.logprobs[self.ptr] = tr.logprob
        self.rewards[self.ptr] = tr.reward
        self.values[self.ptr] = tr.value
        self.dones[self.ptr] = float(tr.done)
        self.ptr += 1

    def compute_returns_and_advantages(self, gamma: float, lam: float):
        N = self.ptr
        returns = np.zeros(N, dtype=np.float32)
        adv = np.zeros(N, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(N)):
            next_value = self.values[t + 1] if t + 1 < N else 0.0
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            last_gae = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            adv[t] = last_gae
            returns[t] = adv[t] + self.values[t]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        return (
            torch.tensor(self.obs[:N], dtype=torch.float32, device=self.device),
            torch.tensor(self.masks[:N], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[:N], dtype=torch.int64, device=self.device),
            torch.tensor(self.logprobs[:N], dtype=torch.float32, device=self.device),
            torch.tensor(returns, dtype=torch.float32, device=self.device),
            torch.tensor(adv, dtype=torch.float32, device=self.device),
        )


# ============================================================
# PPO UPDATE
# ============================================================

def ppo_update(
    policy: SharedPolicyNet,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    gamma: float,
    lam: float,
    clip_ratio: float = 0.2,
    policy_epochs: int = 4,
    batch_size: int = 256,
):
    obs, masks, actions, old_logps, returns, adv = buffer.compute_returns_and_advantages(gamma, lam)
    N = obs.shape[0]
    idxs = np.arange(N)

    for _ in range(policy_epochs):
        np.random.shuffle(idxs)
        for start in range(0, N, batch_size):
            batch_idx = idxs[start:start + batch_size]

            b_obs = obs[batch_idx]
            b_masks = masks[batch_idx]
            b_actions = actions[batch_idx]
            b_old_logps = old_logps[batch_idx]
            b_returns = returns[batch_idx]
            b_adv = adv[batch_idx]

            logits, values = policy(b_obs, b_masks)
            dist = Categorical(logits=logits)
            logps = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(logps - b_old_logps)

            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * b_adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (b_returns - values.squeeze()).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
