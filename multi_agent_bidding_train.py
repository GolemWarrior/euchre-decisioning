import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from euchre.round import (
    Round,
    ACTIONS,
    EAction,
    PLAYING_STATE,
)
from euchre.players import (
    EPlayer,
    eplayer_to_team_index,
    get_teammate,
)
from euchre.deck import DECK_SIZE


class BiddingPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.logits = nn.Linear(256, act_dim)

    def forward(self, obs):
        x = self.net(obs)
        return self.logits(x)


def encode_bidding_obs(round_obj: Round, player: EPlayer) -> np.ndarray:
    """
    Features:
      - hand one-hot (DECK_SIZE)
      - upcard one-hot (DECK_SIZE)
      - is_dealer (1)
      - estate one-hot (6)
    """
    parts = []

    # Hand one-hot
    hand_vec = np.zeros(DECK_SIZE, dtype=np.float32)
    hand = round_obj.hands[int(player)]
    for card in hand:
        if card is not None:
            hand_vec[int(card)] = 1.0
    parts.append(hand_vec)

    # Upcard one-hot
    up_vec = np.zeros(DECK_SIZE, dtype=np.float32)
    up_vec[int(round_obj.upcard)] = 1.0
    parts.append(up_vec)

    # Dealer flag
    is_dealer = 1.0 if round_obj.dealer == player else 0.0
    parts.append(np.array([is_dealer], dtype=np.float32))

    # Estate one-hot (6)
    estate_vec = np.zeros(6, dtype=np.float32)
    estate_vec[int(round_obj.estate)] = 1.0
    parts.append(estate_vec)

    return np.concatenate(parts, axis=0)


def encode_bidding_mask(round_obj: Round) -> np.ndarray:
    """
    ACTIONS is EAction enum list; mask over ACTIONS indices.
    """
    mask = np.zeros(len(ACTIONS), dtype=np.float32)
    legal = round_obj.get_actions()
    if not legal:
        # fail-safe
        mask[int(EAction["PASS"])] = 1.0
        return mask

    for a in legal:
        mask[int(a)] = 1.0
    return mask


def random_legal_action(round_obj: Round) -> EAction:
    legal = list(round_obj.get_actions())
    if not legal:
        return EAction["PASS"]
    return random.choice(legal)


def play_round_with_bidding(policy: BiddingPolicy, gamma: float = 0.99):
    """
    Team 0 (players 0 & 2) uses this bidding policy.
    Team 1 (players 1 & 3) bids randomly.
    Play phase is fully random for everyone.

    We collect REINFORCE-style trajectories for Team 0 bidding decisions only.
    """
    round_obj = Round()

    logprobs = []
    rewards = []  # per-step (we'll only use final reward really)

    # --- BIDDING PHASE ---
    while round_obj.estate != PLAYING_STATE and not round_obj.finished:
        player = round_obj.get_current_player()
        team = eplayer_to_team_index(player)

        if team == 0:
            # RL bidding decision
            obs_vec = encode_bidding_obs(round_obj, player)
            mask_vec = encode_bidding_mask(round_obj)

            obs_t = torch.tensor(obs_vec, dtype=torch.float32)
            mask_t = torch.tensor(mask_vec, dtype=torch.float32)

            logits = policy(obs_t)
            masked_logits = logits.clone()
            masked_logits[mask_t == 0] = -1e9

            dist = torch.distributions.Categorical(logits=masked_logits)
            action_idx = dist.sample()
            logprob = dist.log_prob(action_idx)

            action_enum = EAction(int(action_idx.item()))
            if action_enum not in round_obj.get_actions():
                # fallback to random legal
                action_enum = random_legal_action(round_obj)

            round_obj.take_action(action_enum)

            logprobs.append(logprob)
            rewards.append(0.0)  # only final reward
        else:
            # Opponents: random legal bidding
            action_enum = random_legal_action(round_obj)
            round_obj.take_action(action_enum)

    # --- PLAYING PHASE (random for all) ---
    if not round_obj.finished and round_obj.estate == PLAYING_STATE:
        while not round_obj.finished:
            legal = list(round_obj.get_actions())
            if not legal:
                break
            action_enum = random.choice(legal)
            round_obj.take_action(action_enum)

    # --- FINAL REWARD ---
    team0_points = round_obj.round_points[0]
    team1_points = round_obj.round_points[1]
    final_reward = float(team0_points - team1_points)

    # Compute returns for each bidding decision
    if len(logprobs) == 0:
        return [], None, final_reward

    returns = []
    G = final_reward
    for _ in reversed(rewards):
        G = 0.0 + gamma * G  # rewards are zero, just discount final
        returns.append(G)
    returns.reverse()

    returns_t = torch.tensor(returns, dtype=torch.float32)
    if len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    return logprobs, returns_t, final_reward


def pg_update(policy: BiddingPolicy, optimizer, logprobs, returns_t):
    logprobs_t = torch.stack(logprobs)
    loss = -(logprobs_t * returns_t).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def main():
    # Determine obs_dim dynamically
    dummy = Round()
    obs_example = encode_bidding_obs(dummy, EPlayer(0))
    obs_dim = len(obs_example)
    act_dim = len(ACTIONS)

    print(f"[BIDDING TRAIN] obs_dim={obs_dim}, act_dim={act_dim}")

    policy = BiddingPolicy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-4)

    import os
    os.makedirs("euchre_models", exist_ok=True)

    num_episodes = 50_000  # you can increase this

    for ep in range(num_episodes):
        logprobs, returns_t, final_reward = play_round_with_bidding(policy)

        if not logprobs:
            continue

        loss = pg_update(policy, optimizer, logprobs, returns_t)

        if ep % 100 == 0:
            print(f"[BIDDING] Episode {ep}, loss={loss:.4f}, final_reward={final_reward:.2f}")
            torch.save(policy.state_dict(), "euchre_models/bidding_policy_latest.pt")

        if ep % 5_000 == 0 and ep > 0:
            torch.save(policy.state_dict(), f"euchre_models/bidding_policy_ep_{ep}.pt")


if __name__ == "__main__":
    main()
