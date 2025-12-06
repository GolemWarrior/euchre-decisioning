import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

from euchre.round import (
    Round,
    PLAYING_STATE,
    PLAY_CARD_ACTIONS,
)
from euchre.players import (
    EPlayer,
    eplayer_to_team_index,
    get_teammate,
)
from euchre.deck import DECK_SIZE, ECard

from state_encoding.multi_agent_play_only_rl import encode_state, encode_playable

class PlayPolicy(nn.Module):
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


def random_legal_bidding_action(round_obj: Round):
    legal = list(round_obj.get_actions())
    if not legal:
        # shouldn't really happen here, but fail-safe
        from euchre.round import PASS
        return PASS
    return random.choice(legal)


def random_legal_play_action(round_obj: Round):
    legal = list(round_obj.get_actions())
    if not legal:
        return None
    return random.choice(legal)


def sample_play_action(policy: PlayPolicy, round_obj: Round, player: EPlayer):
    """
    Use encode_state + encode_playable to pick a card (ECard index),
    then map to PLAY_CARD_ACTIONS based on its position in hand.
    """
    obs_vec = encode_state(round_obj, agent_player=player)
    mask_vec = encode_playable(round_obj, agent_player=player)

    obs_t = torch.tensor(obs_vec, dtype=torch.float32)
    logits = policy(obs_t)

    mask_t = torch.tensor(mask_vec, dtype=torch.float32)
    masked_logits = logits.clone()
    masked_logits[mask_t == 0] = -1e9

    dist = torch.distributions.Categorical(logits=masked_logits)
    card_idx = dist.sample().item()
    logprob = dist.log_prob(torch.tensor(card_idx))

    chosen_card = ECard(card_idx)
    hand = round_obj.hands[int(player)]

    if chosen_card not in hand:
        # fallback to random legal
        action_enum = random_legal_play_action(round_obj)
        return action_enum, logprob

    hand_pos = hand.index(chosen_card)
    action_enum = PLAY_CARD_ACTIONS[hand_pos]

    if action_enum not in round_obj.get_actions():
        action_enum = random_legal_play_action(round_obj)

    return action_enum, logprob


def play_round_with_play_policy(policy: PlayPolicy, gamma: float = 0.99):
    """
    - Bidding phase: completely random for both teams.
    - Play phase:
        * Team 0 (players 0 & 2) uses this policy.
        * Team 1 (players 1 & 3) plays random legal.
    Reward: team0_points - team1_points.
    """
    round_obj = Round()

    # --- BIDDING (random) ---
    from euchre.round import PLAYING_STATE
    while round_obj.estate != PLAYING_STATE and not round_obj.finished:
        action_enum = random_legal_bidding_action(round_obj)
        round_obj.take_action(action_enum)

    logprobs = []
    rewards = []

    # --- PLAYING PHASE ---
    while not round_obj.finished and round_obj.estate == PLAYING_STATE:
        current_player = round_obj.get_current_player()
        team = eplayer_to_team_index(current_player)

        # Handle "going alone" partner skipping, if needed:
        if round_obj.going_alone and round_obj.maker is not None:
            if current_player == get_teammate(round_obj.maker):
                # partner of maker auto-plays random, since they don't "learn"
                action_enum = random_legal_play_action(round_obj)
                round_obj.take_action(action_enum)
                continue

        if team == 0:
            # RL play
            action_enum, logprob = sample_play_action(policy, round_obj, current_player)
            if action_enum is None:
                break
            round_obj.take_action(action_enum)
            logprobs.append(logprob)
            rewards.append(0.0)
        else:
            # Opponent: random
            action_enum = random_legal_play_action(round_obj)
            if action_enum is None:
                break
            round_obj.take_action(action_enum)

    # --- FINAL REWARD ---
    team0_points = round_obj.round_points[0]
    team1_points = round_obj.round_points[1]
    final_reward = float(team0_points - team1_points)

    if len(logprobs) == 0:
        return [], None, final_reward

    returns = []
    G = final_reward
    for _ in reversed(rewards):
        G = 0.0 + gamma * G
        returns.append(G)
    returns.reverse()

    returns_t = torch.tensor(returns, dtype=torch.float32)
    if len(returns_t) > 1:
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

    return logprobs, returns_t, final_reward


def pg_update(policy: PlayPolicy, optimizer, logprobs, returns_t):
    logprobs_t = torch.stack(logprobs)
    loss = -(logprobs_t * returns_t).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    dummy = Round()
    # Need dummy in PLAYING_STATE so encode_state doesn't assert
    from euchre.round import PLAYING_STATE
    # Do random bidding until playing or finished
    while dummy.estate != PLAYING_STATE and not dummy.finished:
        dummy.take_action(random_legal_bidding_action(dummy))
    # Play one random legal card so any "played_ecards[0]"-based encodings are safe
    if not dummy.finished and dummy.estate == PLAYING_STATE:
        if dummy.get_actions():
            dummy.take_action(random.choice(list(dummy.get_actions())))

    obs_example = encode_state(dummy, agent_player=EPlayer(0))
    obs_dim = len(obs_example)
    act_dim = DECK_SIZE

    print(f"[PLAY TRAIN] obs_dim={obs_dim}, act_dim={act_dim}")

    policy = PlayPolicy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    import os
    os.makedirs("euchre_models", exist_ok=True)

    num_episodes = 200_000

    for ep in range(num_episodes):
        logprobs, returns_t, final_reward = play_round_with_play_policy(policy)

        if not logprobs:
            continue

        loss = pg_update(policy, optimizer, logprobs, returns_t)

        if ep % 100 == 0:
            print(f"[PLAY] Episode {ep}, loss={loss:.4f}, final_reward={final_reward:.2f}")
            torch.save(policy.state_dict(), "euchre_models/play_policy_latest.pt")

        if ep % 5_000 == 0 and ep > 0:
            torch.save(policy.state_dict(), f"euchre_models/play_policy_ep_{ep}.pt")


if __name__ == "__main__":
    main()
