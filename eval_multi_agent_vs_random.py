import torch
import numpy as np
import random

# Your environment
from multi_agent_rl.multi_agent_play_only_env import EuchreMultiAgentEnv

# Your policy architecture & constants
from state_encoding.multi_agent_play_only_rl import (
    SharedPolicyNet,
    NUM_ACTIONS,
)

from euchre.players import PLAYERS


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===============================================================
# Load trained policy
# ===============================================================
def load_policy(path: str, obs_dim: int, act_dim: int):
    policy = SharedPolicyNet(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=256).to(DEVICE)
    state_dict = torch.load(path, map_location=DEVICE)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


# ===============================================================
# Action selection
# ===============================================================
def select_trained_action(policy, obs, mask):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    mask_t = torch.tensor(mask, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        logits, _ = policy(obs_t, mask_t)

    # Apply mask to logits
    logits = logits.squeeze(0)
    mask_t = mask_t.squeeze(0)
    logits = logits.masked_fill(mask_t < 0.5, -1e9)

    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample().item()
    return action


def select_random_action(mask):
    """Choose uniformly among legal actions."""
    legal = np.where(mask > 0.5)[0]
    if len(legal) == 0:
        return random.randrange(NUM_ACTIONS)
    return int(np.random.choice(legal))


# ===============================================================
# Evaluation function
# ===============================================================
def evaluate(policy_path: str, num_games: int = 500):

    # Create env
    env = EuchreMultiAgentEnv()
    obs, infos = env.reset()
    first_agent = env.agent_selection
    obs_dim = env.observe(first_agent)["observation"].shape[0]
    policy = SharedPolicyNet(obs_dim, NUM_ACTIONS)

    # Load trained policy
    policy = load_policy(policy_path, obs_dim=obs_dim, act_dim=NUM_ACTIONS)

    TEAM_TRAINED = {0, 2}
    TEAM_RANDOM = {1, 3}

    wins = 0
    total_reward = 0.0

    print("========== BEGIN EVALUATION ==========")

    for game_idx in range(1, num_games + 1):
        observations, infos = env.reset()
        done_dict = {agent: False for agent in env.agents}

        game_reward_team = 0.0

        while not all(done_dict.values()):
            current_agent = env.agent_selection
            agent_idx = PLAYERS.index(current_agent)

            obs = observations[current_agent]["observation"]
            mask = observations[current_agent]["action_mask"]

            # Choose action based on agent seat
            if agent_idx in TEAM_TRAINED:
                action = select_trained_action(policy, obs, mask)
            else:
                action = select_random_action(mask)

            # Step
            next_obs, rewards, terminations, truncations, infos = env.step(action)

            # Track reward for trained team
            if agent_idx in TEAM_TRAINED:
                game_reward_team += rewards[current_agent]

            observations = next_obs
            done_dict = {
                agent: terminations[agent] or truncations[agent]
                for agent in env.agents
            }

        # End of game
        total_reward += game_reward_team
        if game_reward_team > 0:
            wins += 1

        if game_idx % 50 == 0:
            print(
                f"[{game_idx}/{num_games}] "
                f"WinRate={wins/game_idx:.3f}  "
                f"AvgReward={total_reward/game_idx:.2f}"
            )

    print("======================================")
    print(f"Evaluated {num_games} games")
    print(f"Final Win Rate: {wins/num_games:.3f}")
    print(f"Final Avg Reward: {total_reward/num_games:.2f}")
    print("======================================\n")


# ===============================================================
# Entry point
# ===============================================================
if __name__ == "__main__":
    evaluate("trained_policy.pt", num_games=5000)
