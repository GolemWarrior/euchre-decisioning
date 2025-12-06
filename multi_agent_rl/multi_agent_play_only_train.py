import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from multi_agent_rl.multi_agent_play_only_env import EuchreMultiAgentEnv
from state_encoding.multi_agent_play_only_rl import encode_state, encode_playable


# ============================
#  PPO Model
# ============================
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.logits = nn.Linear(256, act_dim)
        self.value = nn.Linear(256, 1)

    def forward(self, obs):
        x = self.shared(obs)
        return self.logits(x), self.value(x)


# ============================
#  Masked Categorical Sampling
# ============================
def masked_sample(logits, mask):
    """
    logits: [1, act_dim]
    mask:   [1, act_dim] with 0/1
    """
    masked_logits = logits.clone()
    masked_logits[mask == 0] = -1e9
    dist = torch.distributions.Categorical(logits=masked_logits)
    action = dist.sample()
    logprob = dist.log_prob(action)
    return action.item(), logprob, dist


# ============================
#  Rollout Collection
# ============================
def generate_episode(env, policy, device="cpu"):
    obs_list, act_list, logprob_list, rew_list, val_list = [], [], [], [], []

    env.reset()
    done = False

    while not all(env.terminations.values()):
        agent = env.agent_selection
        step_obs = env.observe(agent)

        obs_vec = torch.tensor(step_obs["observation"], dtype=torch.float32).to(device)
        mask_vec = torch.tensor(step_obs["action_mask"], dtype=torch.float32).to(device)

        logits, value = policy(obs_vec)

        action, logprob, dist = masked_sample(logits, mask_vec)

        # Step environment
        obs_dict, rewards, terms, truncs, infos = env.step(action)

        obs_list.append(obs_vec)
        act_list.append(action)
        logprob_list.append(logprob)
        rew_list.append(rewards[agent])
        val_list.append(value)

        if all(terms.values()):
            break

    return {
        "obs": torch.stack(obs_list),
        "acts": torch.tensor(act_list),
        "logprobs": torch.stack(logprob_list),
        "rewards": torch.tensor(rew_list),
        "values": torch.stack(val_list).squeeze(),
    }


# ============================
#  PPO Update Step
# ============================
def ppo_update(policy, optimizer, batch, clip_ratio=0.2, gamma=0.99):
    rewards = batch["rewards"]
    values = batch["values"]
    logprobs_old = batch["logprobs"]

    # Advantages & returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    returns = torch.tensor(returns, dtype=torch.float32)

    advantages = returns - values.detach()

    # Compute logprobs under current policy
    logits, values_new = policy(batch["obs"])
    dist = torch.distributions.Categorical(logits=logits)
    logprobs_new = dist.log_prob(batch["acts"])

    # Ratio for PPO
    ratio = torch.exp(logprobs_new - logprobs_old)

    # PPO loss
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -(torch.min(ratio * advantages, clip_adv)).mean()

    value_loss = ((returns - values_new.squeeze()) ** 2).mean()

    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


# ============================
#  Training Loop
# ============================
def main():
    env = EuchreMultiAgentEnv()
    env.reset()
    first_agent = env.possible_agents[0]
    obs_dim = len(env.observe(first_agent)["observation"])
    act_dim = env.action_spaces[first_agent].n

    policy = PolicyNet(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    import os
    os.makedirs("euchre_models", exist_ok=True)

    for episode in range(20_000):
        batch = generate_episode(env, policy)
        loss = ppo_update(policy, optimizer, batch)

        if episode % 50 == 0:
            print(f"Episode {episode}, loss={loss:.4f}, reward={batch['rewards'].sum().item()}")

            # Save latest checkpoint
            torch.save(policy.state_dict(), "euchre_models/policy_latest.pt")

        if episode % 500 == 0 and episode > 0:
            # Save milestone checkpoints
            torch.save(policy.state_dict(), f"euchre_models/policy_ep_{episode}.pt")

if __name__ == "__main__":
    main()
