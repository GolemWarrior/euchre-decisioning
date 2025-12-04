
import time
import numpy as np
import torch
import torch.optim as optim

from multi_agent_rl.multi_agent_play_only_env import EuchreSelfPlayEnv
from state_encoding.multi_agent_play_only_rl import (
    SharedPolicyNet,
    RolloutBuffer,
    Transition,
    ppo_update,
    NUM_ACTIONS,
)


def train_selfplay(
    total_steps: int = 200_000,
    rollout_size: int = 4096,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_ratio: float = 0.2,
    lr: float = 3e-4,
    policy_epochs: int = 4,
    batch_size: int = 512,
    device_str: str = "cpu",
):
    device = torch.device(device_str)

    env = EuchreSelfPlayEnv()

    # Build policy net
    dummy_obs, dummy_info = env.reset()
    obs_dim = dummy_obs.shape[0]
    act_dim = NUM_ACTIONS

    policy = SharedPolicyNet(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=256).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    buffer = RolloutBuffer(max_size=rollout_size + 1, obs_dim=obs_dim, act_dim=act_dim, device=device)

    global_step = 0
    episode_return = 0.0
    episode_len = 0
    episode_count = 0

    obs = dummy_obs
    mask = dummy_info["action_mask"]

    while global_step < total_steps:
        buffer.reset(obs_dim, act_dim)
        buffer.ptr = 0

        # Collect one rollout
        while buffer.ptr < rollout_size and global_step < total_steps:
            action, logprob, value = policy.act(obs, mask, device=device)
            next_obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            tr = Transition(
                obs=obs,
                mask=mask,
                action=action,
                logprob=logprob,
                reward=reward,
                value=value,
                done=done,
            )
            buffer.store(tr)

            episode_return += reward
            episode_len += 1
            global_step += 1

            if done:
                episode_count += 1
                print(
                    f"[Episode {episode_count}] return={episode_return:.2f}, "
                    f"len={episode_len}, steps={global_step}"
                )
                obs, info = env.reset()
                mask = info["action_mask"]
                episode_return = 0.0
                episode_len = 0
            else:
                obs = next_obs
                mask = info["action_mask"]

        # For the GAE bootstrapping, store one extra value for the last state
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_value = policy(obs_t, mask_t)
        buffer.values[buffer.ptr] = float(last_value.item())  # last value

        # PPO update
        ppo_update(
            policy=policy,
            optimizer=optimizer,
            buffer=buffer,
            gamma=gamma,
            lam=lam,
            clip_ratio=clip_ratio,
            policy_epochs=policy_epochs,
            batch_size=batch_size,
        )

    env.close()
    return policy


if __name__ == "__main__":
    # Example call – tweak hyperparams as you like
    trained_policy = train_selfplay(
        total_steps=100_000,
        rollout_size=4096,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.2,
        lr=3e-4,
        policy_epochs=4,
        batch_size=512,
        device_str="cpu",
    )
