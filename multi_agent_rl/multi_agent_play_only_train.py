import time
import numpy as np
import torch
import torch.optim as optim

# Use YOUR environment
from multi_agent_rl.multi_agent_play_only_env import EuchreMultiAgentEnv

# Your neural network + PPO components
from state_encoding.multi_agent_play_only_rl import (
    SharedPolicyNet,
    RolloutBuffer,
    Transition,
    ppo_update,
    NUM_ACTIONS,
)


# ===============================================================
#                    TRAINING FUNCTION
# ===============================================================
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
    checkpoint_interval: int = 50_000,
    save_path: str = "trained_policy.pt",
):

    device = torch.device(device_str)

    # -----------------------------------------------------------
    # Initialize environment
    # -----------------------------------------------------------
    env = EuchreMultiAgentEnv()
    obs, infos = env.reset()
    first_agent = env.agent_selection
    obs_dim = env.observe(first_agent)["observation"].shape[0]
    act_dim = NUM_ACTIONS

    print(f"[ENV READY] Obs dim = {obs_dim}, Action dim = {act_dim}")

    # -----------------------------------------------------------
    # Build policy + optimizer
    # -----------------------------------------------------------
    policy = SharedPolicyNet(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=256).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Create rollout buffer
    buffer = RolloutBuffer(
        max_size=rollout_size + 1,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )

    global_step = 0
    episode_count = 0
    episode_return = 0.0
    episode_len = 0

    print("========== TRAINING BEGIN ==========")

    # ===========================================================
    # Main training loop
    # ===========================================================
    while global_step < total_steps:

        buffer.reset(obs_dim, act_dim)
        buffer.ptr = 0

        # Collect rollout
        while buffer.ptr < rollout_size and global_step < total_steps:

            agent = env.agent_selection
            obs_dict = env.observe(agent)

            obs = obs_dict["observation"].astype(np.float32)
            mask = obs_dict["action_mask"].astype(np.float32)

            action, logprob, value = policy.act(obs, mask, device=device)

            # Step returns (next_obs, rewards, terminations, truncations, infos)
            next_obs, rewards, terminations, truncations, infos = env.step(action)

            reward = rewards[agent]
            done = terminations[agent] or truncations[agent]

            buffer.store(
                Transition(
                    obs=obs,
                    mask=mask,
                    action=action,
                    logprob=logprob,
                    reward=reward,
                    value=value,
                    done=done,
                )
            )

            global_step += 1
            episode_return += reward
            episode_len += 1

            if done:
                episode_count += 1
                print(
                    f"[Episode {episode_count}] "
                    f"Return={episode_return:.2f} Len={episode_len} Step={global_step}"
                )
                env.reset()
                episode_return = 0.0
                episode_len = 0

            if global_step % checkpoint_interval == 0:
                ckpt = f"policy_step_{global_step}.pt"
                torch.save(policy.state_dict(), ckpt)
                print(f"[Checkpoint] Saved: {ckpt}")

        # -------------------------------------------------------
        # Bootstrap final value for GAE
        # -------------------------------------------------------
        agent = env.agent_selection
        last_obs_dict = env.observe(agent)

        obs_last = torch.tensor(
            last_obs_dict["observation"], dtype=torch.float32, device=device
        ).unsqueeze(0)
        mask_last = torch.tensor(
            last_obs_dict["action_mask"], dtype=torch.float32, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            _, last_value = policy(obs_last, mask_last)

        buffer.values[buffer.ptr] = float(last_value.item())

        # -------------------------------------------------------
        # PPO UPDATE
        # -------------------------------------------------------
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

    # ===========================================================
    # Save final policy
    # ===========================================================
    torch.save(policy.state_dict(), save_path)
    print(f"[DONE] Saved final policy to: {save_path}")

    env.close()
    return policy


# ===============================================================
# Entry point
# ===============================================================
if __name__ == "__main__":
    train_selfplay(
        total_steps=100000,
        rollout_size=1024,
        device_str="cpu",
        save_path="trained_policy.pt",
        checkpoint_interval=150_000,
    )

