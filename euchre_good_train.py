from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from rl_gym.euchre_good_env import EuchreEnvironment
import torch

def make_env():
    return Monitor(EuchreEnvironment())
env = DummyVecEnv([make_env])

#TIME_STEPS = 10000 #200000
#TIME_STEPS = 200000
#TIME_STEPS = 3000000  # About 20 mins, "Mean reward: 2.6364 +/- 15.607512134866337"
#TIME_STEPS = 8000000  # About 55 mins, "Mean reward: 2.89266 +/- 15.559961379270836"
TIME_STEPS = 20000000 # About 2.5 hrs, "Mean reward: 3.5897 +/- 15.56185380698585"

# Specify a DQN model
# (POTENTIALLY) TODO: Ideally would wrap the DQN to add action masking so only legal actions can be selected
# model = DQN(
#     policy="MlpPolicy",
#     env=env,
#     learning_rate=0.00001,
#     buffer_size=2000000,  # Have a large buffer since the state space is large
#     learning_starts=50000,  # Initially fill the buffer around 1%-10% before starting learning
#     batch_size=64,
#     tau=0.01,  # Soft update b|c high dimensional space
#     gamma=1,  # No discounting: game has end goal and game is relatively short
#     train_freq=8,
#     #target_update_interval=1000,  # Ignored when tau < 1
#     exploration_fraction=0.2,
#     exploration_final_eps=0.005,
#     verbose=2,
#     tensorboard_log="./dqn_euchre_tensorboard/",
#     policy_kwargs=dict(net_arch=[256, 128], activation_fn=torch.nn.ReLU),  # ReLU is probably good because the state vector is sparse
# )

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

model = MaskablePPO(
    MaskableActorCriticPolicy,
    env,

    learning_rate=0.00001,
    
    n_steps=512*4,
    batch_size=128*2,
    n_epochs=int(10/2),
    
    gamma=0.999,
    gae_lambda=0.95,
    clip_range=0.2,
    clip_range_vf=None,
    
    ent_coef=0.02,
    vf_coef=0.5,
    max_grad_norm=0.5,
    
    verbose=2,

    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 128, 64],  # Policy
            vf=[128, 64, 16]    # Value
        ),
        activation_fn=torch.nn.ReLU,
    ),
)

# Train the model
model.learn(total_timesteps=TIME_STEPS, log_interval=1)

model.save("dqn_good_euchre_model")

model.load("dqn_good_euchre_model")

# Do a quick eval of the model
env = DummyVecEnv([make_env])
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100000, deterministic=True)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

env = DummyVecEnv([make_env])
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100000, deterministic=False)
print(f"Mean (non-deterministic) reward: {mean_reward} +/- {std_reward}")