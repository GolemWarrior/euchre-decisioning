from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from play_only_env import EuchreEnvironment
import torch
import random

def make_env():
    return Monitor(EuchreEnvironment())
env = DummyVecEnv([make_env])

#TIME_STEPS = 10000
TIME_STEPS = 200000
#TIME_STEPS = 3000000  # About 20 mins, "Mean reward: 2.6364 +/- 15.607512134866337"
#TIME_STEPS = 8000000  # About 55 mins, "Mean reward: 2.89266 +/- 15.559961379270836"
#TIME_STEPS = 20000000 # About 2.5 hrs, "Mean reward: 3.5897 +/- 15.56185380698585"

SELF_PLAY_UPDATE_FREQ = 20000

class SelfPlayUpdateCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.pool = []

    def _on_step(self):
        if self.num_timesteps % SELF_PLAY_UPDATE_FREQ == 0:
            # Add snapshot
            snapshot = self.model.__class__.load(self.model.save("tmp.zip"))
            self.pool.append(snapshot)

            # Random agents
            self.training_env.env_method("set_agents", [random.choice(self.pool), random.choice(self.pool), random.choice(self.pool)])

        return True

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
            pi=[256, 32],  # Policy
            vf=[256, 32]   # Value
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