import os
import random

import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
#from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from play_only_env import EuchreEnvironment

INITIAL_TRAINING_MODEL = "play_only_euchre_agent_model_start.zip"

def make_env():
    return Monitor(EuchreEnvironment())
env = DummyVecEnv([make_env])

# Training against random play, scored against random play, reward is points scored - opponent points scored (with 0.1 * (tricks won minus opponet tricks won)):
#TIME_STEPS = 3000000  # About 20 mins, "Mean reward: 0.26364 +/- 1.5607512134866337"
#TIME_STEPS = 8000000  # About 55 mins, "Mean reward: 0.289266 +/- 1.5559961379270836"
#TIME_STEPS = 20000000 # About 2.5 hrs, "Mean reward: 0.35897 +/- 1.556185380698585"

# Training against self-play, scoreed against random play, reward is roughly points scored - opponent points scored (with 0.1 * (tricks won minus opponet tricks won)):
#TIME_STEPS = 3000000  # Something like 30 mins, "Mean reward: 0.12175199999999999 +/- 1.5261715665337237"
#TIME_STEPS = 30000000  # Long time, "Mean reward: 0.20163800000000004 +/- 1.40952450030356"
TIME_STEPS = 60000000  # Long time, "Mean reward: 0.23198600000000003 +/- 1.4297022402598383"

SELF_PLAY_UPDATE_FREQ = 1000

class SelfPlayUpdateCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.pool = []

    def _on_step(self):
        if self.num_timesteps % SELF_PLAY_UPDATE_FREQ == 0:
            # Add snapshot
            path = "tmp.zip"
            self.model.save(path)
            snapshot = self.model.__class__.load(path)
            self.pool.append(snapshot)

            # Once there are 500 in the pool, remove a random model each time a new one is added (tends to remove older models)
            if len(self.pool) > 500:
                index = random.randrange(len(self.pool))
                self.pool.pop(index)

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

if __name__ == "__main__":
    if os.path.exists(INITIAL_TRAINING_MODEL):
        model.load(INITIAL_TRAINING_MODEL)

    # Train the model
    model.learn(total_timesteps=TIME_STEPS,
                log_interval=1,
                callback=SelfPlayUpdateCallback())

    model.save("play_only_euchre_model")

    model.load("play_only_euchre_model")

    # Do a quick eval of the model
    env = DummyVecEnv([make_env])
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100000, deterministic=True)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    env = DummyVecEnv([make_env])
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100000, deterministic=False)
    print(f"Mean (non-deterministic) reward: {mean_reward} +/- {std_reward}")

