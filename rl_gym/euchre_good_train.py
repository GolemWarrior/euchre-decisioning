from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from .euchre_good_env import EuchreEnvironment

env = DummyVecEnv([lambda: EuchreEnvironment()])

# Specify a DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./dqn_euchre_tensorboard/"
)

# Train the DQN model
model.learn(total_timesteps=200_000)

model.save("dqn_good_euchre_model")

# Do a quick eval of the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")