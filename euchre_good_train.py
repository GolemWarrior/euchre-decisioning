from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from rl_gym.euchre_good_env import EuchreEnvironment

env = DummyVecEnv([lambda: EuchreEnvironment()])

#TIME_STEPS = 10000 #200000
TIME_STEPS = 200000

# Specify a DQN model
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=0.0001,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=1.0,
    gamma=1.0,  # No discounting
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    verbose=1,
    tensorboard_log="./dqn_euchre_tensorboard/"
)

# Train the DQN model
model.learn(total_timesteps=TIME_STEPS)

model.save("dqn_good_euchre_model")

# Do a quick eval of the model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")