import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC

env = gym.make("Pusher-v5")
model_expert = SAC.load("pusher_expert_sac", env=env)

observations = []
actions = []

n_episodes = 50   # nombre d’épisodes pour le dataset

for _ in range(n_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _ = model_expert.predict(obs, deterministic=True)
        observations.append(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

env.close()

observations = np.array(observations)
actions = np.array(actions)

np.save("expert_obs.npy", observations)
np.save("expert_act.npy", actions)
