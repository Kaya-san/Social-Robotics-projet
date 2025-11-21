import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC

# Environnement avec rendu
env = gym.make("Pusher-v5", render_mode="human")

# On augmente le nombre max de steps par épisode (ex : 300 au lieu de 100)
env = TimeLimit(env, max_episode_steps=3000)

model_expert = SAC.load("pusher_expert_sac", env=env)

n_episodes = 5

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    t = 0

    while not done:
        action, _ = model_expert.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        t += 1

        done = terminated or truncated

    print(f"Épisode {ep+1} terminé en {t} steps, retour = {ep_reward:.2f}")

env.close()
