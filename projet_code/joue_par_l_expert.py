import gymnasium as gym
from stable_baselines3 import SAC

# Environnement avec rendu
env = gym.make("Pusher-v5", render_mode="human")

# ⚠️ Hack : le TimeLimit par défaut a _max_episode_steps = 100
# on le remplace par une valeur plus grande
env._max_episode_steps = 300  # ou 500, 1000, etc.

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

        if terminated:
            print(f"→ Terminated à t={t} (condition logique de l'env)")
        if truncated:
            print(f"→ Truncated à t={t} (TimeLimit atteint)")

        done = terminated or truncated

    print(f"Épisode {ep+1} terminé en {t} steps, retour = {ep_reward:.2f}")

env.close()
