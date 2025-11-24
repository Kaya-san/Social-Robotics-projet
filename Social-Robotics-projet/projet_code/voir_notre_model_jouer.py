import gymnasium as gym
import torch
import numpy as np
from gymnasium.wrappers import TimeLimit

# 1. Redéfinir la même classe que dans train_bc.py
class BCPolicy(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_dim),
        )

    def forward(self, x):
        return self.net(x)

# 2. Créer l'environnement avec rendu
env = gym.make("Pusher-v5", render_mode="human")

# (optionnel) augmenter la durée max de l'épisode
if hasattr(env, "_max_episode_steps"):
    env._max_episode_steps = 300  # par exemple

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Charger le modèle entraîné
policy = BCPolicy(obs_dim, act_dim).to(device)
policy.load_state_dict(torch.load("bc_policy_pusher.pt", map_location=device))
policy.eval()

n_episodes = 5

for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0.0
    t = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action_tensor = policy(obs_tensor)
        action = action_tensor.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        t += 1

        done = terminated or truncated

    print(f"[BC] Épisode {ep+1} terminé en {t} steps, retour = {ep_reward:.2f}")

env.close()
