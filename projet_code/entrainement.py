import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Chargement des données expert
obs = np.load("expert_obs.npy")
act = np.load("expert_act.npy")

obs_dim = obs.shape[1]
act_dim = act.shape[1]

# Conversion en tenseurs
obs_tensor = torch.from_numpy(obs).float()
act_tensor = torch.from_numpy(act).float()

# Modèle de Behavioral Cloning : MLP simple
class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )

    def forward(self, x):
        return self.net(x)

policy_bc = BCPolicy(obs_dim, act_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(policy_bc.parameters(), lr=3e-4)

# Entraînement supervisé
n_epochs = 50
batch_size = 256

for epoch in range(n_epochs):
    permutation = torch.randperm(obs_tensor.size(0))

    epoch_loss = 0.0
    for i in range(0, obs_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_obs = obs_tensor[indices]
        batch_act = act_tensor[indices]

        pred_act = policy_bc(batch_obs)
        loss = criterion(pred_act, batch_act)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{n_epochs}, loss = {epoch_loss:.4f}")

# Sauvegarde du modèle
torch.save(policy_bc.state_dict(), "bc_policy_pusher.pt")
