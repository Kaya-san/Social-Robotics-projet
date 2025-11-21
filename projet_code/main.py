import gymnasium as gym
env = gym.make('Pusher-v5', render_mode="human")   #render_mode="human utilasation du mode par defaut de genysum






#======================== test de l'envirenement =============================>
# Réinitialisation de l’environnement avant de commencer.
# - obs : l’observation initiale de l’environnement (état du robot)
# - info : informations supplémentaires (dépend de l’environnement)
obs, info = env.reset()
# Boucle principale : exécute 200 étapes dans l’environnement.
for _ in range(200):
    # Génère une action aléatoire autorisée par l’environnement.
    # Cela sert uniquement à tester si l’environnement fonctionne.
    action = env.action_space.sample()
    # Applique l’action dans l’environnement :
    # - obs : nouvelle observation
    # - reward : récompense reçue
    # - terminated : True si l’objectif est atteint ou episode fini
    # - truncated : True si limite de temps atteinte
    # - info : informations supplémentaires
    obs, reward, terminated, truncated, info = env.step(action)
    # Si l’épisode est terminé, on réinitialise l’environnement
    if terminated or truncated:
        obs, info = env.reset()
# Ferme proprement l’environnement et la fenêtre graphique.
env.close()