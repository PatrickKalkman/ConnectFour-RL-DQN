import numpy as np
from pettingzoo.classic import connect_four_v3

env = connect_four_v3.env(render_mode="human")
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # The observation space is a 6x7x2 array
        # Channel 0: positions of the current player's pieces (1s)
        # Channel 1: positions of the opponent's pieces (1s)
        mask = observation["action_mask"]
        # Choose a random valid action
        valid_actions = [i for i in range(len(mask)) if mask[i] == 1]
        action = np.random.choice(valid_actions)

    env.step(action)
env.close()
