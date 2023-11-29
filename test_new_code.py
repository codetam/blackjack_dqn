from blackjack import BlackjackEnv
import numpy as np

env = BlackjackEnv(2)

current_state=env.reset()
print(current_state)
print(np.reshape(current_state, (1,13)))
print(env.observation_space.n)