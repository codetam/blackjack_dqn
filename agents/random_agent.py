import numpy as np
import random

class RandomAgent():
    def get_action(self, current_state):
        state = np.array(current_state)
        if state[13] == 1:
            action = random.randint(0,2)
        else:
            action = random.randint(0,1)
        return action