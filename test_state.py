from blackjack_envs import BlackjackEnv
from agents.dqn import DQNAgent
import numpy as np

import matplotlib.pyplot as plt

blackjack_env = BlackjackEnv(2)
DQN_model = 'models/128x64x32_val=-0.165_1702903139.model'
agent = DQNAgent(env=blackjack_env, epsilon=0, loaded_model=DQN_model)

cards_out = np.array([4,5,7,5,2,4,6,2,5,24])
player_sum = 17
dealer_card = 10
soft_hand = 0
first_hand = 1


state = blackjack_env.reset()
def normalize(cards_out, player_sum, dealer_card, soft_hand, first_hand, num_decks):
    # normalization of cards out
    new_state = np.empty((14,), dtype=np.float32)
    new_state[:9] = cards_out[:9] / (4 * num_decks)
    new_state[9] = cards_out[-1] / (16 * num_decks)
    
    # normalization of player sum and dealer card
    new_state[10] = player_sum / 21
    new_state[11] = dealer_card / 10

    new_state[12] = soft_hand
    new_state[13] = first_hand
    return new_state

state = normalize(cards_out, player_sum, dealer_card, soft_hand, first_hand, 2)
print("Agent Q's:")
print(agent.get_qs(state)[0])