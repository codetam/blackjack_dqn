import numpy as np
from collections import deque
from blackjack import calculate_sum, get_card_num

class BettingAgent():
    def __init__(self, env, playerAgent, stake):
        self.env = env
        self.playerAgent = playerAgent
        self.stake = stake

    def get_value_to_add(self, next_state, i):
        if next_state[i] == 1:
            value_to_add = 0
        else:
            value_to_add = 1 / (4 * self.env.num_decks)
            if i == 9:
                # There are more 10's in the deck
                value_to_add = 1 / (16 * self.env.num_decks)

        return value_to_add
    
    def update_state_buffer(self, iter_number, state, player_cards):
        for i in range(10):
            next_state = np.copy(state)
            value_to_add = self.get_value_to_add(next_state, i)
            if value_to_add == 0:
                continue
            next_state[i] = next_state[i] + value_to_add
            if iter_number == 0:
                # Adds the card to the dealer's hand
                next_state[11] = (i + 1) / 10
            elif iter_number == 1:
                player_cards = np.array([0, 0])
                player_cards[0] = i + 1
            elif iter_number == 2:
                player_cards = np.copy(player_cards)
                player_cards[1] = i + 1
                next_state[10] = calculate_sum(player_cards) / 21
                next_state[12] = 0
                player_card_numbers = np.array([get_card_num(card) for card in player_cards])
                hard_sum = np.sum(player_card_numbers)
                if 1 in player_card_numbers and (hard_sum + 10) <= 21:
                    # The current sum is soft
                    next_state[12] = 1
                self.state_buffer.append(next_state)
            if iter_number < 2:
                self.update_state_buffer(iter_number + 1, next_state, player_cards)

    def calculate_bet(self, current_state):
        state = np.array(current_state)
        self.state_buffer = deque()
        self.update_state_buffer(0, state, np.array([0, 0]))
        return self.state_buffer
        #qs = self.playerAgent.get_qs(states=all_states, batch_size=1)