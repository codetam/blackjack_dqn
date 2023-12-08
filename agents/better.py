import numpy as np
from gamecode import calculate_sum, check_soft_hand

class BettingAgent():
    def __init__(self, env, dqnAgent, max_bet):
        self.num_decks = env.blackjackEnv.num_decks
        self.dqnAgent = dqnAgent
        self.max_bet = max_bet
        self.state_buffer = []
        self.combinations = []

    # Increase the element of the state by the amount returned
    def get_value_to_add(self, next_state, i):
        # Card value won't be considered if there are no more left
        if next_state[i] == 1:
            value_to_add = 0
        else:
            value_to_add = 1 / (4 * self.num_decks)
            if i == 9:
                value_to_add = 1 / (16 * self.num_decks)
        return value_to_add
    
    # Finds the number of possible ways the selected cards can be drawn
    def get_num_combinations(self, state, cards):
        state_denormalized = np.copy(state)
        # Finds the amount of cards drawn
        state_denormalized[0:9] = state_denormalized[0:9] * 4 * self.num_decks
        state_denormalized[9] = state_denormalized[9] * 16 * self.num_decks
        num_combinations = 1
        for card in cards:
            state_denormalized[card - 1] = state_denormalized[card - 1] - 1
            if card == 10:
                total = 16 * self.num_decks
            else:
                total = 4 * self.num_decks
            # the number of combination
            num_combinations *= (total - state_denormalized[card - 1])
        return num_combinations
    
    # Recursive function to fill the state and the combinations buffers
    def update_buffers(self, iter_number, state, player_cards):
        for i in range(10):
            next_state = np.copy(state)
            value_to_add = self.get_value_to_add(next_state, i)
            if value_to_add == 0:
                continue
            next_state[i] = next_state[i] + value_to_add
            if iter_number == 0:
                # Adds the card to the dealer's hand
                next_state[11] = (i + 1) / 10
                self.update_buffers(iter_number + 1, next_state, player_cards)
            elif iter_number == 1:
                # Adds the card to the player's hand
                player_cards = np.array([0, 0])
                player_cards[0] = i + 1
                self.update_buffers(iter_number + 1, next_state, player_cards)
            elif iter_number == 2:
                player_cards = np.copy(player_cards)
                # Adds the card to the player's hand
                player_cards[1] = i + 1
                next_state[10] = calculate_sum(player_cards) / 21
                next_state[12] = check_soft_hand(player_cards)

                cards = [int(next_state[11] * 10), player_cards[0], player_cards[1]]
                num_combinations = self.get_num_combinations(next_state[:10], cards)

                self.state_buffer.append(next_state)
                self.combinations.append(num_combinations)

    def fill_buffers(self, current_state):
        state = np.array(current_state)
        self.state_buffer = []
        self.combinations = []
        self.update_buffers(0, state, np.array([0, 0]))
    
    def get_expected_return(self, current_state):
        self.fill_buffers(current_state)
        qs_list = self.dqnAgent.get_qs(states=np.array(self.state_buffer), batch_size=len(self.state_buffer))
        max_qs = [max(qs) for qs in qs_list]
        expected_return = 0
        for i in range(len(max_qs)):
            expected_return += max_qs[i] * self.combinations[i]
        expected_return /= sum(self.combinations)
        return expected_return
    
    # Returns the bet amount
    def get_action(self, current_state):
        expected_return = self.get_expected_return(current_state)
        if expected_return < 0:
            expected_return = 0
        return min(expected_return * self.max_bet, self.max_bet)