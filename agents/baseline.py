import numpy as np

blackjack_hard_chart = [
    # dealer's card
    # 2 3  4  5  6  7  8  9  10 A
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # sum = 4
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # sum = 5
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # sum = 6
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # sum = 7
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], # sum = 8
    [1, 2, 2, 2, 2, 1, 1, 1, 1, 1], # sum = 9
    [2, 2, 2, 2, 2, 2, 2, 2, 1, 1], # sum = 10
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2], # sum = 11
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1], # sum = 12
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], # sum = 13
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], # sum = 14
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], # sum = 15
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1], # sum = 16
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sum = 17
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sum = 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sum = 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sum = 20
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # sum = 21
]

blackjack_soft_chart = [
    # dealer's card
    # 2 3  4  5  6  7  8  9  10 A
    [1, 1, 0, 0, 0, 1, 1, 1, 1, 1], # sum = 12
    [1, 1, 1, 2, 2, 1, 1, 1, 1, 1], # sum = 13
    [1, 1, 1, 2, 2, 1, 1, 1, 1, 1], # sum = 14
    [1, 1, 2, 2, 2, 1, 1, 1, 1, 1], # sum = 15
    [1, 1, 2, 2, 2, 1, 1, 1, 1, 1], # sum = 16
    [1, 2, 2, 2, 2, 1, 1, 1, 1, 1], # sum = 17
    [3, 3, 3, 3, 3, 0, 0, 1, 1, 1], # sum = 18
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sum = 19
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # sum = 20
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # sum = 21
]

class BaselineAgent():
    def get_action(self, current_state):
        state = np.array(current_state)
        current_sum = (state[10] * 21).astype(int)
        dealer_card = (state[11] * 11).astype(int)
        has_soft = state[12].astype(int)
        if has_soft:
            chart = blackjack_soft_chart
            action = chart[current_sum - 12][dealer_card - 2]
        else:
            chart = blackjack_hard_chart
            action = chart[current_sum - 4][dealer_card - 2]
        if state[13] != 1:
            if action == 2:
                action = 1
            elif action == 3:
                action = 0
        if action == 3:
            return 2
        return action