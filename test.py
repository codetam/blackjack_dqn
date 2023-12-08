from blackjack import BlackjackEnv
from baseline import BaselineAgent
from better import BettingAgent
import numpy as np

env = BlackjackEnv(2)
agent = BaselineAgent()
# 
# total_reward = 0
# 
# num_turns = 20
# for i in range(num_turns):
#     current_state = env.reset()
#     done = False
#     env.render()
#     current_stake = 0
#     while not done:
#         action = agent.get_action(current_state)
#         new_state, reward, done, _ = env.step(int(action))
#         env.render(action, done, reward)
#         current_state = new_state
#         total_reward += reward
# 
# print("Average reward: ", total_reward/num_turns)

better = BettingAgent(env=env, playerAgent=agent, stake=1000)
state = np.zeros(13)
state_buffer = better.calculate_bet(state)
i = 1
for state in state_buffer:
    print(str(i) + " [", end="")
    for value in state[:9]:
        print(str(int(value * 8)) + " ", end="")
    print(str(int(state[9] * 32)) + " ", end="")
    print(str(int(state[10] * 21)) + " ", end="")
    print(str(int(state[11] * 10)) + " ", end="")
    print(str(state[12]), end="")
    print("]\n")
    i += 1