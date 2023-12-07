from blackjack import BlackjackEnv
from baseline import BaselineAgent

env = BlackjackEnv(2)
agent = BaselineAgent()

total_reward = 0

num_turns = 20
for i in range(num_turns):
    current_state = env.reset()
    done = False
    env.render()
    current_stake = 0
    while not done:
        action = agent.get_action(current_state)
        new_state, reward, done, _ = env.step(int(action))
        env.render(action, done, reward)
        current_state = new_state
        total_reward += reward

print("Average reward: ", total_reward/num_turns)
