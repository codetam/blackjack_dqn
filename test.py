from blackjack import BlackjackEnv
from baseline import BaselineAgent

env = BlackjackEnv(2)
agent = BaselineAgent()

for i in range(1, 13):
    current_state = env.reset()
    done = False
    env.render()
    current_stake = 0
    while not done:
        action = agent.get_action(current_state)
        new_state, reward, done, _ = env.step(int(action))
        env.render()
        current_state = new_state
        print("Reward: ", reward)