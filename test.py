from blackjack import BlackjackEnv, BlackjackWithBetEnv
from dqn import DQNAgent
from better import BettingAgent

blackjack_env = BlackjackEnv(2)
agent = DQNAgent(env=blackjack_env, epsilon=0, loaded_model="models/128x64_noexp_2")
betting_env = BlackjackWithBetEnv(blackjackEnv=blackjack_env, playerAgent=agent, max_bet=10_000)
better = BettingAgent(env=betting_env, dqnAgent=agent, max_bet=100)

total_reward = 0
num_turns = 13
if better is not None:
    for i in range(num_turns):
        current_state = betting_env.reset()
        done = False
        betting_env.render()
        print("Expected return: ", better.get_expected_return(current_state))

        action = better.get_action(current_state)
        new_state, reward, done, _ = betting_env.step(int(action), render_steps=True)
        current_state = new_state
        total_reward += reward
else:
    for i in range(num_turns):
        current_state = blackjack_env.reset()
        done = False
        blackjack_env.render()
        current_stake = 0
        while not done:
            action = agent.get_action(current_state)
            new_state, reward, done, _ = blackjack_env.step(int(action))
            blackjack_env.render(action, done, reward)
            current_state = new_state
            total_reward += reward

print("Money lost/won: ", total_reward)
