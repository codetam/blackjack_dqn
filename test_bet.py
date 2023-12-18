from blackjack_envs import BlackjackEnv, BlackjackWithBetEnv
from agents.dqn import DQNAgent
from agents.better import BettingAgent

import matplotlib.pyplot as plt

blackjack_env = BlackjackEnv(2)
DQN_model = 'models/128x64x32_val=-0.165_1702903139.model'
agent = DQNAgent(env=blackjack_env, epsilon=0, loaded_model=DQN_model)
betting_env = BlackjackWithBetEnv(blackjackEnv=blackjack_env, playerAgent=agent, max_bet=10_000)
better = BettingAgent(env=betting_env, dqnAgent=agent, max_bet=100)

num_turns=1000
stake = 1000
plt.show()
plays = []
stakes = []

for i in range(num_turns):
    current_state = betting_env.reset()
    done = False
    print(f"Stake: ${stake}")
    betting_env.render()
    print("Expected return: ", better.get_expected_return(current_state))

    action = better.get_action(current_state, min_expected_return=0.15)
    new_state, reward, done, _ = betting_env.step(int(action), render_steps=True)
    stake += reward
    current_state = new_state
    
    plays.append(i+1)
    stakes.append(stake)

    ax = plt.gca()
    fig = plt.gcf()
    for txt in fig.texts:
        txt.set_visible(False)
    fig.subplots_adjust(bottom=0.3)
    fig.text(0.2,0.1,f'Stake: ${str(stake)}')

    ax.clear()
    ax.set_xlabel('Number of plays')
    ax.set_ylabel('Stake ($)')
    ax.plot(plays, stakes)
    plt.pause(0.1)