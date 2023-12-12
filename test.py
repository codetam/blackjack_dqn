from blackjack_envs import BlackjackEnv, BlackjackWithBetEnv
import matplotlib.pyplot as plt
import matplotlib as mpl

test_bet = False
test_baseline = True

blackjack_env = BlackjackEnv(2)

num_turns = 1000
if test_bet:
    from agents.dqn import DQNAgent
    from agents.better import BettingAgent
    agent = DQNAgent(env=blackjack_env, epsilon=0, loaded_model="models/128x64_noexp_2")
    betting_env = BlackjackWithBetEnv(blackjackEnv=blackjack_env, playerAgent=agent, max_bet=10_000)
    better = BettingAgent(env=betting_env, dqnAgent=agent, max_bet=100)
    for i in range(num_turns):
        current_state = betting_env.reset()
        done = False
        betting_env.render()
        print("Expected return: ", better.get_expected_return(current_state))

        action = better.get_action(current_state)
        new_state, reward, done, _ = betting_env.step(int(action), render_steps=True)
        current_state = new_state
if test_baseline:
    from agents.baseline import BaselineAgent
    agent = BaselineAgent()

    plays = []
    avg_rewards = []
    num_plays = 0
    avg_reward = 0
    plt.show()
    ax = plt.gca()
    for i in range(num_turns):
        current_state = blackjack_env.reset()
        done = False
        # blackjack_env.render()
        while not done:
            action = agent.get_action(current_state)
            new_state, reward, done, _ = blackjack_env.step(int(action))
            # blackjack_env.render(action, done, reward)
            current_state = new_state
        
        num_plays += 1
        avg_reward += reward
        avg_reward /= num_plays

        plays.append(num_plays)
        avg_rewards.append(avg_reward)
        
        ax.clear()
        ax.set_xlabel('number of plays')
        ax.set_ylabel('average reward per game')
        ax.plot(plays, avg_rewards)
        plt.pause(0.1) 
