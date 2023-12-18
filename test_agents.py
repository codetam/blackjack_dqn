from blackjack_envs import BlackjackEnv
import matplotlib.pyplot as plt

test_baseline = False
test_dqn = False
all = True

def test_agent(agent, blackjack_env, num_turns):
    plays = []
    rewards = []
    avg_rewards = []
    num_plays = 0
    avg_reward = 0
    plt.show()
    ax = plt.gca()
    for i in range(num_turns):
        current_state = blackjack_env.reset()
        done = False
        blackjack_env.render()
        while not done:
            action = agent.get_action(current_state)
            new_state, reward, done, _ = blackjack_env.step(int(action))
            blackjack_env.render(action, done, reward)
            current_state = new_state
        
        num_plays += 1
        plays.append(num_plays)
        rewards.append(reward)

        avg_reward = sum(rewards)
        avg_reward /= num_plays
        avg_rewards.append(avg_reward)
        
        ax.clear()
        ax.set_xlabel('number of plays')
        ax.set_ylabel('average reward per game')
        ax.plot(plays, avg_rewards)
        plt.pause(0.1) 

def end_play(agent, env, render = False):
    current_state = env.reset()
    done = False
    while not done:
        action = agent.get_action(current_state)
        new_state, reward, done, _ = env.step(int(action))
        if render:
            env.render(action, done, reward)
        current_state = new_state
    return reward

def print_plot(agent_data, avg_reward, subplot, title, text_pos, line_color):
    plt.subplot(subplot)
    ax = plt.gca()
    fig = plt.gcf()
    fig.subplots_adjust(right=0.7)
    fig.text(0.8,text_pos,f'{title}\nAverage: {str(avg_reward)}')
    ax.clear()
    ax.set_xlabel('Number of plays')
    ax.set_ylabel('Average reward per game')
    ax.plot(plays, agent_data['avg_rewards'], color=line_color)

env_1 = BlackjackEnv(2)
env_2 = BlackjackEnv(2)
env_3 = BlackjackEnv(2)

DQN_model = 'models/128x64x32_val=-0.165_1702903139.model'

num_turns = 1000
if test_baseline:
    from agents.baseline import BaselineAgent
    agent = BaselineAgent()
    test_agent(agent=agent, blackjack_env=env_2, num_turns=num_turns)
if test_dqn:
    from agents.dqn import DQNAgent
    agent = DQNAgent(env=env_1, epsilon=0, loaded_model=DQN_model)
    test_agent(agent=agent, blackjack_env=env_2, num_turns=num_turns)
if all:
    from agents.baseline import BaselineAgent
    from agents.dqn import DQNAgent
    from agents.random_agent import RandomAgent
    baseline_agent = BaselineAgent()
    dqn_agent = DQNAgent(env=env_1, epsilon=0, loaded_model=DQN_model)
    random_agent = RandomAgent()
    plays = []
    num_plays = 0
    baseline_data = {'rewards': [], 'avg_rewards': []}
    dqn_data = {'rewards': [], 'avg_rewards': []}
    random_data = {'rewards': [], 'avg_rewards': []}
    plt.show()
    for i in range(num_turns):
        rew1 = end_play(agent=baseline_agent, env=env_1)
        rew2 = end_play(agent=dqn_agent, env=env_2)
        rew3 = end_play(agent=random_agent, env=env_3)

        num_plays += 1
        plays.append(num_plays)

        baseline_data['rewards'].append(rew1)
        dqn_data['rewards'].append(rew2)
        random_data['rewards'].append(rew3)

        avg_reward1 = sum(baseline_data['rewards'])
        avg_reward1 /= num_plays
        baseline_data['avg_rewards'].append(avg_reward1)

        avg_reward2 = sum(dqn_data['rewards'])
        avg_reward2 /= num_plays
        dqn_data['avg_rewards'].append(avg_reward2)

        avg_reward3 = sum(random_data['rewards'])
        avg_reward3 /= num_plays
        random_data['avg_rewards'].append(avg_reward3)

        plt.subplot(311)
        fig = plt.gcf()
        for txt in fig.texts:
            txt.set_visible(False)
        
        print_plot(baseline_data, avg_reward1, 311, 'Baseline Agent', 0.75, 'blue')
        print_plot(dqn_data, avg_reward2, 312, 'DQN Agent', 0.5, 'red')
        print_plot(random_data, avg_reward3, 313, 'Random Agent', 0.25, 'black')
        
        plt.pause(0.01) 