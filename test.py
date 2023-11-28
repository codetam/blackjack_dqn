from blackjack import BlackjackEnv
from dqn import DQNAgent
import numpy as np
from tqdm import tqdm
import time
import os
import tensorflow as tf
from collections import deque

if not os.path.isdir('models'):
    os.makedirs('models')

TRAIN = False

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

EPISODES = 50_000
# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.1

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# For stats
ep_rewards = [0]

env = BlackjackEnv(2)

agent = DQNAgent(env)

if TRAIN:
    for episode in tqdm(range(1, EPISODES+1), total=EPISODES, position=0, leave=True, ascii=True, unit='episodes'):
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0

        # Reset environment and get initial state
        current_state = env.reset()
        done = False
        episode_memory = deque()
        while not done:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()
            episode_memory.append((current_state, action, reward, new_state, done))

            current_state = new_state

        # Wait for the final reward, then update all previous instances
        final_rewad = reward
        for index, (current_state, action, reward, new_current_state, done) in enumerate(episode_memory):
            agent.update_replay_memory((current_state, action, final_rewad, new_current_state, done))
            agent.train(done)

        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            if episode % 5000 == 0:
                agent.model.save(f'models/128x64__{average_reward:_>7.2f}avg__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)
else:
    for episode in range(15):
        # Reset environment and get initial state
        current_state = env.reset()
        done = False
        episode_memory = deque()
        while not done:
            if np.random.random() > epsilon:
                # Get action from Q table
                action = np.argmax(agent.get_qs(current_state))
            else:
                # Get random action
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, _ = env.step(action)
            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()
            episode_memory.append((current_state, action, reward, new_state, done))

            current_state = new_state

        # Wait for the final reward, then update all previous instances
        final_rewad = reward
        for index, (current_state, action, reward, new_current_state, done) in enumerate(episode_memory):
            print((current_state, action, final_rewad, new_current_state, done))
            agent.update_replay_memory((current_state, action, final_rewad, new_current_state, done))