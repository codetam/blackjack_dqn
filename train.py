from blackjack_envs import BlackjackEnv
from agents.dqn import DQNAgent
from tqdm import tqdm
import time
import os
import tensorflow as tf
import time

if not os.path.isdir('models'):
    os.makedirs('models')

MODEL_NAME = "256x128x64x32"
LOAD_MODEL = None
TRAIN = True

# os.environ["OMP_NUM_THREADS"] = "16"
# tf.config.threading.set_inter_op_parallelism_threads(8) 
# tf.config.threading.set_intra_op_parallelism_threads(8)
# os.environ["TF_ENABLE_MKL_NATIVE_FORMAT"] = "1"

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

EPISODES = 800_000
#  Stats settings
AGGREGATE_STATS_EVERY = 1000  # episodes
SHOW_PREVIEW = False

# For stats
ep_rewards = [0]

env = BlackjackEnv(2)

agent = DQNAgent(env=env, epsilon=1, eps_decay=0.999995, eps_min=0.01, loaded_model=LOAD_MODEL, model_name=MODEL_NAME)

if TRAIN:
    for episode in tqdm(range(1, EPISODES+1), total=EPISODES, position=0, leave=True, ascii=True, unit='episodes'):
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0

        # Reset environment and get initial state
        current_state = env.reset()
        done = False
        while not done:
            action = agent.get_action(current_state)
            new_state, reward, done, _ = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward
            if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
                env.render()
            agent.train(current_state, action, reward, new_state, done)
            current_state = new_state
        
        # Append episode reward to a list and log stats (every given number of episodes)
        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            validation_reward = agent.validate(200)
            agent.tensorboard.update_stats(epsilon=agent.epsilon, train_reward=average_reward, val_reward=validation_reward)

            # Save model, but only when min reward is greater or equal a set value
            if episode % 50_000 == 0 or validation_reward > 0.2:
                agent.model.save(f'models/{agent.get_model_name()}_val={validation_reward}_{int(time.time())}.model')

        tf.keras.backend.clear_session()