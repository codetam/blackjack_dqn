import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from collections import deque
import numpy as np
import random
import time
import os

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step = self.step)
                self.writer.flush()

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)

    def len(self):
        return len(self.buffer)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        minibatch = random.sample(self.buffer, k=batch_size)
        return minibatch

class DQNAgent:
    def __init__(self, env, loaded_model=None, model_name="model",
                 discount=0.99, learning_rate=0.001, double_dqn=False,
                 epsilon=1, eps_decay=0.99999, eps_min=0.3, 
                 replay_memory_size=10000, min_replay_memory_size=1000,
                 minibatch_size=128, target_update_frequency=500):
        self.env = env
        self.obs_space_size = len(env.observation_space.sample())
        self.model_name = model_name

        # Q-learning parameters
        self.discount = discount
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.double_dqn = double_dqn

        # Epsilon
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
        # Replay buffer
        self.min_replay_memory_size = min_replay_memory_size
        self.replay_buffer = ReplayBuffer(maxlen=replay_memory_size)

        # Main model, this is trained every step
        self.model = self.create_model(loaded_model)
        # Target model, this is used for predictions during training
        self.target_model = self.create_model(loaded_model)
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_frequency = target_update_frequency

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{self.model_name}-{int(time.time())}")
        # it tracks when we're ready to update the target model
        self.target_update_counter = 0 

    def get_model_name(self):
        return self.model_name

    def create_model(self, loaded_model):
        if loaded_model is not None:
            print(f"Loading {loaded_model}")
            model = load_model(loaded_model)
            print(f"Model {loaded_model} loaded")
            # model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        else:
            model = Sequential()
            model.add(Dense(128, activation="relu", input_shape=(self.obs_space_size,)))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(32, activation="relu"))
            model.add(Dense(self.env.action_space.n, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def get_qs(self, states, batch_size=1):
        return self.model.predict(np.reshape(states, (batch_size, self.obs_space_size)), verbose=0)
    
    def create_train_dataset(self, minibatch):
        current_states = np.reshape([transition[0] for transition in minibatch], (self.minibatch_size, self.obs_space_size))
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        new_current_states = np.reshape([transition[3] for transition in minibatch], (self.minibatch_size, self.obs_space_size))
        done = np.array([transition[4] for transition in minibatch])
        done = done + 0

        q_next = self.target_model.predict(new_current_states, verbose=0)
        if self.double_dqn:
            q_eval = self.model.predict(new_current_states, verbose=0)
        else:
            q_eval = q_next
        # We never consider doubling down as an option for the next state
        q_eval = q_eval[:, :-1]
        q_pred = self.model.predict(current_states, verbose=0)
        q_target = np.copy(q_pred)

        max_actions = np.argmax(q_eval, axis=1)
        batch_index = np.arange(self.minibatch_size, dtype = np.int32)

        q_target[batch_index, actions] = rewards + self.discount * q_next[batch_index, max_actions.astype(int)] * (1 - done)
        
        return current_states, q_target

    def train(self, current_state, action, reward, new_state, done):
        # Adds experience to the replay buffer
        self.replay_buffer.add((current_state, action, reward, new_state, done))
        if self.replay_buffer.len() < self.min_replay_memory_size:
            return
        
        # Creates minibatch
        minibatch = self.replay_buffer.sample(self.minibatch_size)

        # Creates training dataset
        X, y = self.create_train_dataset(minibatch)
        
        self.model.fit(X, y, batch_size = self.minibatch_size, 
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if done else None)
        
        if done:
            self.target_update_counter += 1
            # Decay epsilon
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        
        if self.target_update_counter > self.target_update_frequency:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    # Epsilon-greedy policy
    def get_action(self, current_state, greedy=False):
        # The player can double-down only in the first stage of the game
        if current_state[13] == 1:
            qs = self.get_qs(current_state)[0]
            alternatives = self.env.action_space.n
        else:
            qs = self.get_qs(current_state)[0][:2]
            alternatives = self.env.action_space.n - 1
        
        if np.random.random() > self.epsilon or greedy:
            action = np.argmax(qs)
        else:
            action = np.random.randint(0, alternatives)
        return action

    # Validates the network by only selecting the greedy action
    def validate(self, steps):
        total_reward = 0
        for episode in range(1, steps+1):
            current_state = self.env.reset()
            done = False
            while not done:
                action = self.get_action(current_state, greedy=True)
                new_state, reward, done, _ = self.env.step(action)
                current_state = new_state
            total_reward += reward
            
        return total_reward/steps
    
