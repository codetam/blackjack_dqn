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

MODEL_NAME="128x64-noexp"
LOAD_MODEL = None

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
    def __init__(self, env, 
                 discount=0.99, learning_rate=0.001,
                 epsilon=1, eps_decay=0.999977, eps_min=0.1, 
                 replay_memory_size=10000, min_replay_memory_size=1000,
                 minibatch_size=256, target_update_frequency=5):
        self.env = env
        self.obs_space_size = len(env.observation_space.sample())

        #q-learning parameters
        self.discount = discount
        self.learning_rate = learning_rate

        # epsilon
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        
        # replay buffer
        self.min_replay_memory_size = min_replay_memory_size
        self.replay_buffer = ReplayBuffer(maxlen=replay_memory_size)

        # main model, this is trained every step
        self.model = self.create_model()
        # target model, this is predicted every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        self.target_update_frequency = target_update_frequency
        self.minibatch_size = minibatch_size

        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0 # track when we're ready to update the target model

    def get_model_name(self):
        return MODEL_NAME

    def create_model(self):
        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded")
            # model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        else:
            model = Sequential()
            model.add(Dense(128, activation="relu", input_shape=(self.obs_space_size,)))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(self.env.action_space.n, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def get_qs(self, state):
        return self.model.predict(np.reshape(state, (1, self.obs_space_size)), verbose=0)[0]
    
    def create_train_dataset(self, minibatch):
        current_states = np.reshape([transition[0] for transition in minibatch], (self.minibatch_size, self.obs_space_size))
        current_qs_list = self.model.predict(current_states, verbose=0)

        new_current_states = np.reshape([transition[3] for transition in minibatch], (self.minibatch_size, self.obs_space_size))
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward # since there is no future q
            
            # Updates Q-value of the current (action,state) pair in the transition
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(np.reshape(current_state, (self.obs_space_size, )))
            y.append(current_qs)
        return np.array(X), np.array(y)

    def train(self, current_state, action, reward, new_state, done):
        # Adds experience to the replay buffer
        self.replay_buffer.add((current_state, action, reward, new_state, done))
        if self.replay_buffer.len() < self.min_replay_memory_size:
            return
        
        # Creates minibatch and updates importance
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


    def get_action(self, current_state):
        if np.random.random() > self.epsilon:
            # Get action from Q table
            action = np.argmax(self.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, self.env.action_space.n)
        return action