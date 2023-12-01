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

OBS_SPACE_SIZE = 13

REPLAY_MEMORY_SIZE = 20_000
MINIBATCH_SIZE = 128
MIN_REPLAY_MEMORY_SIZE = MINIBATCH_SIZE * 10
MODEL_NAME="128x64"
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
LEARNING_RATE = 0.001
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
        self.priorities = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, alpha):
        scaled_priorities = np.array(self.priorities) ** alpha
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities
    
    def len(self):
        return len(self.buffer)

    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample_all(self, batch_size, alpha=1.0):
        sample_probs = self.get_probabilities(alpha)
        sample_indices = random.choices(range(len(self.buffer)), k=batch_size, weights=sample_probs)
        samples = [self.buffer[pos] for pos in sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return samples, importance, sample_indices
    
    def sample(self, batch_size, alpha=1.0):
        # trova indici a caso
        random_indices = random.sample(range(len(self.buffer)), batch_size * 10)
        # trova le priorità associate a ciascun indice
        priorities = np.array([self.priorities[index] for index in random_indices])
        # trova gli indici in base alle priorità
        indices = random.choices(random_indices, k=batch_size, weights=priorities)
        samples = [self.buffer[pos] for pos in indices]
        importance = self.get_importance(self.get_probabilities(alpha)[indices])
        return samples, importance, indices
    
    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e[0]) + abs(e[1]) + offset


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.importance_in = 1
        # main model, this is trained every step
        self.model = self.create_model()

        # target model, this is predicted every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        # epsilon
        self.epsilon = 1
        self.eps_decay = 0.99995
        self.eps_min = 0.1
        
        self.replay_buffer = ReplayBuffer(maxlen=10000)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0 # track when we're ready to update the target model

    def get_model_name():
        return MODEL_NAME

    def create_model(self):
        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded")
            model.compile(loss=self.weighted_mse_loss, optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'], run_eagerly=True)
        else:
            model = Sequential()
            model.add(Dense(128, activation="relu", input_shape=(OBS_SPACE_SIZE,)))
            model.add(Dense(128, activation="relu"))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(self.env.action_space.n, activation="linear"))
            model.compile(loss=self.weighted_mse_loss, optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'], run_eagerly=True)
        return model
    
    def weighted_mse_loss(self, y_true, y_pred):
        importance_float32 = tf.cast(self.importance_in, dtype=tf.float32)
        importance_reshaped = tf.reshape(importance_float32, [-1, 1])
        td_error = y_true - y_pred
        self.losses = tf.square(td_error) * importance_reshaped
        return tf.reduce_mean(self.losses)

    def get_qs(self, state):
        return self.model.predict(np.reshape(state, (1, OBS_SPACE_SIZE)), verbose=0)[0]
    
    def create_train_dataset(self, minibatch):
        current_states = np.reshape([transition[0] for transition in minibatch], (MINIBATCH_SIZE, OBS_SPACE_SIZE))
        current_qs_list = self.model.predict(current_states, verbose=0)

        new_current_states = np.reshape([transition[3] for transition in minibatch], (MINIBATCH_SIZE, OBS_SPACE_SIZE))
        future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward # since there is no future q
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(np.reshape(current_state, (OBS_SPACE_SIZE, )))
            y.append(current_qs)
        return X, y

    def train(self, current_state, action, reward, new_state, done):
        self.replay_buffer.add((current_state, action, reward, new_state, done))
        if self.replay_buffer.len() < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch, importance, indices = self.replay_buffer.sample_all(MINIBATCH_SIZE)
        self.importance_in = importance**(1-self.epsilon)
        X, y = self.create_train_dataset(minibatch)
        
        self.model.fit(np.array(X), np.array(y), batch_size = MINIBATCH_SIZE, 
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if done else None)
        
        self.replay_buffer.set_priorities(indices, self.losses)
        if done:
            self.target_update_counter += 1
            # Decay epsilon
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
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