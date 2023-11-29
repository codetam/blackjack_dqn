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

REPLAY_MEMORY_SIZE = 10_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 128
MODEL_NAME="128x64"
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 5
LEARNING_RATE = 0.1
LOAD_MODEL = "models/128x64_____0.04avg__1701208440.model"

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

class DQNAgent:
    def __init__(self, env):
        self.env = env
        # main model, this is trained every step
        self.model = self.create_model()

        # target model, this is predicted every step
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0 # track when we're ready to update the target model

    def get_model_name():
        return MODEL_NAME

    def create_model(self):
        if LOAD_MODEL is not None:
            print(f"Loading {LOAD_MODEL}")
            model = load_model(LOAD_MODEL)
            print(f"Model {LOAD_MODEL} loaded")
        else:
            model = Sequential()
            model.add(Dense(128, activation="relu", input_shape=(30,)))
            model.add(Dense(64, activation="relu"))
            model.add(Dense(self.env.action_space.n, activation="linear"))
            model.compile(loss="mse", optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
        return model
    
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.reshape(state, (1, 30)), verbose=0)[0]
    
    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.reshape([transition[0] for transition in minibatch], (MINIBATCH_SIZE, 30))
        current_qs_list = self.model.predict(current_states, verbose=0)

        new_current_states = np.reshape([transition[3] for transition in minibatch], (MINIBATCH_SIZE, 30))
        # future_qs_list = self.target_model.predict(new_current_states, verbose=0)

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # if not done:
            #     max_future_q = np.max(future_qs_list[index])
            #     new_q = reward + DISCOUNT * max_future_q
            #else:
            new_q = reward # since there is no future q
            
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(np.reshape(current_state, (30, )))
            y.append(current_qs)

        # we only fit if we are in our terminal state
        self.model.fit(np.array(X), np.array(y), batch_size = MINIBATCH_SIZE, 
                       verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)
        
        if terminal_state:
            self.target_update_counter += 1
        
        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0
            

    
    

