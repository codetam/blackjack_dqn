from gamecode import print_gamestate, hit, dealer_turn, turn_init, get_card_num, calculate_sum, print_table, check_soft_hand
import gym
from gym import spaces
import numpy as np

class BlackjackWithBetEnv(gym.Env):
    metadata = {
        "render_modes": [],
    }

    def __init__(self, blackjackEnv, playerAgent, max_bet):
        self.blackjackEnv = blackjackEnv
        self.playerAgent = playerAgent
        self.action_space = spaces.Discrete(max_bet + 1)
        self.observation_space = spaces.Box(0, 1, shape=(14,))
    
    def reset(self, seed=None):
        observation = self.blackjackEnv.reset(init_hands=False)
        return observation

    def step(self, action, render_steps=False):
        assert self.action_space.contains(action)
        bet = action
        current_state = self.blackjackEnv.initialize_hands()
        if render_steps:
            self.blackjackEnv.render()
        done = False
        total_reward = 0
        while not done:
            action = self.playerAgent.get_action(current_state)
            new_state, reward, done, _ = self.blackjackEnv.step(int(action))
            if render_steps:
                self.blackjackEnv.render(action, done, reward*bet)
            current_state = new_state
            total_reward += reward
        reward *= bet
        return current_state, reward, done, {}
    
    def render(self, action=-1, done=False, outcome=0):
        self.blackjackEnv.render(action, done, outcome)

    def get_obs(self):
        return self.blackjackEnv.get_obs()


class BlackjackEnv(gym.Env):
    metadata = {
        "render_modes": [],
    }

    def __init__(self, num_decks=2):
        # 0 -> Stand
        # 1 -> Hit
        # 2 -> Double down
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(0, 1, shape=(14,))
        self.num_decks = num_decks
        self.turn = 0
        self.max_turn = 13
        self.first_hand = 1


    def get_obs(self):
        # normalization of cards out
        box_obs = np.empty((14,), dtype=np.float32)
        box_obs[:9] = self.cards_out[:-1] / (4 * self.num_decks)
        box_obs[9] = self.cards_out[-1] / (16 * self.num_decks)
        
        # normalization of player sum and dealer card
        box_obs[10] = calculate_sum(self.player_cards) / 21
        box_obs[11] = get_card_num(self.dealer_cards[0]) / 10

        box_obs[12] = check_soft_hand(self.player_cards)
        box_obs[13] = self.first_hand
        return (box_obs)
    
    def initialize_hands(self):
        self.player_cards, self.dealer_cards, self.deck, self.cards_out = turn_init(self.deck, self.cards_out)
        observation = self.get_obs()
        return observation
    
    def reset(self, seed = None, init_hands=True):
        if self.turn % self.max_turn == 0:
            super().reset(seed=seed)
            self.turn = 1
            # Randomize decks and initialize cards_out
            deck = np.arange(1, 53, dtype=np.int16)
            for _ in range(self.num_decks - 1):
                self.deck = np.concatenate((deck, deck))
            np.random.shuffle(self.deck)
            self.cards_out = np.zeros((10,), dtype=np.int8)
        else:
            self.turn += 1
        self.first_hand = 1
        self.player_cards = np.zeros((10,), dtype=np.int8)
        self.dealer_cards = np.zeros((10,), dtype=np.int8)

        if init_hands:
            observation = self.initialize_hands()
        else:
            observation = self.get_obs()
        return observation

    def step(self, action):
        assert self.action_space.contains(action)
        self.first_hand = 0
        is_done = True
        # player chose to stand
        if action == 0:
            reward = self.agent_has_finished(False, 1)
        # player chose to hit
        elif action == 1:
            self.player_cards, self.deck, self.cards_out, result = hit(self.player_cards, self.deck, self.cards_out)
            if result == -1:
                reward = self.agent_has_finished(True, 1)
            elif result == 21:
                reward = self.agent_has_finished(False, 1)
            else:
                reward = 0
                is_done = False
        # player chose to double down
        else:
            self.player_cards, self.deck, self.cards_out, result = hit(self.player_cards, self.deck, self.cards_out)
            if result == -1:
                reward = self.agent_has_finished(True, 2)
            else:
                reward = self.agent_has_finished(False, 2)

        observation = self.get_obs()
        return observation, reward, is_done, {}
        
    def agent_has_finished(self, busted, factor):
        if busted == False:
            self.dealer_cards, self.deck, self.cards_out, result = dealer_turn(self.dealer_cards, 
                                                    self.player_cards, self.deck, self.cards_out)
        else:
            result = -1
        # Reward is either -1, 0 or 1
        reward = factor * result
        return reward
    
    def render(self, action=-1, done=False, outcome=0):
        print_gamestate(self.dealer_cards, self.player_cards, self.turn, action, done, outcome)
        print_table(self.num_decks, self.cards_out)
    