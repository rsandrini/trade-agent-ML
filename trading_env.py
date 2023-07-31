import gym
from gym import spaces
import numpy as np


class TradingEnv(gym.Env):
    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, 1)

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # BUY or SELL
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,))  # open, close, high, low, volume

        # Initialize state
        self.state = ...

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1

        if self.current_step >= len(self.df):
            done = True
            return self.state, self.calculate_reward(), done, {}

        self.execute_trade(action)

        done = False
        return self.state, self.calculate_reward(), done, {}

    def execute_trade(self, action):
        current_price = self.df.iloc[self.current_step]['close']
        if action == 0:  # BUY
            if self.status == 'AWAITING_BUY':
                self.status = 'AWAITING_SELL'
                self.buy_price = current_price
        elif action == 1:  # SELL
            if self.status == 'AWAITING_SELL':
                self.status = 'AWAITING_BUY'
                self.sell_price = current_price
    def calculate_reward(self):
        if self.status == 'AWAITING_SELL':
            return 0  # no profit since we haven't sold yet
        else:
            return self.sell_price - self.buy_price  # profit

    def reset(self):
        # Reset the state of the environment to an initial state
        ...
        return self.state

    def render(self, mode='human'):
        # Render the environment to the screen (optional)
        pass