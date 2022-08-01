import gym
from gym import spaces
import numpy as np

from .. import env_configs


class TradeExecutionEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, config = env_configs.trade_execution_default_config):
        super(TradeExecutionEnv, self).__init__()    # Define action and observation space
        # They must be gym.spaces objects    # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1, shape=[1], dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=[1], dtype=np.float32)
        self.T = config['T']
        self.Q = config['Q']
    
    def reset(self):
        self.state = [0.5]
        self.timestep = 0
        self.market_sold = 0 # amount of inventory sold
        return np.asarray(self.state)
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        action = action[0]
        if self.market_sold + action > 1: # checks if we are selling more than we have
            action = 1 - self.market_sold
    
        done = False
        if self.timestep == self.T - 1: # updates action in last timestep
            action = 1 - self.market_sold
            done = True
    
        self.market_sold += action
        
        self.timestep += 1
        
        # Calculates new price and updates new state
        
        price = self.state[0]
        price = np.clip(price + np.random.uniform(0, .1),0,1)
        self.state = [price]
        
        # Calculates reward
        reward = action * self.Q * price
        
        return np.asarray(self.state), reward, done, {}
    
    def render(self, mode='human', close=False):
        print(f'Current state: {self.state}')
        print(f'Shares remaining: {1 - self.market_sold}')