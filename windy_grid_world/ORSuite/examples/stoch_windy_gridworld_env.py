# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import sys


class StochWindyGridWorldEnv(gym.Env):
    '''Creates the Stochastic Windy GridWorld Environment'''


    def __init__(self, GRID_HEIGHT=7, GRID_WIDTH=10,\
                 START_CELL = (3, 0), GOAL_CELL = (3, 7),\
                 REWARD = -1,
                 PROB=[0.35, 0.1, 0.1, 0.1, 0.35],\
                 EPLEN = 5
                 ):


        self.grid_height = GRID_HEIGHT
        self.grid_width = GRID_WIDTH
        self.grid_dimensions = (self.grid_height, self.grid_width)

        self.eplen = EPLEN

        self.start_cell = START_CELL
        self.goal_cell = GOAL_CELL

        self.reward = REWARD


        self.probabilities = PROB


        self.action_space =  spaces.Discrete(4)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(self.grid_height),
                spaces.Discrete(self.grid_width)))
        self.actions = { 'U':0,   #up
                         'R':1,   #right
                         'D':2,   #down
                         'L':3 }  #left

    
    def get_config(self):
        return {}


    def reward_func(self,state):
        if state == self.goal_cell:
            return self.reward
        else:
            return 0
        

    def dim2to1(self, cell):
        '''Transforms the 2 dim position in a grid world to 1 state'''
        return np.ravel_multi_index(cell, self.grid_dimensions)
    
    def dim1to2(self, state):
        '''Transforms the state in a grid world back to its 2 dim cell'''
        return np.unravel_index(state, self.grid_dimensions)

    def step(self, action, force_noise=None):
        """
        Parameters
        ----------
        action : 0 = Up, 1 = Right, 2 = Down, 3 = Left

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                 Agent current position in the grid.
            reward (float) :
                 Reward is -1 at every step except at goal state.
            episode_over (bool) :
                 True if the agent reaches the goal, False otherwise.
            info (dict) :
                 Contains the realized noise that is added to the wind in each 
                 step. However, official evaluations of your agent are not 
                 allowed to use this for learning.
        """
        assert self.action_space.contains(action)

        isdone = False
        if self.timestep == self.eplen - 1:
            isdone = True
        
        self.timestep += 1

        x,y = self.state

        if x == np.ceil(self.grid_width / 2):
            prob = self.probabilities[x]
            val = np.random.binomial(n=1, p=prob)
            if val == 1:
                action = np.random.choice(4,1)[0]

        if action == 0:
            new_x = min(self.grid_height -1, x+1)
            new_y = y
        elif action == 1:
            new_x = x
            new_y = min(self.grid_width-1, y+1)
        elif action == 2:
            new_x = max(0, x-1)
            new_y = y
        elif action == 3:
            new_x = x
            new_y = max(0, y-1)

        self.state = (new_x, new_y)
        reward = self.reward_func(self.state)

        assert self.observation_space.contains(self.state)
        return self.state, reward, isdone, {}
        
    def reset(self):
        ''' resets the agent position back to the starting position'''
        self.state = self.start_cell
        self.timestep = 0
        return self.state   

    def render(self, mode='human', close=False):
        pass

        
