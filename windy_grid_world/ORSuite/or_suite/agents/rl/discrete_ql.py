import numpy as np
from .. import Agent
from gym import spaces

''' Q Learning Algorithm '''


class DiscreteQl(Agent):
    """
    Q-Learning algorithm  implemented for enviroments with discrete states and
    actions.

    Attributes:
        epLen: (int) number of steps per episode
        scaling: (float) scaling parameter for confidence intervals
        action_space: (MultiDiscrete) the action space
        state_space: (MultiDiscrete) the state space
        action_size: (list) representing the size of the action sapce
        state_size: (list) representing the size of the state sapce
        matrix_dim: (tuple) a concatenation of epLen, state_size, and action_size used to create the estimate arrays of the appropriate size
        qVals: (list) The Q-value estimates for each episode, state, action tuple
        num_visits: (list) The number of times that each episode, state, action tuple has been visited
    """

    def __init__(self, action_space, observation_space, epLen, scaling):
        """
        action_space: Gym Spaces for the action space of the environment
        observation_space: Gym Spaces for the state space of the environment
        epLen: Number of steps
        scaling: Parameter for the bonus terms
        """

        """
            Gets out the dimension of the state space, i.e. if MultiDiscrete(2,3,2) returns [2,3,2]
            Repeats fro action space similarly
        """
        self.state_space = observation_space
        if isinstance(action_space, spaces.Discrete):
            self.action_space = spaces.MultiDiscrete(
                nvec=np.array([action_space.n]))
            self.multiAction = False
        else:
            self.action_space = action_space
            self.multiAction = True

        if isinstance(observation_space, spaces.Tuple):
            list_vals = []
            for space in observation_space.spaces:
                list_vals.append(space.n)
            self.state_space = spaces.MultiDiscrete(nvec = np.array(list_vals))
        else:
            self.state_space = observation_space

        # stores the epLen + scaling parameter
        self.epLen = epLen
        self.scaling = scaling

        # starts calculating total dimension for the matrix of estimates of Q Values
        dim = np.concatenate((
            np.array([self.epLen]), self.state_space.nvec, self.action_space.nvec))
        self.matrix_dim = dim

        # Initialize with upper bound on max reward via H*max_one_step_reward
        self.qVals = self.epLen * np.ones(self.matrix_dim, dtype=np.float32)
        # Set max_reward as 1 assuming that the reward is normalized
        max_reward = 1
        self.qVals = self.epLen * max_reward * self.qVals
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)



    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.environment = env
        pass

    def update_parameters(self, param):
        """Update the scaling parameter.
        Args:
            param: (float) The new scaling value to use"""
        self.scaling = param

    def reset(self):
        """
        TODO: Reinstantiate the matrix of estimates the same way they are created
        i.e. the qVals and num_visits
        """
        pass

    '''
        Adds the observation to records by using the update formula
    '''
    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records

        Args:
            obs: (list) The current state
            action: (list) The action taken 
            reward: (int) The calculated reward
            newObs: (list) The next observed state
            timestep: (int) The current timestep
        '''
        if not self.multiAction:
            action = [action]

        # Gets the timestep, state, and action and tuples them together to get the index
        # into the large tensor storing Q estimates and n values
        dim = tuple(np.append(np.append([timestep], obs), action))

        """
        TODO:
            - Update number of visits
            - Calculate bonus term
            - Update Q estimate via one-step update formula

        Note that these can be updated + accessed via:

        self.num_visits[dim]
        self.qVals[dim]
        """

        # Updates number of visits

        t = ...

        # Calculates learning rate
        lr = (self.epLen + 1) / (self.epLen + t)

        # Calculates bonus term
        bonus = self.scaling * np.sqrt(1 / t)

        # If at end of time horizon, value function at next step is zero
        # otherwise, take value from the newState at next timestep
        if timestep == self.epLen-1:
            vFn = 0
        else:
            vFn = np.max(self.qVals[np.append([timestep+1], newObs)])
        vFn = min(self.epLen, vFn)


        # Update q Value estimate
        self.qVals[dim] = ...
      
      
    def update_policy(self, k):
        '''Update internal policy based upon records'''
        pass

    def pick_action(self, state, step):
        '''
        Select action according to a greedy policy

        Args:
            state: int - current state
            timestep: int - timestep *within* episode

        Returns:
            list: action
        '''
        # returns the state location and takes action based on
        # maximum q value

        qFn = self.qVals[tuple(np.append([step], state))]
        action = np.asarray(np.where(qFn == qFn.max()))
        index = np.random.choice(len(action[0]))
        action = action[:, index]

        if not self.multiAction:
            action = action[0]
        return action
