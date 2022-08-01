import numpy as np
from gym import spaces
from .. import Agent
import itertools


class DiscreteMB(Agent):

    """
    Uniform model-based algorithm implemented for MultiDiscrete enviroments
    and actions


    TODO: Note the Flag parameter is used to help save on some compute time.  If the flag is set to True
    we will do so-called "One-Step" updates where we do not solve the full Bellman Equations but only update 
    for current time-step.  Otherwise, we do full Bellman updates.

    Attributes:
        epLen: (int) number of steps per episode
        scaling: (float) scaling parameter for confidence intervals
        action_space: (MultiDiscrete) the action space
        state_space: (MultiDiscrete) the state space
        action_size: (list) representing the size of the action sapce
        state_size: (list) representing the size of the state sapce
        alpha: (float) parameter for prior on transition kernel
        flag: (bool) for whether to do full step updates or not
        matrix_dim: (tuple) a concatenation of epLen, state_size, and action_size used to create the estimate arrays of the appropriate size
        qVals: (list) The Q-value estimates for each episode, state, action tuple
        num_visits: (list) The number of times that each episode, state, action tuple has been visited
        vVals: (list) The value function values for every step, state pair
        rEst: (list) Estimates of the reward for a step, state, action tuple
        pEst: (list) Estimates of the number of times that each step, state, action, new_state tuple is considered
    """

    def __init__(self, action_space, state_space, epLen, scaling, alpha, flag):
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

        self.epLen = epLen
        self.scaling = scaling
        self.alpha = alpha
        self.flag = flag
        if isinstance(action_space, spaces.Discrete):
            self.action_space = spaces.MultiDiscrete(
                nvec=np.array([action_space.n]))
            self.multiAction = False
        else:
            self.action_space = action_space
            self.multiAction = True

        if isinstance(state_space, spaces.Tuple):
            list_vals = []
            for space in state_space.spaces:
                list_vals.append(space.n)
            self.state_space = spaces.MultiDiscrete(nvec = np.array(list_vals))
        else:
            self.state_space = state_space

        # sizes of action and state spaces
        self.action_size = self.action_space.nvec
        self.state_size = self.state_space.nvec



        self.matrix_dim = np.concatenate((
            np.array([self.epLen]), self.state_size, self.action_size))

        """
            Matrix initialization:
            Creates matrix of Q(h,s,a), n(h,s,a), r(h,s,a) and transition estimate
        """

        # Matrix of size h*S*A
        self.qVals = np.ones(self.matrix_dim, dtype=np.float32) * self.epLen
        # matrix of size h*S*A
        self.num_visits = np.zeros(self.matrix_dim, dtype=np.float32)

        # matrix of size h*S
        self.vVals = np.ones(np.append(np.array([self.epLen]), self.state_size),
                             dtype=np.float32) * self.epLen
        # matrix of size h*S*A
        self.rEst = np.zeros(self.matrix_dim, dtype=np.float32)

        # matrix of size h*S*A*S
        self.pEst = np.zeros(np.concatenate((
            np.array([self.epLen]), self.state_size, self.action_size, self.state_size)),
            dtype=np.float32)


    def reset(self): 
        """
        TODO: Reinstantiate the matrix of estimates the same way they are created
        """



    def update_parameters(self, param):
        """Update the scaling parameter.
        Args:
            param: (int) The new scaling value to use"""
        self.scaling = param

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records

        Args:
            obs: (list) The current state
            action: (list) The action taken 
            reward: (int) The calculated reward
            newObs: (list) The next observed state
            timestep: (int) The current timestep
        '''


        dim = tuple(np.append(np.append([timestep], obs), action))

        """
        TODO:
            - Update number of visits
            - Update transition kernel
            - Update mean reward
        """


    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # Update value estimates
        if self.flag:  # update estimates via full step updates
            for h in np.arange(self.epLen - 1, -1, -1):
                for state in itertools.product(*[np.arange(self.state_size[i]) for i in range(self.state_space.shape[0])]):
                    for action in itertools.product(*[np.arange(self.action_size[j]) for j in range(self.action_space.shape[0])]):
                        dim = tuple(np.append(np.append([h], state), action))

                        """
                        TODO:
                         - Implement Bellman update procedure
                        """

                    self.vVals[tuple(np.append([h], state))] = min(self.epLen,
                                                                   self.qVals[tuple(np.append([h], state))].max())

    def pick_action(self, state, step):
        '''
        Select action according to a greedy policy

        Args:
            state: int - current state
            step: int - timestep *within* episode

        Returns:
            list: action
        '''
        if self.flag == False:  # updates estimates via one step update
            for action in itertools.product(*[np.arange(self.action_size[i]) for i in range(self.action_space.shape[0])]):
                dim = tuple(np.append(np.append([step], state), action))
                if self.num_visits[dim] == 0:
                    self.qVals[dim] == 0
                else:
                        """
                        TODO:
                         - Implement Bellman update procedure
                        """

            self.vVals[tuple(np.append([step], state))] = min(self.epLen,
                                                              self.qVals[tuple(np.append([step], state))].max())



        qFn = self.qVals[tuple(np.append([step], state))]
        action = np.asarray(np.where(qFn == qFn.max()))

        index = np.random.choice(len(action[0]))
        action = action[:, index]

        if not self.multiAction:
            action = action[0]
        return action