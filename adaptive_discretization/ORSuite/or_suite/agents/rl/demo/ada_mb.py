import numpy as np
from .. import Agent
from or_suite.agents.rl.utils.tree_model_based import MBTree, MBNode


class AdaptiveDiscretizationMB(Agent):
    """
    Adaptive model-based Q-Learning algorithm  implemented for enviroments
    with continuous states and actions using the metric induces by the l_inf norm.


    Attributes:
        epLen: (int) number of steps per episode
        scaling: (float) scaling parameter for confidence intervals
        inherit_flag: (bool) boolean of whether to inherit estimates
        dim: (int) dimension of R^d the state_action space is represented in
    """

    def __init__(self, epLen, scaling, alpha, split_threshold, inherit_flag, flag, state_dim, action_dim):
        """
        Args:
            epLen: number of steps per episode
            numIters: total number of iterations
            scaling: scaling parameter for UCB term
            alpha: parameter to add a prior to the transition kernels
            inherit_flag: boolean on whether to inherit when making children nodes
            flag: boolean of full (true) or one-step updates (false)
            split_threshold: scaling for splitting a region based on number of visits
            dimensions: state+action space dimensions
        """

        self.epLen = epLen
        self.scaling = scaling
        self.alpha = alpha
        self.split_threshold = split_threshold
        self.inherit_flag = inherit_flag
        self.flag = flag
        self.dim = state_dim + action_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        for _ in range(epLen):
            tree = MBTree(self.epLen, self.state_dim, self.action_dim)
            self.tree_list.append(tree)

    def update_parameters(self, param):
        self.scaling = param

    def reset(self):
        # TODO: Reset tree lists similar to the init statement
        pass


    def update_config(self, env, config):
        ''' Update agent information based on the config__file.'''
        pass

    # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records.'''

        tree = self.tree_list[timestep]

        # Gets the active ball by finding the argmax of Q values of relevant
        active_node, _ = tree.get_active_ball(obs)

        # TODO: Update value function estimate for this node within the tree.
        # First up: 
        # Increments the number of visits


        # Update empirical estimate of average reward for that node


        # If it is not the last timestep - updates the empirical estimate
        # of the transition kernel based on the induced state partition at the next step
        if timestep != self.epLen - 1:

            next_tree = self.tree_list[timestep+1]
            new_obs_loc = np.argmin(
                np.max(np.abs(np.asarray(next_tree.state_leaves) - newObs), axis=1))
            active_node.pEst[new_obs_loc] += 1


        if self.flag == False:  # we are doing one-step updates for the estimates
            if timestep == self.epLen - 1:  # q value estimate at last step is straightforward
                ...
            else:  # otherwise we need to add on an additional estimate of the value function at the next step using transition kernel
                ...
                # NOTE: Can use \alpha parameter as a prior for taking the expectation

            # Update estimate of value function  for state leaves
            index = 0
            for state_val in tree.state_leaves:
                _, qMax = tree.get_active_ball(state_val)
                tree.vEst[index] = min(qMax, self.epLen, tree.vEst[index])
                index += 1

        t = active_node.num_visits
        '''Determines if it is time to split the current ball.'''
        if t >= 2**(self.split_threshold * active_node.depth):

            if timestep >= 1:
                _ = tree.tr_split_node(
                    active_node, timestep, self.inherit_flag, self.epLen, self.tree_list[timestep-1])
            else:
                _ = tree.tr_split_node(
                    active_node, timestep, self.inherit_flag, self.epLen, self.tree_list[timestep-1])

    def update_policy(self, k):
        '''Update internal policy based upon records.'''

        # Solves the empirical Bellman equations

        if self.flag:  # Only done if we are doing full-step updates
            for h in np.arange(self.epLen-1, -1, -1):

                # Gets the current tree for this specific time step
                tree = self.tree_list[h]
                for node in tree.leaves:
                    # If the node has not been visited before - set its Q Value
                    # to be optimistic (i.e. epLen)
                    if node.num_visits == 0:
                        node.qVal = self.epLen
                    else:
                        # Otherwise solve for the Q Values with the bonus term

                        # If h == H then the value function for the next step is zero
                        if h == self.epLen - 1:
                            ...
                        else:  # Gets the next tree to estimate the transition kernel
                            next_tree = self.tree_list[h+1]
                            ...

                # After updating the Q Value for each node - computes the estimate of the value function
                index = 0
                for state_val in tree.state_leaves:
                    _, qMax = tree.get_active_ball(state_val)
                    tree.vEst[index] = min(qMax, self.epLen, tree.vEst[index])
                    index += 1

        pass

    def pick_action(self, state, timestep):
        '''
        Select action according to a greedy policy.

        Args:
            state: int - current state
            timestep: int - timestep *within* episode

        Returns:
            int: action
        '''
        # Considers the partition of the space for the current timestep
        tree = self.tree_list[timestep]

        # Gets the selected ball
        active_node, _ = tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        action_dim = self.dim - len(state)

        action = np.random.uniform(
            active_node.bounds[action_dim:, 0], active_node.bounds[action_dim:, 1])

        return action
