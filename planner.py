from utils import *
from anytree import Node
from copy import copy, deepcopy
from kalmanFilter import KalmanFilterCovAndInnovationCov
import numpy as np
from cost_function import *
import sys

"""
class Search_Node:
    def __init__(self, x, Sigma, time_step):
        self.x = x   # state of the sensor
        self.Sigma = Sigma   # full target system covariance
        self.cost = 0
        self.time_step = time_step
        self.children = None
"""

class SearchNode(Node):
    def __init__(self, state, robot, cost_func, parent=None, action=None):
        Node.__init__(self, action, parent)
        self.state = state
        self.robot = robot
        self.cost_func = cost_func
        self.action = action

    def make_children(self, actions):
        for action in actions:
            if not any(child.name == action for child in self.children):
                self._make_child(action)

    def _make_child(self, action):
        child = SearchNode(deepcopy(self.state), self.robot, self.cost_func, self, action)
        child.state.move(action)
        child.state.inn_cov = []  # reset inn cov to empty
        child.state.filter_cov(self.robot, child.depth)
        child.state.total_cost = self.state.total_cost + child.state.get_cost(self.cost_func, child.depth)

        return child  # caller doesn't actually store it

class SearchState:
    def __init__(self, state, Sigma, y, dt):
        self.state = state  # Auv state
        self.Sigma = Sigma  # Target system covariance
        self.targ_state = y
        self.y_dim = 4  # todo: make this automatic
        self.z_dim = 1
        self.dt = dt
        self.inn_cov = []
        self.total_cost = 0
        self.node_cost = 0

    def move(self, action):
        self.state = propagateOwnshipEuler(self.state, action[0], action[1], self.dt)

    def filter_cov(self, robot, depth):
        targ_num = 0
        for target in robot.tmm.targets.values():
            start_block = targ_num * self.y_dim
            end_block = start_block + self.y_dim
            A = target.getJacobian()
            W = target.getNoise()
            H = np.zeros((self.z_dim, self.y_dim))
            V = np.zeros((self.z_dim, self.z_dim))
            robot.sensor.getJacobian(H, V, self.state, self.targ_state[depth])
            Sigma_targ = self.Sigma[start_block:end_block, start_block:end_block]  # todo: use block operator
            cov_update_targ, inn_cov_targ = KalmanFilterCovAndInnovationCov(Sigma_targ, A, W, H, V)
            self.Sigma[start_block:end_block, start_block:end_block] = cov_update_targ
            self.inn_cov.append(inn_cov_targ)
            targ_num += 1

        return

    def get_cost(self, cost_func, depth):

        if isinstance(cost_func, GateOverlapCost):
            cost = cost_func.getCost(self.state, self.targ_state[depth], self.inn_cov)
            self.node_cost = cost
            return cost

        elif isinstance(cost_func, LogDetCost):
            cost = cost_func.getCost(self.Sigma)
            self.node_cost = cost
            #print(cost)
            return cost

        else:
            print("Cost function not recognized")
            sys.exit()

    def __deepcopy__(self, memodict={}):
        cls = self.__class__
        state_copy = cls.__new__(cls)
        memodict[id(self)] = state_copy
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(state_copy, k, deepcopy(v, memodict))
            elif isinstance(v, (list, tuple)):
                if len(v) == 0:
                    setattr(state_copy, k, copy(v))  # if list is empty
                elif isinstance(v[0], (list, tuple)):
                    setattr(state_copy, k, deepcopy(v, memodict))
                else:
                    setattr(state_copy, k, copy(v))
            else:
                setattr(state_copy, k, v)
        return state_copy




class Planner:
    def __init__(self, actions, cost_function, dt=1, logfile=None, targ_log=None):
        self.actions = actions
        self.cost_function = cost_function
        self.dt = dt
        self.logfile = logfile
        self.targ_log = targ_log


    def planFVI(self, robot, T, debug=False):

        planner_output = []
        x0 = robot.getState()

        # predict target state
        y_T = robot.tmm.predictTargetState(T)  # target predcited T steps into future
        Sigma0 = robot.tmm.getCovarianceMatrix()  # get initial Sigma at current step

        S0 = SearchState(x0, Sigma0, y_T, dt=1)
        #S0.cost = 0  # every path starts at this node so cost doesn't matter
        root = SearchNode(S0, robot, self.cost_function)

        for i in range(T):
            for leaf in root.leaves:
                leaf.make_children(self.actions)
        print("Planner finished")

        #get output path from final node cost
        leaves = root.leaves
        optimal_node_idx = np.argsort([node.state.total_cost for node in leaves])[0]
        optimal_node = leaves[optimal_node_idx]
        path_to_node = []  # will be populated

        self.get_optimal_path(optimal_node, path_to_node)

        if self.logfile != None:
            self.log_planner_data(root, path_to_node, self.logfile)
            self.log_target_state(root, path_to_node, self.targ_log)

        return path_to_node


    def plan_RVI(self, robot, T, delta, eps, debug=False):



        return 0

    def get_optimal_path(self, node, path):
        """
        populates empty list while finding path from root of
        :param node:
        :param path:
        :return:
        """
        while node.parent != None:
            path.insert(0, node.action)
            node = node.parent

    def log_target_state(self, root, path, file):
        curr_node = root
        t = 0
        f = open(file, 'a')
        while True:
            np.savetxt(f, curr_node.state.targ_state[curr_node.depth])
            if curr_node.children == ():
                break
            child = [node for node in curr_node.children if node.action == path[t]]
            t += 1
            curr_node = child[0]
        f.close()


    def log_planner_data(self, root, path, file):
        """
        function to log the data over the optimal path a computed planning tree
        :param root:
        :param path: list of actions to take from root node
        :return:
        """
        curr_node = root
        t = 0
        f = open(self.logfile, 'a')
        while True:
            array_list = curr_node.state.Sigma.tolist()
            np.savetxt(f, curr_node.state.Sigma)
            if curr_node.children == ():
                break
            child = [node for node in curr_node.children if node.action == path[t]]
            t += 1
            curr_node = child[0]
        f.close()
