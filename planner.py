from utils import *
from anytree import Node
from copy import copy, deepcopy
from kalmanFilter import KalmanFilterCovAndInnovationCov, GaussianBelief
import numpy as np
from cost_function import *
import sys
from sensor import Measurement
import json

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
    def __init__(self, state, robot, cost_func, JPDA_sim=None, parent=None, action=None):
        Node.__init__(self, action, parent)
        self.state = state
        self.robot = robot
        self.cost_func = cost_func
        self.action = action
        self.JPDA_sim = JPDA_sim

    def make_children(self, actions):
        for action in actions:
            if not any(child.name == action for child in self.children):
                self._make_child(action)

    def _make_child(self, action):
        child = SearchNode(deepcopy(self.state), self.robot, self.cost_func, self.JPDA_sim, self, action)
        child.state.move(action)
        child.state.inn_cov = []  # reset inn cov to empty
        child.state.filter_cov(self.robot, child.depth)
        #child.state.filter_cov_JPDA(self.robot, child.depth, self.JPDA_sim)
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

    def filter_cov_JPDA(self, robot, depth, JPDA_sim):
        # first, grab the predicted mean and covariance
        ownship = self.state   # already computed SearhState.move()

        y_curr = self.targ_state[depth]
        beliefs = []
        meas = []
        for i in range(int(len(self.targ_state)/self.y_dim)): # loop over number of targets
            y_targ_predict = y_curr[i * self.y_dim:i * self.y_dim + self.y_dim]
            cov_targ = self.Sigma[i * self.y_dim:i * self.y_dim + self.y_dim, i * self.y_dim:i * self.y_dim + self.y_dim]

            #already have mean_predict, get cov predict
            A = list(robot.tmm.targets.values())[0].getJacobian()
            W = list(robot.tmm.targets.values())[0].getNoise()
            cov_targ_predict = A @ cov_targ @ A.transpose() + W

            beliefs.append(GaussianBelief(y_targ_predict, cov_targ_predict))
            predicted_meas = robot.sensor.observationModel(ownship, y_targ_predict)
            meas.append(Measurement(predicted_meas, 0, 1))

        #now, with measurements, and predicted beliefs, apply the JPDAF
        filter_output = JPDA_sim.filter(meas, beliefs, ownship)

        # Take filter output and update state mean and covariance
        targ_num = 0
        for i in range(int(len(self.targ_state)/self.y_dim)):
            start_block = targ_num * self.y_dim
            end_block = start_block + self.y_dim
            self.Sigma[start_block:end_block, start_block:end_block] = filter_output[i]._cov
            #dont need to update mean since we have Y_T
            targ_num += 1

        return None

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

        return None

    def get_cost(self, cost_func, depth):

        if isinstance(cost_func, GateOverlapCost):
            cost = cost_func.getCost(self.state, self.targ_state[depth], self.inn_cov)
            self.node_cost = cost
            if cost < 0:
                raise SystemExit
            return cost

        elif isinstance(cost_func, LogDetCost):
            cost = cost_func.getCost(self.Sigma)
            self.node_cost = cost
            #print(cost)
            return cost

        elif isinstance(cost_func, DeltaBearingCost):
            cost = cost_func.getCost(self.state, self.targ_state[depth])
            self.node_cost = cost
            return cost

        elif isinstance(cost_func, MaxEigCost):
            cost = cost_func.getCost(self.Sigma)
            self.node_cost = cost
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
    def __init__(self, actions, cost_function, JPDAF_sim=None, final_cost=False, dt=1, log_file=None, log_flag=False):
        self.actions = actions
        self.cost_function = cost_function
        self.dt = dt
        self.log_file = log_file
        self.final_cost = final_cost
        self.JPDAF_sim = JPDAF_sim      # object that will be passed predicted beliefs and measurements
        self._log_flag = log_flag

    def planFVI(self, robot, T, debug=False):

        planner_output = []
        x0 = robot.getState()

        # predict target state
        y_T = robot.tmm.predictTargetState(T)  # target predcited T steps into future
        Sigma0 = robot.tmm.getCovarianceMatrix()  # get initial Sigma at current step

        S0 = SearchState(x0, Sigma0, y_T, dt=1)
        #S0.cost = 0  # every path starts at this node so cost doesn't matter
        root = SearchNode(S0, robot, self.cost_function, self.JPDAF_sim)

        for i in range(T):
            for leaf in root.leaves:
                leaf.make_children(self.actions)
        print("Planner finished")

        #get output path from final node cost
        leaves = root.leaves
        if self.final_cost == True:
            optimal_node_idx = np.argsort([node.state.node_cost for node in leaves])[0]
            print("Using final node cost")
        else:
            optimal_node_idx = np.argsort([node.state.total_cost for node in leaves])[0]
            print("using total node cost")
        optimal_node = leaves[optimal_node_idx]
        path_to_node = []  # will be populated

        self.get_optimal_path(optimal_node, path_to_node)

        if self._log_flag:
            self.log_planner_path(optimal_node)


        return path_to_node


    def plan_RVI(self, robot, T, delta, eps, debug=False):

        return 0

    def get_optimal_path(self, node, path):
        """
        populates empty list of actions while finding path from root of to the given optimal node
        :param node:
        :param path:
        :return:
        """
        while node.parent != None:
            path.insert(0, node.action)
            node = node.parent

    def log_planner_path(self, node):
        data = {}
        data['cov'] = []
        data['pos'] = []
        data['ownship'] = []
        data['node_cost'] = []
        data['total_cost'] = []
        while True:
            data['cov'].insert(0, node.state.Sigma.tolist())
            data['pos'].insert(0, node.state.targ_state[node.depth].tolist())
            data['ownship'].insert(0, node.state.state.tolist())
            data['node_cost'].insert(0, node.state.node_cost)
            data['total_cost'].insert(0, node.state.total_cost)
            node = node.parent
            if node == None:
                break

        with open(self.log_file, 'w') as outfile:
            json.dump(data, outfile)

        return None

