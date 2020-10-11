from trackingLib.utils import *
from anytree import Node
from copy import copy, deepcopy
from trackingLib.kalmanFilter import KalmanFilterCovAndInnovationCov, GaussianBelief
import numpy as np
from trackingLib.cost_function import *
import sys
from trackingLib.sensor import Measurement
import json
from copy import copy, deepcopy
from trackingLib.dataAssociation import get_unresolved_prob_bearing
from trackingLib.dataAssociationPlan import get_bearings
from trackingLib.graph import Graph

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
    def __init__(self, state, beliefs_model, sensor, cost_func, JPDA_sim=None, JPDAM_sim=None, parent=None, action=None):
        Node.__init__(self, action, parent)
        self.state = state
        self.beliefs_model = beliefs_model
        self.sensor = sensor
        self.cost_func = cost_func
        self.action = action
        self.JPDA_sim = JPDA_sim
        self.JPDAM_sim = JPDAM_sim

    def make_children(self, actions, filter_type):
        for action in actions:
            if not any(child.name == action for child in self.children):
                if filter_type == 'kalman':
                    self._make_child_kalman(action)
                elif filter_type == 'JPDAF':
                    self._make_child_JPDAF(action)
                elif filter_type == 'JPDAF_merged':
                    self._make_child_JPDAF_merged(action)
                else:
                    print("filter type not recognized in SearchNode.make_children")
                    exit()

    def _make_child_kalman(self, action):
        child = SearchNode(deepcopy(self.state), self.beliefs_model, self.sensor, self.cost_func, JPDA_sim=self.JPDA_sim,
                           parent=self, action=action)
        child.state.move(action)
        #child.state.inn_cov = []  # reset inn cov to empty
        child.state.filter_cov(self.beliefs_model, self.sensor, child.depth)
        child.state.total_cost = self.state.total_cost + child.state.get_cost(self.cost_func, child.depth)

        return child  # caller doesn't actually store it

    def _make_child_JPDAF(self, action):
        child = SearchNode(deepcopy(self.state), self.beliefs_model, self.sensor, self.cost_func, JPDA_sim=self.JPDA_sim,
                           parent=self, action=action)
        child.state.move(action)
        #child.state.inn_cov = []  # reset inn cov to empty
        child.state.filter_cov_JPDA(self.beliefs_model, self.sensor, child.depth, self.JPDA_sim)
        child.state.total_cost = self.state.total_cost + child.state.get_cost(self.cost_func, child.depth)

        return child  # caller doesn't actually store it

    def _make_child_JPDAF_merged(self, action):
        child = SearchNode(deepcopy(self.state), self.beliefs_model, self.sensor, self.cost_func, JPDAM_sim=self.JPDAM_sim,
                           parent=self, action=action)
        child.state.move(action)
        child.state.filter_cov_JPDAM_most_likely(self.beliefs_model, self.sensor, child.depth, self.JPDAM_sim)
        child.state.total_cost = self.state.total_cost + child.state.get_cost(self.cost_func, child.depth)

        return child  # caller doesn't actually store it.

class SearchState:
    def __init__(self, state, Sigma, y, dt):
        self.state = state  # Auv state
        self.Sigma = Sigma  # Target system covariance
        self.targ_state = y
        self.targ_state_at_node = self.targ_state[0, :]
        self.y_dim = 4  # todo: make this automatic
        self.z_dim = 1
        self.dt = dt
        #self.inn_cov = []
        self.total_cost = 0
        self.node_cost = 0
        #self.predicted_meas = []

    def move(self, action):
        self.state = propagateOwnshipEuler(self.state, action[0], action[1], self.dt)

    def filter_cov_JPDAM_most_likely(self, beliefs_model, sensor, depth, JPDAM_sim):
        # todo: implement
        ownship = self.state
        thresh = JPDAM_sim.merging_threshold
        bearing_res = JPDAM_sim.bearing_res

        y_curr = self.targ_state[depth, :]
        self.targ_state_at_node = y_curr
        beliefs = []
        for i in range(self.targ_state.shape[1] // self.y_dim):  # loop over number of targets
            y_targ_predict = y_curr[i * self.y_dim:i * self.y_dim + self.y_dim].reshape(self.y_dim, 1)
            cov_targ = self.Sigma[i * self.y_dim:i * self.y_dim + self.y_dim,
                       i * self.y_dim:i * self.y_dim + self.y_dim]

            # already have mean_predict, get cov predict
            A = beliefs_model.targets[i].getJacobian()
            W = beliefs_model.targets[i].getNoise()
            cov_targ_predict = A @ cov_targ @ A.transpose() + W

            beliefs.append(GaussianBelief(y_targ_predict, cov_targ_predict))

        bearings = get_bearings(ownship, beliefs, sensor)
        n_targs = len(beliefs)
        sorted_index = sorted(range(len(bearings)), key=lambda k: bearings[k], reverse=True)

        # construct feasible edge set
        feasible_edges = []
        for i in range(n_targs - 1):
            bearing_i = bearings[sorted_index[i]]
            bearing_j = bearings[sorted_index[i + 1]]
            if sensor.in_same_FOV(bearing_i, bearing_j):
                feasible_edges.append((sorted_index[i], sorted_index[i + 1]))
        """
        if n_targs > 2:  # todo: make sure this isn't a wrap around case.  Should be able to check sign
            bearing_i = bearings[sorted_index[n_targs - 1]]
            bearing_j = bearings[sorted_index[0]]
            if sensor.in_same_FOV(bearing_i, bearing_j):
                feasible_edges.append((sorted_index[n_targs - 1], sorted_index[0]))
        """
        # now loop through feasible edges and decide whether edge is there or not based on threshold
        edges = []
        for edge in feasible_edges:
            bearing_i = bearings[edge[0]]
            bearing_j = bearings[edge[1]]
            prob = get_unresolved_prob_bearing(bearing_i, bearing_j, bearing_res)
            if prob >= thresh:
                edges.append(edge)

        most_likely_graph = Graph(n_targs, edges, feasible_edges)

        # now, with measurements (in FOV) and predicted beliefs, apply simulated JPDAM
        filter_output, predicted_meas = JPDAM_sim.filter_most_likely(most_likely_graph, beliefs, ownship, bearings)
        self.predicted_meas = predicted_meas
        # Take filter output and update mean and covariances in
        targ_num = 0
        for i in range(self.targ_state.shape[1] // self.y_dim):
            start_block = targ_num * self.y_dim
            end_block = start_block + self.y_dim
            self.Sigma[start_block:end_block, start_block:end_block] = filter_output[i]._cov
            # dont need to update mean since we have Y_T
            targ_num += 1

        return None

    def filter_cov_JPDA_merged(self, beliefs_model, sensor, depth, JPDAM_sim):
        # todo: implement
        ownship = self.state
        thresh = JPDAM_sim.merging_threshold

        y_curr = self.targ_state[depth]
        self.targ_state_at_node = y_curr
        beliefs = []
        meas = []
        for i in range(self.targ_state.shape[1] // self.y_dim):  # loop over number of targets
            y_targ_predict = y_curr[i * self.y_dim:i * self.y_dim + self.y_dim]
            cov_targ = self.Sigma[i * self.y_dim:i * self.y_dim + self.y_dim,
                       i * self.y_dim:i * self.y_dim + self.y_dim]

            # already have mean_predict, get cov predict
            A = beliefs_model.targets[i].getJacobian()
            W = beliefs_model.targets[i].getNoise()
            cov_targ_predict = A @ cov_targ @ A.transpose() + W

            beliefs.append(GaussianBelief(y_targ_predict, cov_targ_predict))
            predicted_meas = sensor.observationModel(ownship, y_targ_predict)
            if sensor.in_FOV(predicted_meas):
                meas.append(Measurement(predicted_meas, 0, 1))

        #now, get most likely graph

        # now, with measurements (in FOV) and predicted beliefs, apply simulated JPDAM
        filter_output = JPDAM_sim.filter(meas, beliefs, ownship)

        return None

    def filter_cov_JPDA(self, beliefs_model, sensor, depth, JPDA_sim):
        # first, grab the predicted mean and covariance
        ownship = self.state   # already computed SearhState.move()

        y_curr = self.targ_state[depth]
        self.targ_state_at_node = y_curr
        beliefs = []
        meas = []
        for i in range(self.targ_state.shape[1] // self.y_dim): # loop over number of targets
            y_targ_predict = y_curr[i * self.y_dim:i * self.y_dim + self.y_dim]
            cov_targ = self.Sigma[i * self.y_dim:i * self.y_dim + self.y_dim, i * self.y_dim:i * self.y_dim + self.y_dim]

            #already have mean_predict, get cov predict
            A = beliefs_model.targets[i].getJacobian()
            W = beliefs_model.targets[i].getNoise()
            cov_targ_predict = A @ cov_targ @ A.transpose() + W

            beliefs.append(GaussianBelief(y_targ_predict, cov_targ_predict))
            predicted_meas = sensor.observationModel(ownship, y_targ_predict)
            if sensor.in_FOV(predicted_meas):
                meas.append(Measurement(predicted_meas, 0, 1))

        #now, with measurements, and predicted beliefs, apply the JPDAF
        filter_output = JPDA_sim.filter(meas, beliefs, ownship)

        # Take filter output and update state mean and covariance
        targ_num = 0
        for i in range(self.targ_state.shape[1] // self.y_dim):
            start_block = targ_num * self.y_dim
            end_block = start_block + self.y_dim
            self.Sigma[start_block:end_block, start_block:end_block] = filter_output[i]._cov
            #dont need to update mean since we have Y_T
            targ_num += 1

        return None

    def filter_cov(self, beliefs_model, sensor, depth):

        ownship = self.state
        y_curr = self.targ_state[depth, :]
        self.targ_state_at_node = y_curr
        num_targs = len(beliefs_model.targets)


        targ_num = 0
        predicted_beliefs = []
        for target in beliefs_model.targets:
            start_block = targ_num * self.y_dim
            end_block = start_block + self.y_dim
            A = target.getJacobian()
            W = target.getNoise()
            H = np.zeros((self.z_dim, self.y_dim))
            V = np.zeros((self.z_dim, self.z_dim))
            sensor.getJacobian(H, V, self.state, self.targ_state[depth])
            Sigma_targ = self.Sigma[start_block:end_block, start_block:end_block]  # todo: use block operator
            bearing_targ = sensor.observationModel(ownship, y_curr[start_block:end_block])
            # update covariance with measurement only if in field of view
            if sensor.in_FOV(bearing_targ):
                cov_update_targ, inn_cov_targ = KalmanFilterCovAndInnovationCov(Sigma_targ, A, W, H, V)
                self.Sigma[start_block:end_block, start_block:end_block] = cov_update_targ
            else:
                cov_update_targ = A @ Sigma_targ @ A.T + W
                self.Sigma[start_block:end_block, start_block:end_block] = cov_update_targ
            #self.inn_cov.append(inn_cov_targ)
            predicted_beliefs.append(GaussianBelief(y_curr[start_block:end_block], None))
            targ_num += 1

        # set predicted measurements
        bearings = get_bearings(ownship, predicted_beliefs, sensor)
        meas = []
        meas_as_list = []
        for i in range(num_targs):
            meas.append(Measurement(bearings[i], 0, 1))
            meas_as_list.append(meas[i].getZ())
        self.predicted_meas = meas_as_list

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
                    #setattr(state_copy, k, copy(v))  # if list is empty
                    setattr(state_copy, k, [])
                elif isinstance(v[0], (list, tuple)):
                    setattr(state_copy, k, deepcopy(v, memodict))
                else:
                    setattr(state_copy, k, copy(v))
            else:
                setattr(state_copy, k, v)
        return state_copy




class Planner:
    def __init__(self, actions, cost_function, filter_type, sensor, horizon, info_target_model, JPDAF_sim=None, JPDAFM_sim = None, final_cost=False, dt=1,
                 log_file=None, log=False, mission_length=None):
        self.actions = actions
        self.cost_function = cost_function
        self.filter = filter_type
        self.sensor = sensor
        self.horizon = horizon
        self.info_target_model = info_target_model
        self.dt = dt
        self.log_file = log_file
        self.final_cost = final_cost
        self.JPDAF_sim = JPDAF_sim      # object that will be passed predicted beliefs and measurements
        self.JPDAFM_sim = JPDAFM_sim
        self._planning_iterations = 0
        self.mission_length = mission_length
        self._log = log
        if self._log:
            self.json_log = {}

    def planFVI(self, system_belief, own_state, tracking_iteration, state_iteration=None, contact_iteration=None, debug=False):

        planner_output = []
        x0 = own_state

        T = self.horizon

        #predict target state
        y_T = self.info_target_model.predictTargetState(system_belief._mean, T)
        Sigma0 = system_belief._cov

        S0 = SearchState(x0, Sigma0, y_T, self.dt)
        #S0.cost = 0  # every path starts at this node so cost doesn't matter
        root = SearchNode(S0, self.info_target_model, self.sensor, self.cost_function, self.JPDAF_sim, self.JPDAFM_sim)

        for i in range(T):
            for leaf in root.leaves:
                leaf.make_children(self.actions, self.filter)


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

        if self._log:
            self.json_log[self._planning_iterations] = {}
            self.json_log[self._planning_iterations]['plan_dt'] = self.dt
            self.json_log[self._planning_iterations]['tracking_iteration'] = tracking_iteration
            self.json_log[self._planning_iterations]['planner_output'] = deepcopy(path_to_node)
            self.json_log[self._planning_iterations]['state_iteration'] = state_iteration
            self.json_log[self._planning_iterations]['contact_iteration'] = contact_iteration

            curr_node = optimal_node
            means = []
            covs = []
            own_states = []
            means.append(optimal_node.state.targ_state_at_node.tolist())
            covs.append(optimal_node.state.Sigma.tolist())
            own_states.append(optimal_node.state.state.tolist())
            while curr_node.parent != None:
                curr_node = curr_node.parent
                if isinstance(curr_node.state.targ_state_at_node, list):
                    means.insert(0, curr_node.state.targ_state_at_node)
                else:
                    means.insert(0, curr_node.state.targ_state_at_node.tolist())
                covs.insert(0, curr_node.state.Sigma.tolist())
                if isinstance(curr_node.state.state, list):
                    own_states.insert(0, curr_node.state.state)
                else:
                    own_states.insert(0, curr_node.state.state.tolist())
            self.json_log[self._planning_iterations]['means'] = means
            self.json_log[self._planning_iterations]['covs'] = covs
            self.json_log[self._planning_iterations]['own_states'] = own_states

        self._planning_iterations += 1

        return path_to_node, optimal_node

    def is_mission_complete(self):
        if self._planning_iterations >= self.mission_length:
            return True
        else:
            return False

    def write_log_file_json(self, directory, filename):
        """
        performs json dump to specified file_name
        :param directory: working directory
        :param filename: name of json file
        :return: None
        """
        filename = filename + '.json'
        with open(directory + filename, 'w') as file:
            json.dump(self.json_log, file, indent=4)


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
