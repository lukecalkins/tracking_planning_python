import itertools as IT
import numpy as np
from utils import restrict_angle, kron_delta
from sensor import Measurement
from kalmanFilter import GaussianBelief, KalmanFilterMeasurementUpdate
from scipy.stats import poisson, multivariate_normal
from scipy.linalg import sqrtm
from math import factorial
from graph import Graph

sqrt = np.sqrt

from copy import copy
from collections import deque

class JPDAF:

    def __init__(self, detection_prob, clutter_density, gate_level=0.99, verbose = False):
        self._detection_prob = detection_prob
        self._gate_level = gate_level
        self._verbose = verbose
        self._clutter_density = clutter_density


    def filter(self, measurements, robot):
        """
        Interfacing function that calls applies the JPDAF filter given target estimates and a set of measurements
        :param measurements: list of measurements (raw, does not use associated target ID)
        :param robot: robot with targets estimates in its robot.tmm field
        :param clutter_density:
        :return:
        """

        Omega = self.gate_measurements(measurements, robot, self._gate_level)
        if self._verbose:
            print("Omega: ", Omega, "\n")

        event_matrices = self.generate_association_events(Omega)
        event_probabilities = self.get_event_probabilities(event_matrices, measurements, robot, self._clutter_density)

        #calculate association probabilites
        association_probability_matrix = np.zeros((Omega.shape[0] + 1, Omega.shape[1] - 1))
        self.get_association_probabilities(association_probability_matrix, event_matrices, event_probabilities)

        #with association probabilities, the weighted innovation and innovation covariance for each target can be calculated
        self.update_estimates(association_probability_matrix, measurements, robot)

    def update_estimates(self, prob_mat, measurements, robot):
        """
        function that obtains the weighted innovation and weighted covariance matrix and updates the respective targets
        mean and covariances through the EKF
        :param prob_mat: association prob matrix with num_rows=num_measurements + 1 (not detected), num_cols = num_targets
        :param robot: robot object
        :return:
        """

        #loop over targets, calculate weighted innovation and weighted inovtaion covariance
        for i in range(len(robot.tmm.targets)):
            info_target = robot.tmm.targets[i]
            weighted_innovation = self.get_weighted_innovation(prob_mat[:, i], measurements, info_target)
            P_k = self.get_P_k(prob_mat[:, i], measurements, info_target, weighted_innovation)
            beta_0 = prob_mat[-1, i]
            W_k = info_target.filter_gain_matrix
            S_k = info_target.innovation_cov
            covariance_update = info_target.cov_predict - (1 - beta_0)*W_k @ S_k @ W_k.transpose() + P_k
            mean_update = info_target.mean_predict + W_k @ weighted_innovation
            info_target.updateBelief(mean_update, covariance_update)

    def get_P_k(self, prob_mat_column, measurements, info_target, weighted_innovation):
        """
        function to generate the P_k matrix (fortman1983sonar) used in covariance update equation
        :param prob_mat_column: association probability matrix column corresponding to target in question
        :param measurements:
        :param info_target:
        :param weighted_innovation:
        :return:
        """

        weighted_innovation_outer_product = np.zeros((weighted_innovation.shape[0], weighted_innovation.shape[0]))
        z_predict = info_target.z_predict
        for i in range(len(measurements)):
            if prob_mat_column[i] != 0:
                innovation = measurements[i].getZ() - z_predict
                weighted_innovation_outer_product += prob_mat_column[i] * (innovation @ innovation.transpose())

        W_k = info_target.filter_gain_matrix
        P_k = W_k @ (weighted_innovation_outer_product - weighted_innovation @ weighted_innovation.transpose()) @ W_k.transpose()

        return P_k

    def get_weighted_innovation(self, prob_mat_column, measurements, info_target):
        """
        function that return the weighted innovation base on probabilities of association. The info_target already has
        its mean_predicted ans stored in the object
        :param prob_mat_column: only the column of the association matrix relavent to the target in question
        :param measurements:
        :param info_target: target in question getting weighted innovation for
        :return: weighted innovation measurement
        """
        weighted_innovation = np.array([[0.]])
        z_predict = info_target.z_predict
        for i in range(prob_mat_column.shape[0] - 1):   # subtract 1 becaus last row is beta_0^t
            if prob_mat_column[i] != 0:
                weighted_innovation += prob_mat_column[i] * (measurements[i].getZ() - z_predict)

        return weighted_innovation

    def get_association_probabilities(self, prob_mat, event_matrices, event_probabilities):
        """
        functiont that takes the full probability matrix as reference and each even matrix and
        its corresponding probability and gets the association probability for each measurement to
        each target
        :param prob_mat: reference to probablilty matrix (num_meas X num_targets + 1)
        :param event_matrices: list of each distinct even
        :param event_probabilities: corresponding probabilities of each event
        :return: None (prob_mat passed in by reference is updated)
        """
        for i in range(event_matrices[0].shape[0]):
            for j in range(event_matrices[0].shape[1] - 1):
                # calculate probability of measurement i is associated to target j
                event_indicators = [1 if event[i, j + 1] == 1 else 0 for event in event_matrices]

                #add up the prboabilities of all event where this is true
                # 1) retrieve
                if sum(event_indicators) != 0:
                    relevant_event_probs = [event_probabilities[i] if event_indicators[i] == 1 else 0 for i in range(len(event_indicators))]
                    prob_mat[i, j] = sum(relevant_event_probs)

        # add last row of association probability matrix = probability none of the mesasurements originated from target
        for i in range(prob_mat.shape[1]):
            prob_mat[-1, i] = 1 - prob_mat[:-1, i].sum()

        return None

    def get_event_probabilities(self, event_matrices, measurements, robot, density):
        """
        function that will loop through each event matrix and calculate its corresponding posterior
        probability of occurence.
        :param event_matrices: matrix representing a joint event
        :param measurements:
        :param robot:
        :param tragetIDs:
        :param density: clutter density per unit volume
        :return:
        """

        event_probs = np.array([])
        for event in event_matrices:
            num_false_alarms = event.sum(axis=0)[0]  # sum first column
            event_prob = 1  # start running product
            for i in range(event.shape[0]):
                if event[i, 1:].sum() != 0:
                    # measurement associated to that target
                    likelihood = self.get_measurement_likelihood(event[i, :], measurements[i], robot)
                    event_prob = event_prob * likelihood

            #find detected/undetected targets and multiply by detection probabilities (or 1 - detection probability)
            for i in range(1, event.shape[1]):
                if event[:, i].sum() == 1:
                    event_prob = event_prob * self._detection_prob
                elif event[:, i].sum() == 0:
                    event_prob = event_prob * (1 - self._detection_prob)
                else:
                    print("multiple measurements assigned to target", event)

            event_prob = event_prob * density ** num_false_alarms
            event_probs = np.append(event_probs, event_prob)


        if self._verbose:
            for event in event_matrices:
                print(event)
            print(event_probs, "sum event probs", event_probs.sum())

        # event probabilites are normalized currently. Dividing by the sum of all of them will make them valid probablies
        event_probs = event_probs/event_probs.sum()

        return event_probs

    def get_measurement_likelihood(self, event_row, measurement, robot):
        """
        functiont that calculates the likelihood of a measuerment given that it is associated to a particulat target
        :param event_row: the row of the even matrix corresponding to the measurement associated
        :param measurement: measurement object associated to the target
        :param robot: robot in which the JPDA filter is running
        :return: value of gaussian likelihood function
        """

        #find the correct target
        target_ndx = np.argmax(event_row) - 1  # -1 because first column represents false alarms
        info_target = robot.tmm.targets[target_ndx]

        #this info target already has a predicted mean and innovation covariance stored from gating measurements
        innovation = measurement.getZ() - info_target.z_predict
        return self.gaussian_likelihood(innovation, info_target.innovation_cov)

    def gaussian_likelihood(self, innovation, cov):
        """
        calculates the gaussian likelihood given the innovation
        :param innovation:
        :param cov:
        :return:
        """
        z_dim = innovation.shape[0]
        inner_product = innovation @ np.linalg.solve(cov, innovation)
        pdf_val = np.exp(-1./2 * inner_product)*(1/(2*np.pi)**z_dim/2.)*(1./np.sqrt(np.linalg.det(cov)))

        return pdf_val

    def gate_measurements(self, measurements, robot, level=0.99):
        """
        Function that creates the fully populated gated matrix encoding all possible data association events based on
        the one dimensional bearing measurement with linearized  measurement model.
        THIS FUNCTION PREDICTS THE MEAN AND COVARIANCE STEPS AND STORE FOR EACH
        :param measurements:
        :param robot:
        :param level:
        :return: Omega gating matrix
        """

        # generate prediction and innovation covariance within each target
        x_t = robot.getState()
        num_targets = robot.tmm.num_targets()
        y_dim = int(robot.tmm.target_dim / num_targets)
        z_dim = robot.sensor.z_dim
        for info_target in robot.tmm.targets:
            #info_target = robot.tmm.targets[targetIDs[i]]
            info_target.predictMeanAndCovariance(1)
            H = np.zeros((z_dim, y_dim))
            V = np.zeros((z_dim, z_dim))
            robot.sensor.getJacobian(H, V, x_t, info_target.mean_predict)
            z_predict = robot.sensor.observationModel(x_t, info_target.mean_predict)
            info_target.set_z_predict_and_innovation_covariance(z_predict, H, V)
            info_target.set_gate_volume(level)

        n_rows = len(measurements)
        # n_cols = robot.tmm.num_targets()
        Omega = np.zeros((n_rows, num_targets + 1))  # add another column for false alarm measurements

        # populate first column with all ones
        for i in range(n_rows):
            Omega[i, 0] = 1

        for i in range(num_targets):
            #gate target
            for j in range(n_rows):
                Omega[j, i + 1] = self.gateMeasurementToTarget(robot.tmm.targets[i], measurements[j],
                                                               level)
        return Omega

    def gateMeasurementToTarget(self, target, measurement, level):
        """
        Compute bearing gate around target based on mahaolobis distance and given level
        :param target: info target stored on robot
        :param level: confidence level of the Gaussian to use for gating
        :return:
        """

        delta = target.gate_volume / 2
        if np.abs(measurement.getZ() - target.z_predict) > delta:
            return 0  # measurement outside gate
        else:
            return 1

    def generate_association_events(self, Omega):
        """
        Function that takes a full binary gate matrix that endcodes the possible data association scenarios and outputs a
        list of individual data association events. Omega has entry in (i,j) it measurement i can be associated to measurement j.
        Column 0 reserved for all false alarm measurements and therefore always fully populated for each measurement.

        Rules for each output matrix
        1) each row must contain a single 1 (each measurement associated to one target, or clutter
        2) each column t>0 can contain at most a single 1 (each target only generates one measurement

        :param Omega: full binary gate matrix
        :return: List of matrices the same size as Omega corresponding to each association event
        """

        n_rows, n_cols = Omega.shape
        rows_lists = []
        for i in range(n_rows):
            col_indicators = []
            for j in range(n_cols):
                if Omega[i, j] == 1:
                    col_indicators.append(j)
            rows_lists.append(col_indicators)

        all_events = list(IT.product(*rows_lists))  # this list satisfies one mark per row
        valid_events = self.get_valid_events(all_events,  n_cols)
        event_matrices = self.generate_valid_event_matrices(valid_events, n_cols)

        return event_matrices

    def generate_valid_event_matrices(self, valid_events, n_cols):
        """
        function that generates valid event matrices with a list ot event tuples
        :param valid_events:
        :param n_cols:
        :return:
        """

        n_rows = len(valid_events[0])
        event_mat_list = []
        for event in valid_events:
            event_mat = np.zeros((n_rows, n_cols))
            for i in range(n_rows):
                event_mat[i, event[i]] = 1
            event_mat_list.append(event_mat)

        return event_mat_list

    def get_valid_events(self, all_events, n_cols):
        """
        function that takes all permutations of assigning each measurement and prunes in order to make sure each measurement
        only comes from one target (column) except for column 0 which can accept up to all the measurements

        :param all_events: list of tuples with tuple length equal to the the number of targets plus one
        :param n_cols:
        :return: list of tuples
        """

        indices_to_delete = np.zeros(len(all_events))
        valid_events = []
        for i in range(len(all_events)):
            for j in range(1, len(all_events[0]) + 1):
                if all_events[i].count(j) > 1:
                    indices_to_delete[i] = 1

        #create list of valid events
        for i in range(len(all_events)):
            if indices_to_delete[i] == 0:
                valid_events.append(all_events[i])

        return valid_events

########################
###### end JPDAF #######
########################

########################################################################################################################
########################################################################################################################

class JPDAFMerged:

    def __init__(self, sensor, unresolved_resolution, clutter_density, gate_level=0.99, FOV=2*np.pi, verbose=False):
        self.sensor = sensor
        self.bearing_res = unresolved_resolution
        self._clutter_density = clutter_density
        self._gate_level = gate_level
        self._verbose = verbose
        self._FOV = FOV  # sensor field of view in radians
        self._inn_cov_list = []   # each iteration, this will be populated with the innovation covariance of each target
        self._z_predict_list = []
        self._H_k_list = []


    def filter(self, measurements, robot):
        """
        fully external functioning JPDAF filter without clutter and with perfect detections
        :param measurements: measurement list
        :param ownship: own_ship state
        :param a targ_predict_beliefs: list of target beliefs (mean, cov) from previous predcited from prevous time step
        :return: a
        """
        # reset inn_cov_list, z predict List and H_k list to be empty. Will be populated with innovation covariance
        # for each target
        self._inn_cov_list = []
        self._z_predict_list = []
        self._H_k_list = []
        z_dim = robot.sensor.z_dim
        y_dim = robot.tmm.targets[0]._y_dim
        b_sigma = robot.sensor.get_b_sigma()

        ownship = robot.getState()
        beliefs = self.get_predicted_beliefs(robot)
        full_belief = self.get_full_predicted_belief(beliefs)
        H_tilde = self.build_H_tilde(beliefs, ownship, z_dim)
        z_target_predict = np.array(self._z_predict_list)

        #resolution_matrix = self.get_resolution_matrix(beliefs, ownship)
        graphs = self.get_feasible_graphs(robot, beliefs)
        for graph in graphs:
            graph.build_resolution_update_multipliers()
            graph.build_resolution_update_D_matrices()

        # Create all data association hypotheses for each graph
        graph_data_association_list = self.get_graph_data_association_hypotheses(measurements, graphs, beliefs)

        #for each GraphDataAssociation object, calculate the probability of each data association
        for gda in graph_data_association_list:
            gda.calculate_association_probabilities(self._clutter_density, self._FOV, self.sensor._detection_prob)
            gda.build_C_k_matrices()
            gda.perform_measurement_update_one_step(full_belief, H_tilde, b_sigma, measurements, z_target_predict, self._FOV)
            gda.perform_resolution_update_one_step(H_tilde, self.bearing_res, z_target_predict)

        # Weights and gaussian beliefs calculated for all graphs, data associations and resolutions updates
        # Now perform moment matching on entire gaussian mixture.
        output = self.full_JPDAF_update(graph_data_association_list, y_dim)

        return output

    def full_JPDAF_update(self, graph_data_association_list, y_dim):
        """
        this function takes the means, covariances and weights from all the graphs and data association hypotheses and
        performs the full moment matching for the JPDAF.
        :param graph_data_association_lsit:
        :return:
        """
        N_targets = graph_data_association_list[0].graph.n_vertices
        mean_update = np.zeros((N_targets * y_dim, 1))
        cov_update = np.zeros((N_targets * y_dim, N_targets * y_dim))

        # calculate normalizing constant
        all_probabilities_list = []
        for gda in graph_data_association_list:
            all_probabilities_list = all_probabilities_list +  gda.resolution_updated_probabilities
        all_probabilities_as_array = np.array(all_probabilities_list)
        normalizing_constant = all_probabilities_as_array.sum()

        #first update the mean
        for gda in graph_data_association_list:
            for i in range(len(gda.resolution_updated_beliefs)):
                weight = gda.resolution_updated_probabilities[i] / normalizing_constant
                mean_update = mean_update + weight * gda.resolution_updated_beliefs[i]._mean

        #use the moment matched mean for the covariance update
        for gda in graph_data_association_list:
            for i in range(len(gda.resolution_updated_probabilities)):
                weight = gda.resolution_updated_probabilities[i] / normalizing_constant
                cov_update = cov_update + weight * (gda.resolution_updated_beliefs[i]._cov + (
                            gda.resolution_updated_beliefs[i]._mean - mean_update) @
                                                    (gda.resolution_updated_beliefs[i]._mean - mean_update).T)


        JPDAF_belief = GaussianBelief(mean_update, cov_update)
        return JPDAF_belief

    def get_graph_data_association_hypotheses(self, measurements, graphs, beliefs):
        """
        function that takes the measurements and all feasible graphs from the target predictions and creates a list of
        objects where each item is a graph and possible data association for the graph.
        :param measurements:
        :param graphs:
        :return:
        """
        # Create size of association matrix
        n_targs = len(beliefs)
        m_meas = len(measurements)

        meas_range = list(range(n_targs + 1))
        rows_list = m_meas * [meas_range]  # every measurement could come from every target, or clutter
        all_events = list(IT.product(*rows_list))

        gda_list = []
        for i in range(len(graphs)):
            gda = GraphDataAssociation(graphs[i])
            gda.build_data_association_hypotheses(all_events)
            gda_list.append(gda)

        return gda_list

    def get_valid_events(self, events, n_targs, graph):
        """
        takes a list of events where each event is a list of length equal to the number of measurements and entries
        indicating the column index (0 for clutter, > 0 for target) where the measurement could have come from and
        returns the set of events that are feasible according to the rules DETERMINED BY THE SPECIFIC GRAPH
        :param events:
        :param n_targs:
        :param graph:
        :return:
        """

        event_mat_list = []
        for event in events:
            Omega = np.zeros((len(event), n_targs + 1), dtype=int)
            for i in range(len(event)):  # i represent measurement number, event[i] is the column (target) to assign a 1
                Omega[i, event[i]] = 1
                if event[i] > 0:  # belongs to a target
                    if graph.is_connected(event[i] - 1):  # connected to another target, add 1's for measurement to appropriate target
                        connected_targs = graph.get_connected_targets(event[i] - 1)
                        Omega[i, connected_targs] = 1
            is_in = [(Omega == item).all() for item in event_mat_list]  # an entry will return true if Omega already present
            if not event_mat_list:
                event_mat_list.append(Omega)
            else:
                if not any(is_in):
                    event_mat_list.append(Omega)

        # Now take away matrices that have columns with more than one entry (except first column)
        indices_to_delete = np.zeros(len(events))
        valid_event_mats = []
        for i in range(len(event_mat_list)):
            for j in range(1, n_targs + 1):
                if event_mat_list[i][:, j].sum() > 1:
                    indices_to_delete[i] = 1

        for i in range(len(event_mat_list)):
            if indices_to_delete[i] == 0:
                valid_event_mats.append(event_mat_list[i])


        return valid_event_mats

    def get_feasible_graphs(self, robot, beliefs):
        """
        Function to take the target beliefs and produce all feasible graphs for resolution event. In Bearing only case
        there is only one feasible graph per resolution event
        :param beliefs: target beliefs
        :return: list of graphs. Each graph will contain list of target indices that are grouped in one list (unresolved)
        or by itself in
        """
        bearings = get_bearings(robot, beliefs)
        bearings_2pi = bearings + np.pi
        sorted_index = sorted(range(len(bearings_2pi)), key=lambda k: bearings_2pi[k])
        n_targs = len(beliefs)

        # construct feasible edge set
        feasible_edges = []
        for i in range(n_targs - 1):
            feasible_edges.append((sorted_index[i], sorted_index[i + 1]))
        if n_targs > 2:
            feasible_edges.append((sorted_index[n_targs - 1], sorted_index[0]))

        # construct all possible graphs of 0 edges, up to n_targs - 1 edges
        graphs = []
        for i in range(n_targs):
            edge_list = list(IT.combinations(feasible_edges, i))
            for edges in edge_list:
                graphs.append(Graph(n_targs, edges, feasible_edges))

        return graphs

    def get_resolution_matrix(self, beliefs, own_state):
        """
        calculated the pairwise resolution probabilities between each target
        :param beliefs:
        :return:
        """

        n_targs = len(beliefs)
        P_u = np.zeros((n_targs, n_targs))
        for i in range(n_targs):
            P_u[i, i] = 1
            for j in range(i + 1, n_targs):
                P_u[i, j] = get_unresolved_prob_bearing(beliefs[i], beliefs[j], self.bearing_res, self.sensor,
                                                        own_state)
                P_u[j, i] = P_u[i, j]

        return P_u

    def build_H_tilde(self, targ_predict_beliefs, ownship, z_dim):
        """
        this function takes the beliefs
        :param beliefs:
        :return:
        """
        y_dim = targ_predict_beliefs[0]._mean.shape[0]
        num_targs = len(targ_predict_beliefs)
        H_tilde = np.zeros((z_dim * num_targs, y_dim * num_targs))
        for i in range(num_targs):
            start_row = i * z_dim
            stop_row = start_row + z_dim
            start_col = i * y_dim
            stop_col = start_col + y_dim
            z_predict = self.sensor.observationModel(ownship, targ_predict_beliefs[i]._mean)
            H = np.zeros((z_dim, y_dim))
            V = np.zeros((z_dim, z_dim))
            self.sensor.getJacobian(H, V, ownship, targ_predict_beliefs[i]._mean)
            self._H_k_list.append(H)
            self._z_predict_list.append(z_predict)
            H_tilde[start_row:stop_row, start_col:stop_col] = H

        return H_tilde

    def get_full_predicted_belief(self, beliefs):
        """
        same as get_predicted beliefs  but just stacking all target beliefs into a single state vector and
        covariance matrix
        :param beliefs: list of beliefs
        :return:
        """
        y_dim = beliefs[0]._mean.shape[0]
        num_targs = len(beliefs)
        system_state_vector = np.zeros((num_targs * y_dim,  1))
        system_covariance = np.zeros((num_targs * y_dim, num_targs * y_dim))
        for i in range(num_targs):
            start = i * y_dim
            stop = start + y_dim
            system_state_vector[start:stop] = beliefs[i]._mean
            system_covariance[start:stop, start:stop] = beliefs[i]._cov
        system_belief = GaussianBelief(system_state_vector, system_covariance)

        return system_belief

    def get_predicted_beliefs(self, robot):
        """
        function that will take targets from the robot target model and generate predicted mean and covariance for the
        current time step
        :param robot: robot object
        :return: list of Gaussian beliefs (mean and cov)
        """

        beliefs = []
        for i in range(len(robot.tmm.targets)):
            target = robot.tmm.targets[i]
            y_targ_belief = target.getState()
            cov_targ_belief = target.getCovariance()
            A = target.getJacobian()
            W = target.getNoise()
            y_targ_predict = A @ y_targ_belief
            cov_targ_predict = A @ cov_targ_belief @ A.transpose() + W
            predicted_belief = GaussianBelief(y_targ_predict, cov_targ_predict)
            beliefs.append(predicted_belief)

        return beliefs

    def gate_measurements(self, measurements, targ_predict_beliefs, ownship):
        """
        create omega gating matrix only with targets and no clutter
        :param measurements:
        :param targ_predict_beliefs:
        :param own_ship:
        :return:
        """
        num_targs = len(targ_predict_beliefs)
        num_meas = len(measurements)
        y_dim = targ_predict_beliefs[0]._mean.shape[0]
        z_dim = measurements[0].get_z_dim()

        Omega = np.zeros((num_meas, num_targs))

        for i in range(num_targs):
            z_predict = self.sensor.observationModel(ownship, targ_predict_beliefs[i]._mean)
            H = np.zeros((z_dim, y_dim))
            V = np.zeros((z_dim, z_dim))
            self.sensor.getJacobian(H, V, ownship, targ_predict_beliefs[i]._mean)
            inn_cov = H @ targ_predict_beliefs[i]._cov @ H.transpose() + V
            gate_volume = self.get_gate_volume(inn_cov)

            self._inn_cov_list.append(inn_cov)   # to be utilized later when calculating probabilities later
            self._z_predict_list.append(np.array([z_predict]))           # both passed in as 2d array for matrix multiplication
            self._H_k_list.append(H)

            if self._verbose:
                print("target ", i, ", gate_volume: ", gate_volume*180/np.pi, " degrees")

            for j in range(num_meas):
                #columns represent targets, rows represent measurements
                Omega[j, i] = self.gate_measurement_to_target(gate_volume, z_predict, measurements[j])

        return Omega

########################
###### end JPDAF_merged #######
########################

class GraphDataAssociation:

    def __init__(self, graph):
        self.graph = graph
        self.data_associations = []
        self.association_probabilities = []
        self.measurement_updated_probabilities = []
        self.C_k_list = []
        self.measurement_updated_beliefs = []
        self.resolution_updated_beliefs = []
        self.resolution_updated_probabilities = []

    def perform_resolution_update_sequential(self, H_tilde, bearing_res, z_target_predict):
        """
        take the measurement updated beliefs that are already stored and performs sequential resolution update for each
        term in the summation or (32) in svenson2012multitarget
        :param H_tilde:
        :param bearing_res:
        :param z_target_predict:
        :return:
        """


    def perform_resolution_update_one_step(self, H_tilde, bearing_res, z_target_predict):
        """
        takes the measurement updated beliefs that are already stored and does a resolution update with the graph
        :param H_tilde:
        :return:
        """
        #R_u = 1. / (np.sqrt(2 * np.log(2))) * bearing_res ** 2
        R_u = bearing_res ** 2
        N_targets = self.graph.n_vertices

        association_index = 0
        for meas_update in self.measurement_updated_beliefs:
            mean = meas_update._mean
            y_dim = mean.shape[0]//N_targets
            cov = meas_update._cov
            for k in range(len(self.graph.resolution_update_D_matrices)):
                if len(self.graph.edge_multipliers[k]) == 0:  # belief multiplied by one in fully resolved
                    self.resolution_updated_beliefs.append(meas_update)
                    self.resolution_updated_probabilities.append(self.measurement_updated_probabilities[association_index])
                    continue
                D_mat = self.graph.resolution_update_D_matrices[k]
                update_sign = self.graph.edge_multipliers_sign[k]
                inn_cov = (D_mat @ H_tilde) @ cov @ (D_mat @ H_tilde).T + R_u * np.eye(N_targets)
                gain = cov @ (D_mat @ H_tilde).T @ np.linalg.inv(inn_cov)
                #mean_update = mean + gain @ (np.zeros((self.graph.n_vertices, 1)) - D_mat @ H_tilde @ mean)
                mean_update = mean + gain @ (np.zeros((N_targets, 1)) - D_mat @ z_target_predict)
                cov_update = (np.eye(N_targets * y_dim) - gain @ D_mat @ H_tilde) @ cov
                self.resolution_updated_beliefs.append(GaussianBelief(mean_update, cov_update))
                probability_factor = sqrt(np.linalg.det(2 * np.pi * R_u * np.eye(N_targets)))
                #probability_factor = probability_factor * multivariate_normal.pdf(np.zeros(N_targets), np.ndarray.flatten(D_mat @(H_tilde @ mean)),
                                                                              #inn_cov)
                probability_factor = probability_factor * multivariate_normal.pdf(np.zeros(N_targets), np.ndarray.flatten(D_mat @ (z_target_predict)),
                                                                                  inn_cov)
                self.resolution_updated_probabilities.append(update_sign * probability_factor *
                                                             self.measurement_updated_probabilities[association_index])
            association_index += 1

        # normalize resolution probabilities
        #probabilities_as_array = np.array(self.resolution_updated_probabilities)
        #self.resolution_updated_probabilities = 1./probabilities_as_array.sum() * probabilities_as_array

        return None

    def perform_measurement_update_one_step(self, full_belief, H_tilde, b_sigma, measurements, z_target_predict, FOV):
        """
        takes full system beliefs and covariance and performs the measurments update for each possible data association
        in the graph
        :param full_belief:
        :param H_tilde:
        :param b_sigma: measurement noise standard deviation
        :param measurements
        :param FOV: sensor field of view (radians)
        :return:
        """
        for i in range(len(self.data_associations)):
            association_mat = self.data_associations[i]
            num_meas = association_mat.shape[0]
            num_clutter_meas = association_mat[:, 0].sum()
            num_targ_meas = num_meas - num_clutter_meas
            if self.C_k_list[i] is None:
                self.measurement_updated_beliefs.append(full_belief)
                self.measurement_updated_probabilities.append(self.association_probabilities[i] *
                                                              1./(FOV**num_clutter_meas))
            else:
                H_caron = self.C_k_list[i] @ H_tilde
                Z_k_target, num_target_per_meas = self.get_target_generated_measurements(self.data_associations[i], measurements)
                V = self.build_measurement_covariance_merged(b_sigma, num_target_per_meas)
                #Z_k_target_predict = self.get_Z_k_target_predict(self.data_associations[i])
                self.measurement_updated_beliefs.append(KalmanFilterMeasurementUpdate(full_belief._mean, full_belief._cov,
                                                                                      H_caron, V, Z_k_target, self.C_k_list[i] @ z_target_predict))
                measurement_likelihood = multivariate_normal.pdf(np.ndarray.flatten(Z_k_target),
                                                                 np.ndarray.flatten(self.C_k_list[i] @ z_target_predict), V)
                self.measurement_updated_probabilities.append(measurement_likelihood * 1./(FOV ** num_clutter_meas) *
                                                              self.association_probabilities[i])

    def build_measurement_covariance_merged(self, b_sigma, num_target_per_meas):
        """
        build measurement noise covariance matrix used in kalman filter. Size of variance depends on number of targets
        that generated the measurement
        :param b_sigma: sensor noise standard deviation parameter
        :param num_target_per_meas: vector with each entry containing the number of targets that generated that measurement
        :return: V: measurement noise covariance matrix
        """
        num_targ_meas = num_target_per_meas.shape[0]
        V = np.zeros((num_targ_meas, num_targ_meas))
        for i in range(num_targ_meas):
            V[i, i] = (num_target_per_meas[i] * b_sigma) ** 2
        return V

    def get_target_generated_measurements(self, event_mat, measurements):
        """
        returns vector of target generated measurements in order of
        :param event_mat:
        :param measurements:
        :return:
        """
        num_meas = len(measurements)
        num_clutter_meas = event_mat[:, 0].sum()
        num_targ_meas = num_meas - num_clutter_meas

        Z_k_target = np.zeros((num_targ_meas, 1))
        num_targets_per_meas = np.zeros(num_targ_meas)
        targ_meas_index = 0
        for i in range(num_meas):
            if sum(event_mat[i, 1:]) > 0:  # measurement comes from target
                Z_k_target[targ_meas_index, 0] = measurements[i].getZ()
                num_targets_per_meas[targ_meas_index] = sum(event_mat[i, 1:])
                targ_meas_index += 1
        return Z_k_target, num_targets_per_meas


    def build_C_k_matrices(self):
        """
        build the C_k matrix for each data association hypothesis.
        :return:
        """
        for event_mat in self.data_associations:
            self.C_k_list.append(self.get_C_k_from_event(event_mat))

    def get_C_k_from_event(self, event_mat):
        num_meas = event_mat.shape[0]
        num_targ = self.graph.n_vertices
        num_clutter_meas = event_mat[:, 0].sum()
        num_targ_meas = num_meas - num_clutter_meas
        if num_targ_meas == 0:
            return None
        C_k = np.zeros((num_targ_meas, num_targ))
        targ_meas_index = 0

        for i in range(num_meas):
            if event_mat[i, 1:].sum() > 0:  # target generated meas
                target_indices = np.nonzero(event_mat[i, 1:])[0]
                C_k[targ_meas_index, target_indices] = 1/len(target_indices)
                targ_meas_index += 1

        return C_k

    def calculate_association_probabilities(self, clutter_density, FOV, detection_prob):
        """
        function that calculates and stores the probability of each data association for the  graph
        :return: None
        """
        num_meas = self.data_associations[0].shape[0]
        num_targ = self.data_associations[0].shape[1] - 1
        for association_mat in self.data_associations:
            num_clutter_meas = np.count_nonzero(association_mat[:, 0])
            num_targ_meas = num_meas - num_clutter_meas
            prob_clutter = poisson.pmf(num_clutter_meas, clutter_density*FOV)
            running_prob = prob_clutter * factorial(num_clutter_meas) / factorial(num_meas)
            visited = set()
            for i in range(1, num_targ + 1):
                if i not in visited:
                    if (association_mat[:, i] > 0).any():
                        running_prob = running_prob * detection_prob
                        visited.add(i)
                        if self.graph.is_connected(i - 1):
                            connected_nodes = self.graph.get_connected_targets(i - 1)
                            for j in connected_nodes:
                                visited.add(j)
                    else:
                        running_prob = running_prob * (1 - detection_prob)
                        visited.add(i)
                        # print('In else')

            self.association_probabilities.append(running_prob)


        return None

    def build_data_association_hypotheses(self, events):
        """
        takes a list of all possible events and generates feasible data associations given the graph
        :param events: events given in rows list format. One entry per row with index of column to add a 1
        :return:
        """
        n_targs = self.graph.n_vertices
        event_mat_list = []
        for event in events:
            Omega = np.zeros((len(event), n_targs + 1), dtype=int)
            for i in range(len(event)):  # i represent measurement number, event[i] is the column (target) to assign a 1
                Omega[i, event[i]] = 1
                if event[i] > 0:  # belongs to a target
                    if self.graph.is_connected(
                            event[i] - 1):  # connected to another target, add 1's for measurement to appropriate target
                        connected_targs = self.graph.get_connected_targets(event[i] - 1)
                        Omega[i, connected_targs] = 1
            is_in = [(Omega == item).all() for item in
                     event_mat_list]  # an entry will return true if Omega already present
            if not event_mat_list:
                event_mat_list.append(Omega)
            else:
                if not any(is_in):
                    event_mat_list.append(Omega)

        # Now take away matrices that have columns with more than one entry (except first column)
        indices_to_delete = np.zeros(len(events))
        valid_event_mats = []
        for i in range(len(event_mat_list)):
            for j in range(1, n_targs + 1):
                if event_mat_list[i][:, j].sum() > 1:
                    indices_to_delete[i] = 1

        for i in range(len(event_mat_list)):
            if indices_to_delete[i] == 0:
                valid_event_mats.append(event_mat_list[i])

        self.data_associations = valid_event_mats
'''
class Graph:

    def __init__(self, n_vertices, edges, feasible_edges):
        self.n_vertices = n_vertices
        self.edges = edges
        self.feasible_edges = feasible_edges
        self.adjacency = self.build_adjacency()
        self.non_edges = [edge for edge in feasible_edges if edge not in edges]
        self.edge_multipliers = []
        self.edge_multipliers_sign = []
        self.resolution_update_D_matrices = []


    def build_resolution_update_D_matrices(self):
        """
        function that takes each product of P_u terms in the resolution update and contructs the associated D matrix
        that is used in the single step resolution update for that term.
        :return:
        """
        for edge_set in self.edge_multipliers:
            G = np.zeros((self.n_vertices, self.n_vertices))
            for edge in edge_set:
                pi_i_j = get_pi_i_j(edge, self.n_vertices)
                G = G + pi_i_j @ pi_i_j.transpose()
            D = sqrtm(G)
            self.resolution_update_D_matrices.append(np.real(D))

    def build_resolution_update_multipliers(self):

        num_resolved = len(self.non_edges)
        num_unresolved = len(self.edges)
        edge_multipliers = []
        edge_multipliers_sign = []
        for i in range(num_resolved + 1):
            edge_list = list(IT.combinations(self.non_edges, i))
            for edge_set in edge_list:
                edge_multipliers.append(list(edge_set))
                if len(edge_set) % 2 == 0:
                    edge_multipliers_sign.append(1)
                else:
                    edge_multipliers_sign.append(-1)
        if num_unresolved > 0:
            for edge_set in edge_multipliers:
                for connected_edge in self.edges:
                    edge_set.append(connected_edge)

        self.edge_multipliers = edge_multipliers
        self.edge_multipliers_sign = edge_multipliers_sign

    def build_adjacency(self):
        adjacency = np.zeros((self.n_vertices, self.n_vertices))
        for edge in self.edges:
            adjacency[edge[0], edge[1]] = 1
            adjacency[edge[1], edge[0]] = 1
        return adjacency

    def is_connected(self, targ_index):
        """
        returns true if target is connected to any other target in the graph
        :param targ_index:
        :return:
        """
        if self.adjacency[targ_index, :].sum() > 0:
            return True
        else:
            return False

    def get_connected_targets(self, targ_index):
        """
        function that takes a target index (starting at zero and produces a list of target indices that are connected
        to it
        :param targ_index:
        :return:
        """
        row = self.adjacency[targ_index, :]
        visited = {targ_index}
        connected = {targ_index}
        for connected_targ in list(row.nonzero()[0]):  # return as a tuple of length 1
            connected.add(int(connected_targ))

        while visited != connected:
            for targ in connected:
                if targ not in visited:
                    row = self.adjacency[targ, :]
                    for connected_targ in list(row.nonzero()[0]):
                        if connected_targ not in connected:
                            connected.add(connected_targ)
                    visited.add(targ)
                    break

        # only want to return connected indices, not the one target itself
        connected.remove(targ_index)

        connected_list = list(connected)
        correct_target_index_for_Omega = [targ + 1 for targ in connected_list]

        return correct_target_index_for_Omega

##############################################
############## End Graph #####################
##############################################

def get_pi_i_j(edge_pair, n):
    """
    returns pi_i_j vector from svennson2012multitarget
    :param edge_pair:
    :param n:
    :return:
    """
    pi_i_j = np.zeros((n, 1))
    for i in range(n):
        pi_i_j[i] = kron_delta(i, edge_pair[0]) - kron_delta(i, edge_pair[1])

    return pi_i_j
'''
def get_bearings(robot, beliefs):
    """
    takes target beliefs and returns list of bearings to each target belief state
    :param ownship:
    :param beliefs:
    :return:
    """
    bearings = []
    own_state = robot.getState()
    for i in range(len(beliefs)):
        bearing = robot.sensor.observationModel(own_state, beliefs[i].getMean())
        bearings.append(bearing)

    return np.array(bearings)

def get_unresolved_prob_bearing(b0, b1, bearing_res, sensor, own_state):
    """
    calculates the probability that two targets are unresolved in bearing
    :param b0: belief for target 0 (mean and cov)
    :param b1: belief for target 1
    :param bearing_res: variance parameters of guassian like resolution probability function
    :param sensor: bearing sensor object
    :return:
    """

    bearing0 = sensor.observationModel(own_state, b0.getMean())
    bearing1 = sensor.observationModel(own_state, b1.getMean())
    delta_bearing = restrict_angle(bearing0 - bearing1)

    res_cov = 1./(np.sqrt((2*np.log(2))))*bearing_res ** 2
    prob_unresolved = np.exp(-delta_bearing * 1./res_cov * delta_bearing)

    return prob_unresolved

def add_masked_measurements_2targ(measurements, robot, proximity):
    """
    function that detects if a marking event is occurring among targets being tracked. If so, it adds artificial
    measurements to the existing measurements set that have already been generated and passed to the function
    :param measurements:
    :param robot:
    :param proximity: proximity in degrees of a masking event
    :return:
    """

    target_dist = []
    target_bearing = []
    own_state = robot.getState()
    for i in range(len(robot.tmm.targets)):
        info_target = robot.tmm.targets[i]
        dist = np.linalg.norm(info_target.getPosition() - own_state[0:2])
        target_dist.append(dist)
        mean_predict = info_target._A @ info_target._state
        z_predict = robot.sensor.observationModel(own_state, info_target.getState())
        target_bearing.append(z_predict)

    delta_bearing = restrict_angle(np.linalg.norm(target_bearing[0] - target_bearing[1]))
    print("Delta bearing in Masked function = ", delta_bearing * 180. / np.pi, " degrees")
    if np.abs(delta_bearing) < (np.pi / 180 * proximity):
        # find which target is closer to sensor and delete that targets measurement
        max_ndx = np.argmax(target_dist)
        measurements.append(Measurement(target_bearing[max_ndx], 0, 1))  # todo: do we need actual infered ID of target?
        print("adding artificial measurement")



