from trackingLib.kalmanFilter import GaussianBelief, KalmanFilterMeasurementUpdate
import numpy as np
import itertools as IT
from copy import copy, deepcopy
from trackingLib.utils import gaussian_pdf
from trackingLib.dataAssociation import get_unresolved_prob_bearing, GraphDataAssociation
from trackingLib.graph import Graph
from trackingLib.sensor import Measurement


class JPDAF_simulate:

    def __init__(self, sensor, gate_level=0.99, verbose=False):
        self.sensor = sensor
        self._gate_level = gate_level
        self._verbose = verbose
        self._inn_cov_list = []   # each iteration, this will be populated with the innovation covariance of each target
        self._z_predict_list = []
        self._H_k_list = []

    def filter(self, measurements, targ_predict_beliefs, ownship):
        """
        fully external functioning JPDAF filter without clutter and with perfect detections
        :param measurements: measurement list
        :param ownship: own_ship state
        :param a targ_predict_beliefs: list of target beliefs (mean, cov) from previous predcited from prevous time step
        :return: a
        """
        #reset inn_cov_list, z predict List and H_k list to be empty. Will be populated with innovation covariance
        # for each target
        self._inn_cov_list = []
        self._z_predict_list = []
        self._H_k_list = []

        Omega = self.gate_measurements(measurements, targ_predict_beliefs, ownship)
        if self._verbose:
            print("Omega: ", Omega, "\n")

        event_matrices = self.generate_association_events_no_clutter(Omega)
        event_probabilities = self.get_event_probabilities_no_clutter(event_matrices, measurements, targ_predict_beliefs)

        association_probability_matrix = np.zeros((Omega.shape[0] + 1, Omega.shape[1]))  # still need extra row for case outside FOV (not detected)
        self.get_association_probabilities_no_clutter(association_probability_matrix, event_matrices, event_probabilities)

        output = self.update_estimates_no_clutter(association_probability_matrix, measurements, targ_predict_beliefs)

        return output

    def update_estimates_no_clutter(self, prob_mat, measurements, targ_predict_beliefs):
        """
        takes the prediceted means and covariances and of each target alond with the measurements and and association
        probabilites for each measurement to each target and applies the JPDAF filtes
        :param prob_mat:
        :param measurements:
        :param targ_predict_beliefs:
        :return:
        """
        output = []
        for i in range(len(targ_predict_beliefs)):
            weighted_innovation = self.get_weighted_innovation(prob_mat[:, i], measurements, self._z_predict_list[i])
            W_k = targ_predict_beliefs[i]._cov @ self._H_k_list[i].transpose() @ np.linalg.inv(self._inn_cov_list[i])
            P_k = self.get_P_k(prob_mat[:, i], measurements, self._z_predict_list[i], weighted_innovation, W_k)
            S_k = self._inn_cov_list[i]
            beta_0 = prob_mat[-1, i]
            P_pred = targ_predict_beliefs[i]._cov
            x_pred = targ_predict_beliefs[i]._mean

            mean_update = x_pred + W_k @ weighted_innovation
            cov_update = P_pred - (1 - beta_0) * W_k @ S_k @ W_k.transpose() + P_k
            output.append(GaussianBelief(mean_update, cov_update))

        return output

    def get_P_k(self, probs, meas, z_predict, weighted_innovation, W_k):

        weighted_outer_product = np.zeros((weighted_innovation.shape[0], weighted_innovation.shape[0]))
        for i in range(len(meas)):
            if probs[i] != 0:
                innovation = meas[i].getZ() - z_predict
                weighted_outer_product += probs[i] * (innovation @ innovation.transpose())
        P_k = W_k @ (weighted_outer_product - weighted_innovation @ weighted_innovation.transpose()) @ W_k.transpose()

        return P_k

    def get_weighted_innovation(self, probs, meas, z_predict):
        """

        :param probs: column of prob mat corresponding to specific target
        :param meas:
        :param z_predict:
        :return:
        """
        weighted_innovation = np.array([[0.]])
        for i in range(len(meas)):
            if probs[i] != 0:
                weighted_innovation += probs[i] * (meas[i].getZ() - z_predict)

        return weighted_innovation

    def get_association_probabilities_no_clutter(self, prob_mat, event_mat, event_prob):
        """
        takes zero-filled full probability matrix as reference and populates it given the event matrics and event
        probabilities
        :param prob_mat: zero-filled probability matrix to be filled
        :param event_mat: list of events matrices
        :param event_prob: list of event probabilities in same order as event matrices
        :return:
        """
        num_meas, num_targ = event_mat[0].shape

        for i in range(num_meas):
            for j in range(num_targ):
                # find events where measurement i is associated target j
                event_indicators = [1 if event[i, j] == 1 else 0 for event in event_mat]

                #add up the probabilities of all events where this is true
                if sum(event_indicators) != 0:
                    relevant_probs = [event_prob[i] if event_indicators[i] == 1 else 0 for i in range(len(event_mat))]
                    prob_mat[i, j] = sum(relevant_probs)

        # add last row of association probability matrix = probability none of the mesasurements originated from target
        for i in range(prob_mat.shape[1]):
            prob_mat[-1, i] = 1 - prob_mat[:-1, i].sum()

        return None

    def get_event_probabilities_no_clutter(self, event_matrices, measurements, targ_beliefs):
        """
        with no clutter, all probabilites are target measurement probabilites
        :param event_matrices:
        :param measurements:
        :param targ:
        :return:
        """

        event_probs = np.array([])
        for event in event_matrices:
            event_prob = 1  # start running product
            for i in range(event.shape[0]):
                # find target for measurement i
                targ_ndx = np.argmax(event[i, :])
                likelihood = gaussian_pdf(measurements[i].getZ(), self._z_predict_list[targ_ndx], self._inn_cov_list[targ_ndx])
                event_prob = event_prob * likelihood

            #don't need detection probabilities or false alarm factors
            event_probs = np.append(event_probs, event_prob)

        if self._verbose:
            for event in event_matrices:
                print(event)
            print(event_probs, "sum event probs: ", event_probs.sum())

        #normalize event probabilities
        event_probs = event_probs/event_probs.sum()
        #print("number of event = ", len(event_matrices), " event probs sum = ", event_probs.sum())

        return event_probs

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
        #z_dim = measurements[0].get_z_dim()
        z_dim = 1  # todo: get this value automatically

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

    def generate_association_events_no_clutter(self, Omega):
        """
        rows_list has one entry per row where each entry is a list with column indices that have a 1
        :param Omega:
        :return:
        """
        n_rows, n_cols = Omega.shape
        rows_lists = []
        for i in range(n_rows):
            col_indicators = []
            for j in range(n_cols):
                if Omega[i, j] == 1:
                    col_indicators.append(j)
            rows_lists.append(copy(col_indicators))

        all_events = list(IT.product(*rows_lists))  # this list satisfies one mark per row
        valid_events = self.get_valid_events_no_clutter(all_events, n_cols)
        event_matrices = self.generate_valid_event_matrices(valid_events, n_cols)

        return event_matrices

    def generate_valid_event_matrices(self, valid_events, n_cols):
        """
        each "event" in valid_events list has length equal to number of rows and each entry represents the index of the
        column to
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
            event_mat_list.append(deepcopy(event_mat))

        return event_mat_list

    def get_valid_events_no_clutter(self, events, n_cols):
        """
        functions that takes a list of tuples where each tuple represents a matrix. The length of the tuple represents
        the number of rows (number of measurements), and each entry of the tuple represents the index
        of the column where a 1 is to be placed. This functions checks that all rows have only a single
        entry and all columns only have a single entry. This verifies that each target produces only one measurement
        and that each measurement gets assigned to only one target
        :return:
        """

        indices_to_delete = np.zeros(len(events))
        valid_events = []
        for i in range(len(events)):
            for j in range(n_cols):
                if events[i].count(j) > 1:
                    indices_to_delete[i] = 1

        #remove invalid matrix representers from all possible events
        for i in range(len(events)):
            if indices_to_delete[i] == 0:
                valid_events.append(events[i])

        return valid_events

    def gate_measurement_to_target(self, gate_volume, z_predict, measurement):
        delta = gate_volume/2
        if np.abs(measurement.getZ() - z_predict) > delta:
            return 0
        else:
            return 1

    def get_gate_volume(self, inn_cov):
        """

        :param inn_cov: innovation covariance
        :return:
        """

        if self._gate_level == 0.95:
            k_alpha = 3.84
        elif self._gate_level == 0.99:
            k_alpha = 6.64
        elif self._gate_level == 0.999:
            k_alpha = 10.83
        else:
            print("Desired gating level not found")

        gate_volume = 2 * np.sqrt(k_alpha) * np.sqrt(np.linalg.det(inn_cov))

        return gate_volume

######################################################################
###################   End JPDAF_simulate #############################
######################################################################

class JPDAF_merged_simulate:

    def __init__(self, sensor, unresolved_resolution, sequential_resolution_update_flag,
                 FOV=2*np.pi, gate_level=0.99, verbose=False, merged_thresh=0.5):
        self.sensor = sensor
        self.bearing_res = unresolved_resolution
        self.sequential_resolution_update_flag = sequential_resolution_update_flag
        self._gate_level = gate_level
        self._verbose = verbose
        self._FOV = FOV
        self.merging_threshold = merged_thresh
        self._inn_cov_list = []   # each iteration, this will be populated with the innovation covariance of each target
        self._z_predict_list = []
        self._H_k_list = []

    def filter_most_likely(self, graph, beliefs, ownship, bearings):
        """
        performs filter update given the correct merging graph based on thresholding.
        :param graph:
        :param beliefs:
        :param ownship:
        :param bearings: numpy array of bearings
        :return:
        """

        n_targs = len(beliefs)
        y_dim = beliefs[0]._mean.shape[0]  # todo: case where no targets tracked
        z_dim = self.sensor.z_dim
        self._inn_cov_list = []
        self._z_predict_list = []
        self._H_k_list = []
        b_sigma = self.sensor.get_b_sigma()

        visited = set()
        meas = []
        targs_on_meas = []
        for i in range(n_targs):
            if i not in visited:
                visited.add(i)
                if not self.sensor.in_FOV(bearings[i]):
                    continue
                if graph.is_connected(i):
                    connected_targs = graph.get_connected_targets_raw_index(i)
                    for j in connected_targs:
                        visited.add(j)
                else:
                    connected_targs = []
                connected_targs.insert(0, i)
                bearings_to_merge = bearings[connected_targs]
                mean_bearing = bearings_to_merge.mean()
                meas.append(Measurement(mean_bearing, 0, 1))
                targs_on_meas.append(connected_targs)

        # now construct association mat from
        num_meas = len(meas)
        Omega = np.zeros((num_meas, n_targs))
        for i in range(num_meas):
            Omega[i, targs_on_meas[i]] = 1

        # now perform measurement update with correct data  association
        C_k = self.get_C_k_from_event_no_clutter(Omega)
        H_tilde = self.build_H_tilde(beliefs, ownship, z_dim)
        full_belief = self.get_full_predicted_belief(beliefs)
        measurement_updated_belief = self.perform_measurement_update_most_likely(full_belief, H_tilde, b_sigma,
                                                                                  meas, self._z_predict_list, C_k, Omega)
        # separate out each target beliefs
        output = []
        for i in range(n_targs):
            start = i * y_dim
            stop = start + y_dim
            mean = measurement_updated_belief._mean[start:stop]
            cov = measurement_updated_belief._cov[start:stop, start:stop]
            output.append(GaussianBelief(mean, cov))

        meas_as_list = []
        for item in meas:
            meas_as_list.append(item.getZ())
        return output, meas_as_list

    def perform_measurement_update_most_likely(self, full_belief, H_tilde, b_sigma, measurements, z_target_predict, C_k, Omega):
        """
        takes the correct data association as defined by C_k and performs the measurement update in one step
        :param full_belief:
        :param H_tilde:
        :param b_sigma:
        :param measurements:
        :param z_target_predict:
        :param FOV:
        :param C_k:
        :return:
        """

        if C_k is None:
            #print("Using prediction only")
            return full_belief
        #if True:
            #return full_belief
        else:
            #print("using measurement update")
            H_caron = C_k @ H_tilde
            Z_k_target, num_target_per_meas = self.get_target_generated_measurements_no_clutter(measurements, Omega)
            V = self.build_measurement_covariance_merged(b_sigma, num_target_per_meas)
            measurement_updated_belief = KalmanFilterMeasurementUpdate(full_belief._mean, full_belief._cov, H_caron, V,
                                                                       Z_k_target, C_k @ z_target_predict)
        return measurement_updated_belief

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

    def get_target_generated_measurements_no_clutter(self, measurements, Omega):
        """
        returns appropriate stacked measurement vector based on merging of targets as dictated by C_k
        :param measurements:
        :param Omega: association mat (no clutter)
        :return:
        """
        num_meas = Omega.shape[0]
        Z_k_target = np.zeros((num_meas, 1))
        num_targets_per_meas = np.zeros(num_meas)
        targ_meas_index = 0
        for i in range(num_meas):
            Z_k_target[targ_meas_index, 0] = measurements[i].getZ()
            num_targets_per_meas[targ_meas_index] = sum(Omega[i, :])
            targ_meas_index += 1
        return Z_k_target, num_targets_per_meas

    def get_C_k_from_event_no_clutter(self, event_mat):
        num_meas = event_mat.shape[0]
        num_targ = event_mat.shape[1]
        num_targ_meas = num_meas
        if num_targ_meas == 0:
            return None
        C_k = np.zeros((num_targ_meas, num_targ))
        targ_meas_index = 0

        for i in range(num_meas):
            target_indices = np.nonzero(event_mat[i, :])[0]
            C_k[targ_meas_index, target_indices] = 1/len(target_indices)
            targ_meas_index += 1

        return C_k


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

    def get_most_likely_graph(self, ownship, beliefs, sensor, thresh=0.5):

        bearings = get_bearings(ownship, beliefs, sensor)
        n_targs = len(beliefs)
        sorted_index = sorted(range(len(bearings)), key=lambda k: bearings[k], reverse=True)

        # construct feasible edge set
        feasible_edges = []
        for i in range(n_targs - 1):
            bearing_i = bearings[sorted_index[i]]
            bearing_j = bearings[sorted_index[i + 1]]
            if sensor.in_same_FOV(bearing_i, bearing_j):
                feasible_edges.append((sorted_index[i], sorted_index[i+1]))
        if n_targs > 2:
            bearing_i = bearings[sorted_index[n_targs - 1]]
            bearing_j = bearings[sorted_index[0]]
            if sensor.in_same_FOV(bearing_i, bearing_j):
                feasible_edges.append((sorted_index[n_targs - 1], sorted_index[0]))

        # now loop through feasible edges and decide whether edge is there or not based on threshold
        edges = []
        for edge in feasible_edges:
            bearing_i = bearings[edge[0]]
            bearing_j = bearings[edge[1]]
            prob = get_unresolved_prob_bearing(bearing_i, bearing_j, self.bearing_res)
            if prob >= thresh:
                edges.append(edge)

        most_likely_graph = Graph(n_targs, edges, feasible_edges)

        return most_likely_graph

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


def get_bearings(ownship, beliefs, sensor):
    """
    returns list of bearing in target order
    :param ownship:
    :param beliefs:
    :return:
    """
    bearings = []
    for i in range(len(beliefs)):
        bearing = sensor.observationModel(ownship, beliefs[i].getMean())
        bearings.append(bearing)

    return np.array(bearings)