from kalmanFilter import GaussianBelief
import numpy as np
import itertools as IT
from copy import copy, deepcopy
from utils import gaussian_pdf


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

        association_probability_matrix = np.zeros((Omega.shape[0] + 1, Omega.shape[1]))  # no extra row for probability not detected
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


class JPDAF_merged_simulate:

    def __init__(self, sensor, gate_level=0.99, verbose=False):
        self.sensor = sensor
        self._gate_level = gate_level
        self._verbose = verbose
        self._inn_cov_list = []   # each iteration, this will be populated with the innovation covariance of each target
        self._z_predict_list = []
        self._H_k_list = []

    def filter(self, measurements, robot, ownship):
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

        Omega = self.gate_measurements(measurements, robot, self._gate_level)
        if self._verbose:
            print("JPDAF_merged, Omega matrix: ", Omega)
