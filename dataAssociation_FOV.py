import itertools as IT
import numpy as np
from utils import restrict_angle, kron_delta, inside_interval
from sensor import Measurement
from kalmanFilter import GaussianBelief, KalmanFilterMeasurementUpdate
from scipy.stats import poisson, multivariate_normal, norm
from scipy.linalg import sqrtm
from math import factorial
from graph import Graph, get_pi_i_j

sqrt = np.sqrt

from copy import copy
from collections import deque

class JPDAF_amb:

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
        self._inn_cov_list = []
        self._z_predict_list = []
        self._H_k_list = []
        self._gain_matrix_list = []
        self._gate_volume_list = []

        beliefs = self.get_predicted_beliefs(robot)

        Omega = self.gate_measurements(measurements, robot, beliefs)
        if self._verbose:
            print("Omega: ", Omega, "\n")

        event_matrices = self.generate_association_events(Omega)
        event_probabilities = self.get_event_probabilities(event_matrices, measurements, robot, self._clutter_density)

        #calculate association probabilites
        association_probability_matrix = np.zeros((Omega.shape[0] + 1, Omega.shape[1] - 1))
        self.get_association_probabilities(association_probability_matrix, event_matrices, event_probabilities)

        #with association probabilities, the weighted innovation and innovation covariance for each target can be calculated
        self.update_estimates(association_probability_matrix, measurements, robot, beliefs)

    def update_estimates(self, prob_mat, measurements, robot, beliefs):
        """
        function that obtains the weighted innovation and weighted covariance matrix and updates the respective targets
        mean and covariances through the EKF
        :param prob_mat: association prob matrix with num_rows=num_measurements + 1 (not detected), num_cols = num_targets
        :param robot: robot object
        :return:
        """

        #loop over targets, calculate weighted innovation and weighted inovtaion covariance
        for i in range(len(robot.tmm.targets)):
            weighted_innovation = self.get_weighted_innovation(prob_mat[:, i], measurements, self._z_predict_list[i])
            W_k = self._gain_matrix_list[i]
            S_k = self._inn_cov_list[i]
            P_k = self.get_P_k(prob_mat[:, i], measurements, self._z_predict_list[i], weighted_innovation, W_k)
            beta_0 = prob_mat[-1, i]
            covariance_update = beliefs[i]._cov - (1 - beta_0) * W_k @ S_k @ W_k.T + P_k
            mean_update = beliefs[i]._mean + W_k @ weighted_innovation
            info_target = robot.tmm.targets[i]
            info_target.updateBelief(mean_update, covariance_update)

    def get_P_k(self, prob_mat_column, measurements, z_predict, weighted_innovation, W_k):
        """
        function to generate the P_k matrix (fortman1983sonar) used in covariance update equation
        :param prob_mat_column: association probability matrix column corresponding to target in question
        :param measurements:
        :param z_predict:
        :param weighted_innovation:
        :param W_k: filter gain matrix for the target
        :return:
        """

        weighted_innovation_outer_product = np.zeros((weighted_innovation.shape[0], weighted_innovation.shape[0]))
        for i in range(len(measurements)):
            if prob_mat_column[i] != 0:
                innovation = measurements[i].getZ() - z_predict
                weighted_innovation_outer_product += prob_mat_column[i] * (innovation @ innovation.T)

        P_k = W_k @ (weighted_innovation_outer_product - weighted_innovation @ weighted_innovation.T) @ W_k.T

        return P_k

    def get_weighted_innovation(self, prob_mat_column, measurements, z_predict):
        """
        function that return the weighted innovation base on probabilities of association. The info_target already has
        its mean_predicted ans stored in the object
        :param prob_mat_column: only the column of the association matrix relavent to the target in question
        :param measurements:
        :param info_target: target in question getting weighted innovation for
        :return: weighted innovation measurement
        """
        weighted_innovation = np.array([[0.]])
        for i in range(prob_mat_column.shape[0] - 1):   # subtract 1 becaus last row is beta_0^t
            if prob_mat_column[i] != 0:
                weighted_innovation += prob_mat_column[i] * (measurements[i].getZ() - z_predict)

        return weighted_innovation

    def get_association_probabilities(self, prob_mat, event_matrices, event_probabilities):
        """
        functiont that takes the full probability matrix as reference and each event matrix and
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
                    likelihood = self.get_measurement_likelihood_ambiguity(event[i, :], measurements[i], robot)
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

    def get_measurement_likelihood_ambiguity(self, event_row, measurement, robot):
        """
        functiont that calculates the likelihood of a measuerment given that it is associated to a particulat target
        :param event_row: the row of the even matrix corresponding to the measurement associated
        :param measurement: measurement object associated to the target
        :param robot: robot in which the JPDA filter is running
        :return: value of gaussian likelihood function
        """

        #find the correct target
        target_ndx = np.argmax(event_row) - 1  # -1 because first column represents false alarms

        #this info target already has a predicted mean and innovation covariance stored from gating measurements
        innovation = measurement.getZ() - self._z_predict_list[target_ndx]
        return self.gaussian_likelihood(innovation, self._inn_cov_list[target_ndx])

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

    def gate_measurements(self, measurements, robot, beliefs):
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
        y_dim = robot.tmm.target_dim // num_targets
        z_dim = robot.sensor.z_dim
        n_rows = len(measurements)
        # n_cols = robot.tmm.num_targets()
        Omega = np.zeros((n_rows, num_targets + 1))  # add another column for false alarm measurements

        for i in range(len(robot.tmm.targets)):
            targ_belief = beliefs[i]
            H = np.zeros((z_dim, y_dim))
            V = np.zeros((z_dim, z_dim))
            robot.sensor.getJacobian(H, V, x_t, targ_belief._mean)
            z_predict_unambiguous = robot.sensor.observationModel(x_t, targ_belief._mean)
            if z_predict_unambiguous < 0:
                H = -1 * H
            z_predict = np.abs(z_predict_unambiguous)
            self._H_k_list.append(H)
            self._z_predict_list.append(z_predict)
            inn_cov = H @ targ_belief._cov @ H.T + V
            self._inn_cov_list.append(inn_cov)
            self._gain_matrix_list.append(targ_belief._cov @ H.T @  np.linalg.inv(inn_cov))
            gate_volume = self.get_gate_volume(inn_cov)
            self._gate_volume_list.append(gate_volume)

            for j in range(n_rows):
                Omega[j, i + 1] = self.gateMeasurementToTarget_ambiguity(z_predict, measurements[j], gate_volume)

        for i in range(n_rows):
            Omega[i, 0] = 1

        return Omega

    def gateMeasurementToTarget_ambiguity(self, z_predict, measurement, volume):
        """
        Compute bearing gate around target based on mahaolobis distance and given level
        :param target: info target stored on robot
        :param level: confidence level of the Gaussian to use for gating
        :return:
        """

        delta = volume / 2
        gate_min = z_predict - delta
        if gate_min < 0:
            gate_min = 0
        gate_max = z_predict + delta
        if gate_max > np.pi:
            gate_max = np.pi
        if inside_interval(measurement.getZ(), gate_min, gate_max):
            return 1
        else:
            return 0



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
