import numpy as np
from trackingLib.utils import restrict_angle
from copy import copy
import sys, os
from trackingLib.graph import Graph

class Measurement:
    def __init__(self, z, ID, size):
        self._z = z
        self._ID = ID
        self._z_dim = size

    def getID(self):
        return self._ID

    def getZ(self):
        return self._z

    def get_z_dim(self):
        return copy(self._z_dim)

class Sensor:
    def __init__(self, z_dim):
        self.z_dim = z_dim

class BearingSensor(Sensor):

    def __init__(self, min_range, max_range, min_hang, max_hang, b_sigma, detection_prob, FOV=None):
        Sensor.__init__(self, 1)              #scalar measurement for bearing only
        self._min_range = min_range
        self._max_range = max_range
        self._min_hang = min_hang
        self._max_hang = max_hang
        self._b_sigma = b_sigma
        self._detection_prob = detection_prob
        self._FOV = FOV * np.pi/180  # provided as degrees, convert to radians

    def get_b_sigma(self):
        return self._b_sigma

    def senseTargets(self, own_state, targets):

        output = []
        for target in targets:
            measurement = self.sense(own_state, target)
            prob = np.random.uniform(0, 1)
            if prob <= self._detection_prob:
                output.append(measurement)

        return output, len(targets)



    def senseTargets_FOV(self, own_state, targets):
        """
        returns bearing only sensors with a limited FOV
        :param own_state:
        :param targets:
        :return:
        """
        output = []
        num_targs_seen = 0
        for target in targets:
            sensor_output = self.sense_FOV(own_state, target)
            if sensor_output:
                output.append(sensor_output)
                num_targs_seen += 1

        return output, num_targs_seen

    def sense_FOV(self, own_state, target):

        target_state = target.getState()
        bearing = self.observationModel(own_state, target_state)
        noise = np.random.normal(0, self._b_sigma)
        z = restrict_angle(bearing + noise)
        if self.in_FOV(z):
            return Measurement(np.array([z]), target.getID(), 1)
        else:
            return None

    def in_FOV(self, bearing):
        """
        determines whether measurement is within FOV of the sensor
        :param bearing:
        :return:
        """
        delta = self._FOV / 2
        port_FOV = [np.pi/2 - delta, np.pi/2 + delta]
        starboard_FOV = [-np.pi/2 - delta, -np.pi/2 + delta]
        if port_FOV[0] <= bearing <= port_FOV[1]:
            return True
        elif starboard_FOV[0] <= bearing <= starboard_FOV[1]:
            return True
        else:
            return False

    def in_same_FOV(self, bearing_i, bearing_j):
        """
        return true if bearings are both in the field of view of the sensor
        :param bearing_i:
        :param bearing_j:
        :return:
        """
        if self.in_FOV(bearing_i):
            if self.in_FOV(bearing_j):
                bearing_i_side = self.get_FOV_side(bearing_i)
                bearing_j_side = self.get_FOV_side(bearing_j)
                if bearing_i_side == bearing_j_side:
                    return True
                else:
                    return False  # both in field of view but on opposite sides
            else:
                return False
        else:
            return False

    def get_FOV_side(self, bearing):
        """
        takes a bearing known to be within the FOV and returns whether it is port or starboard
        :param bearing:
        :return:
        """
        delta = self._FOV / 2
        port_FOV = [np.pi / 2 - delta, np.pi / 2 + delta]
        starboard_FOV = [-np.pi / 2 - delta, -np.pi / 2 + delta]
        if port_FOV[0] <= bearing <= port_FOV[1]:
            return "port"
        elif starboard_FOV[0] <= bearing <= starboard_FOV[1]:
            return "starboard"
        else:
            exit("get_FOV_side called on bearing not within field of view")



    def senseTargets_ambiguity(self, own_state, targets):
        """
        returns contacts in the range 0 to pi measured from the front of the sensor going in either direction
        :param own_state:
        :param targets:
        :return:
        """

        output = []
        for target in targets:
            target_state = target.getState()
            delta_east = target_state[0] - own_state[0]
            delta_north = target_state[1] - own_state[1]
            beta = np.arctan2(delta_north, delta_east)
            aspect = unsigned_angular_difference(own_state[2], beta)
            noise = np.random.normal(0, self._b_sigma)
            z = aspect + noise
            if z < 0:
                z = -1 * z
            elif z > np.pi:
                z = 2 * np.pi - z
            output.append(Measurement(np.array([z]), target.getID(), 1))

        return output, len(targets)

    def senseTargets_interference_2(self, own_state, targets, proximity):
        """
        function that will create vector of target measurements for 2 targets only where targets close in bearing are
        masked. If targets are within proimity parameter in bearing, only the louder one is sensed.
        :param own_state: state vector of ownship (pos_x, pos_y, heading)
        :param targets: list of ground truth targets
        :param proximity: proximity (in degrees) in which targets mask each other
        :return:
        """

        target_dist = []
        output = []
        masked = False
        for target in targets:
            dist = np.linalg.norm(target.getPosition() - own_state[0:2])
            target_dist.append(dist)
            measurement = self.sense(own_state, target)
            output.append(measurement)

        # calculate distance between 2 targets
        delta_bearing = restrict_angle(np.linalg.norm(output[0].getZ() - output[1].getZ()))
        print("Delta bearing = ", delta_bearing*180./np.pi, " degrees")
        if np.abs(delta_bearing) < (np.pi/180 * proximity):
            # find which target is closer to sensor and delete that targets measurement
            max_ndx = np.argmax(target_dist)
            del output[max_ndx]
            print("Target marking occuring, number of target measurements occuring = ", len(output))
            masked = True

        return output, masked

    def senseTargets_interference_n(self, own_state, targets, proximity):
        proximity = np.pi/180 * proximity
        bearings = []
        targ_dists = []
        meas_list = []
        true_bearings = []
        for target in targets:
            dist = np.linalg.norm(target.getPosition() - own_state[0:2])
            targ_dists.append(dist)
            measurement = self.sense(own_state, target)
            meas_list.append(measurement)
            true_bearings.append(self.observationModel(own_state, target.getPosition()))

        proximity_list = []
        for i in range(len(targets)):
            targ_proximity  = []
            for j in range(len(targets)):
                if np.abs(true_bearings[i] - true_bearings[j]) < proximity:
                    if j != i:
                        targ_proximity.append(j)
            proximity_list.append(targ_proximity)

        masked_indicator = np.zeros(len(targets))
        for i in range(len(proximity_list)):
            for j in range(len(proximity_list[i])):
                targ_to_check = proximity_list[i][j]
                if targ_dists[i] > targ_dists[targ_to_check]:
                    masked_indicator[i] = 1

        #loop over indicators and only add to output if not masked
        output = []
        for i in range(len(masked_indicator)):
            if masked_indicator[i] != 1:
                output.append(meas_list[i])

        num_targs_seen = len(targets) - masked_indicator.sum()
        return output, int(num_targs_seen)

    def senseTargets_resolution_model_2(self, own_state, targets, bearing_res):
        """
        for 2 targets, this will return measurements according to the resolution model used in the merged measurement
        tracker
        :param own_state:
        :param targets:
        :param bearing_res:
        :return:
        """

        #find merging probability
        true_bearings = []
        measurements = []
        for target in targets:
            true_bearings.append(self.observationModel(own_state, target.getPosition()))
            measurements.append(self.sense(own_state, target))
        true_bearings = np.array(true_bearings)
        true_bearings_2pi = true_bearings + np.pi

        #calculate probability of targets merging
        delta_bearing = np.abs(true_bearings[0] - true_bearings[1])
        if delta_bearing > np.pi:
            delta_bearing = 2*np.pi - delta_bearing
        p_merged = np.exp(-1/2 * delta_bearing * 1./(bearing_res ** 2) * delta_bearing)
        print("Merging Probability: ", p_merged)
        rand_draw = np.random.uniform()
        if p_merged > rand_draw:
            # measurement of the mean target bearing
            mean_bearing = true_bearings_2pi.mean() - np.pi
            noise = np.random.normal(0, 2 * self._b_sigma)
            measurement = mean_bearing + noise
            measurement = Measurement(np.array([measurement]), None, 1)
            return [measurement], 1
        else:
            return measurements, 2

    def senseTargets_resolution_model_n(self, own_state, targets, bearing_res):
        """
        for any number of targets, this function will return the number of targets according to the resolution model in
        the merged measurement tracker
        :param own_state:
        :param targets:
        :param bearing_res:
        :return:
        """
        n_targs = len(targets)
        true_bearings = []
        for target in targets:
            true_bearings.append(self.observationModel(own_state, target.getState()))

        # sort the bearings
        true_bearings = np.array(true_bearings)
        sorted_index = sorted(range(len(true_bearings)), key=lambda k: true_bearings[k], reverse=True)

        # construct feasible edge set
        feasible_edges = []
        for i in range(n_targs - 1):
            bearing_i = true_bearings[sorted_index[i]]
            bearing_j = true_bearings[sorted_index[i + 1]]
            bearing_diff = bearing_i - bearing_j
            if n_targs > 2:
                if bearing_diff < np.pi:
                    feasible_edges.append((sorted_index[i], sorted_index[i + 1]))
            else:
                feasible_edges.append((sorted_index[i], sorted_index[i + 1]))
        if n_targs > 2:
            bearing_i = true_bearings[sorted_index[n_targs - 1]]
            bearing_j = true_bearings[sorted_index[0]]
            bearing_difference = bearing_i - bearing_j
            if bearing_difference + 2 * np.pi < np.pi:
                feasible_edges.append((sorted_index[n_targs - 1], sorted_index[0]))

        # only consider feasible edges when evaluating merging
        merged_mat = np.zeros((n_targs, n_targs))
        merged_edges = []
        for edge in feasible_edges:
            bearing_i = true_bearings[edge[0]]
            bearing_j = true_bearings[edge[1]]
            delta_bearing = np.abs(bearing_i - bearing_j)  # todo: delta changes when wrapped around in bearing
            if delta_bearing > np.pi:
                delta_bearing = 2*np.pi - delta_bearing
            p_merged = np.exp(-1./2 * delta_bearing * 1./(bearing_res ** 2) * delta_bearing)
            rand_draw = np.random.uniform()
            if rand_draw < p_merged:  #targets are merged
                merged_mat[edge[0], edge[1]] = 1
                merged_mat[edge[1], edge[0]] = 1  # symmetric
                merged_edges.append(edge)

        merged_graph = Graph(n_targs, merged_edges, feasible_edges)

        # Generate appropriate number of measurements using merged measurement matrix
        visited = set()
        num_targs_seen = 0
        meas_list = []
        for i in range(n_targs):
            if i not in visited:
                conn_seq = merged_graph.get_connected_edge_sequence(i)
                connected_targs = merged_graph.get_connected_targets_raw_index(i)
                visited.add(i)
                num_targs_on_meas = 1
                for k in range(len(connected_targs)):
                    visited.add(connected_targs[k])
                    num_targs_on_meas += 1
                targs_on_meas = np.concatenate((np.array([i]), np.array(connected_targs, dtype=int)))
                if conn_seq:
                    if n_targs == 2: # check if connected sequence needs to be flipped  to go across boundary in correct direction
                        if true_bearings[conn_seq[0][0]] - true_bearings[conn_seq[0][1]] > np.pi:
                            conn_seq = [(conn_seq[0][1], conn_seq[0][0])]
                    mean_bearing = generate_mean_bearing(true_bearings,  conn_seq)
                else:
                    mean_bearing = true_bearings[i]
                noise = np.random.normal(0, num_targs_on_meas * self._b_sigma)
                bearing_meas = Measurement(restrict_angle(mean_bearing + noise), None, 1)
                num_targs_seen += 1
                meas_list.append(bearing_meas)

        return meas_list, num_targs_seen

    def senseTargets_resolution_model_n_FOV(self, own_state, targets, bearing_res):
        """
        Applies resolution model in
        :param own_state:
        :param targets:
        :return:
        """

        n_targs = len(targets)
        true_bearings = []
        for target in targets:
            true_bearings.append(self.observationModel(own_state, target.getState()))

        # sort the bearings
        true_bearings = np.array(true_bearings)
        true_bearings_2pi = true_bearings + np.pi
        sorted_index = sorted(range(len(true_bearings)), key=lambda k: true_bearings[k], reverse=True)

        # construct feasible edge set (can only merge with targets in same FOV)
        feasible_edges = []
        for i in range(n_targs - 1):
            bearing_i = true_bearings[sorted_index[i]]
            bearing_j = true_bearings[sorted_index[i + 1]]
            if self.in_same_FOV(bearing_i, bearing_j):
                feasible_edges.append((sorted_index[i], sorted_index[i + 1]))
        if n_targs > 2:
            bearing_i = true_bearings[sorted_index[n_targs - 1]]
            bearing_j = true_bearings[sorted_index[0]]
            if self.in_same_FOV(bearing_i, bearing_j):
                feasible_edges.append((sorted_index[n_targs - 1], sorted_index[0]))

        # only consider feasible edges when evaluating merging
        merged_mat = np.zeros((n_targs, n_targs))
        merged_edges = []
        for edge in feasible_edges:
            bearing_i = true_bearings[edge[0]]
            bearing_j = true_bearings[edge[1]]
            delta_bearing = np.abs(bearing_i - bearing_j)  # todo: delta changes when wrapped around in bearing
            if delta_bearing > np.pi:
                delta_bearing = 2*np.pi - delta_bearing
            p_merged = np.exp(-1./2 * delta_bearing * 1./(bearing_res ** 2) * delta_bearing)
            rand_draw = np.random.uniform()
            if rand_draw < p_merged:  #targets are merged
                merged_mat[edge[0], edge[1]] = 1
                merged_mat[edge[1], edge[0]] = 1  # symmetric
                merged_edges.append(edge)

        merged_graph = Graph(n_targs, merged_edges, feasible_edges)

        # Generate appropriate number of measurements using merged measurement matrix
        visited = set()
        num_targs_seen = 0
        meas_list = []
        for i in range(n_targs):
            if i not in visited:
                visited.add(i)
                if not self.in_FOV(true_bearings[i]):
                    continue
                connected_targs = merged_graph.get_connected_targets_raw_index(i)
                num_targs_on_meas = 1
                for k in range(len(connected_targs)):
                    visited.add(connected_targs[k])
                    num_targs_on_meas += 1
                targs_on_meas = np.concatenate((np.array([i]), np.array(connected_targs, dtype=int)))
                mean_bearing = true_bearings[targs_on_meas].mean()
                noise = np.random.normal(0, num_targs_on_meas * self._b_sigma)
                bearing_meas = Measurement(restrict_angle(mean_bearing + noise), None, 1)
                num_targs_seen += 1
                meas_list.append(bearing_meas)

        return meas_list, num_targs_seen

    def sense(self, x, target):
        y = target.getPosition()
        z = restrict_angle(np.arctan2(y[1] - x[1], y[0] - x[0]) - x[2])
        noise = np.random.normal(0, self._b_sigma)
        z = restrict_angle(z + noise)

        #make z a np.array for further procerring down the road in Kalman Filtering
        return Measurement(np.array([z]), target.getID(), 1)


    def getJacobian(self, H, V, x, y):
        """
        :param H: reference to Sensor Jaobian
        :param x: ownship state
        :param y: Predicted target state
        :return: no return, just modifying the H and V matrices passed by reference
        """
        #z = restrict_angle(np.arctan2(y[1] - x[1], y[0] - x[0]) - x[2]) # Don't need bearing for Jacobian
        range_squared = (y[0] - x[0])**2 + (y[1] - x[1])**2 # + 0.001 (prevent a divide by zero
        H[0, 0] = (x[1] - y[1])/range_squared
        H[0, 1] = (y[0] - x[0])/range_squared
        V[:] = self._b_sigma ** 2

    def computeInnovation(self, measurement, predicted_measurement):
        innovation = measurement - predicted_measurement
        for i in range(len(measurement)):
            innovation[i] = restrict_angle(innovation[i])

        return innovation

    def observationModel(self, x, y):
        """

        :param x: ownship vector
        :param y: target vector
        :return:
        """
        return restrict_angle(np.arctan2(y[1] - x[1], y[0] - x[0]) - x[2])

    def observationModel_ambiguity(self, x, y):
        """
        returns positive angle in the range of 0 to pi with left-right ambiguity
        :param x:
        :param y:
        :return:
        """
        beta = np.arctan2(y[1] - x[1], y[0] - x[0])
        abs_difference = np.abs(x[2] - beta)
        if abs_difference > np.pi:
            output = 2*np.pi - abs_difference
        else:
            output = abs_difference
        return output


def unsigned_angular_difference(heading, beta):
    """
    given an ownship heading and a relative target bearing beta, this function returns the shortest unsigned
    angular distance between 0 and pi
    :param heading:
    :param beta:
    :return:
    """
    abs_difference = np.abs(heading - beta)
    if abs_difference > np.pi:
        output = 2*np.pi - abs_difference
    else:
        output = abs_difference
    return output

def generate_mean_bearing(bearings, conn_seq):
    """
    function that generates the mean value of bearing given that the bearing could wrap around the -pi to pi barrier.
    :param bearing_array: np.array of bearings GIVEN IN POSITIVE VALUES -PI to PI
    :param conn_seq: sequence of edges in connected group starting from greatest bearing (unless greatest is wrapped)
    :return: bearing value
    """

    # get target indices in correct order
    ordered_targs = [conn_seq[0][0]]
    mapped_bearings = []
    for i in range(len(conn_seq)):
        ordered_targs.append(conn_seq[i][1])
    for i in range(len(ordered_targs)):
        if i == 0:
            mapped_bearings.append(bearings[ordered_targs[i]])
            continue
        if mapped_bearings[i-1] < 0:
            if bearings[ordered_targs[i]] > 0:
                mapped_bearings.append(restrict_angle(bearings[ordered_targs[i]],  -2*np.pi, 0))
            else:
                mapped_bearings.append(bearings[ordered_targs[i]])
        else:
            mapped_bearings.append(bearings[ordered_targs[i]])

    # take mean of mapped bearing and map it back to -pi to pi
    mean = restrict_angle((np.array(mapped_bearings)).mean(), -np.pi, np.pi)

    return mean


def add_clutter(measurements, density, volume = 2*np.pi):
    """
    function to take a list of measurements and add additional clutter to it given a clutter density/unit volume. For a
    1D sensing problem such as bearing only, volume = length of bearing FOV.
    :param measurements:
    :param density:
    :param volume:
    :return:
    """

    # draw from Poisson distribution with rate = density*volume
    rate = density * volume
    num_clutter = np.random.poisson(rate, 1)

    #However many clutter measurements are generated, uniformly place them in the "volume"
    if num_clutter != 0:
        bearings = np.random.uniform(0, volume, num_clutter)
        for i in range(bearings.shape[0]):
            measurements.append(Measurement(bearings[i], 0, 1))  # all clutter measurements get ID of 0