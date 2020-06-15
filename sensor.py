import numpy as np
from utils import restrict_angle
from copy import copy

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

    def __init__(self, min_range, max_range, min_hang, max_hang, b_sigma, detection_prob):
        Sensor.__init__(self, 1)              #scalar measurement for bearing only
        self._min_range = min_range
        self._max_range = max_range
        self._min_hang = min_hang
        self._max_hang = max_hang
        self._b_sigma = b_sigma
        self._detection_prob = detection_prob

    def get_b_sigma(self):
        return self._b_sigma

    def senseTargets(self, own_state, targets):

        output = []
        for target in targets:
            measurement = self.sense(own_state, target)
            prob = np.random.uniform(0, 1)
            if prob <= self._detection_prob:
                output.append(measurement)

        return output

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

    def sense(self, x, target):
        y = target.getPosition()
        z = restrict_angle(np.arctan2(y[1] - x[1], y[0] - x[0]) - x[2])
        z += np.random.normal(0, self._b_sigma)

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