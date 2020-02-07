import numpy as np
from utils import restrict_angle

class Measurement:
    def __init__(self, z, ID, size):
        self._z = z
        self._ID = ID
        self._z_dim = size

    def getID(self):
        return self._ID

class Sensor:
    def __init__(self, z_dim):
        self.z_dim = z_dim

class BearingSensor(Sensor):

    def __init__(self, min_range, max_range, min_hang, max_hang, b_sigma):
        Sensor.__init__(self, 1)              #scalar measurement for bearing only
        self._min_range = min_range
        self._max_range = max_range
        self._min_hang = min_hang
        self._max_hang = max_hang
        self._b_sigma = b_sigma

    def senseTargets(self, own_state, targets):

        output = []
        for target in targets:
            measurement = self.sense(own_state, target)
            output.append(measurement)

        return output


    def sense(self, x, target):
        y = target.getPosition()
        z = restrict_angle(np.arctan2(y[1] - x[1], y[0] - x[0]) - x[2])
        z += np.random.normal(0, self._b_sigma)

        #make z a np.array for further procerring down the road in Kalman Filtering
        return Measurement(np.array([z]), target.getID(), 1)


    def getJacobian(self, H, V, x, y):
        """
        :param H: reference to Sensor Jaobian
        :param V: reference to sensor noise covariance
        :param x: ownship state
        :param y: Predicted target state
        :return: no return, just modifying the H and V matrices passe by reference
        """
        #z = restrict_angle(np.arctan2(y[1] - x[1], y[0] - x[0]) - x[2]) # Don't need bearing for Jacobian
        range_squared = (y[0] - x[0])**2 + (y[1] - y[0])**2 # + 0.001 (prevent a divide by zero
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

