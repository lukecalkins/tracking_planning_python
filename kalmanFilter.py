import numpy as np

class GaussianBelief:
    def __init__(self, mean, cov):
        self._mean = mean
        self._cov = cov

    def getMean(self):
        return self._mean.copy()

    def getCovariance(self):
        return self._cov.copy()



def KalmanFilter(mean_prior, cov_prior, A, W, H, V, innovation, debug = 0):

    mean_predict = np.matmul(A, mean_prior)
    cov_predict = np.matmul(A, np.matmul(cov_prior, A.transpose())) + W
    R = H.dot(cov_predict.dot(H.transpose())) + V
    K = cov_predict.dot(H.transpose().dot(np.linalg.inv(R)))
    C = np.eye(len(mean_prior)) - K.dot(H)
    cov_update = C.dot(cov_predict)
    mean_update = mean_predict + K.dot(innovation)

    output = GaussianBelief(mean_update, cov_update)
    return output

def MultiTargetFilter(measurements, robot, debug = False):
    x_t = robot.getState()
    mean_prior = robot.tmm.getTargetState()
    cov_prior = robot.tmm.getCovarianceMatrix()
    #print(mean_prior)
    #print(cov_prior)

    num_targets = robot.tmm.num_targets()
    y_dim = int(robot.tmm.target_dim / num_targets)
    z_dim = robot.sensor.z_dim

    #allocate matrices
    A = np.zeros((num_targets * y_dim, num_targets * y_dim))
    W = np.zeros((num_targets * y_dim, num_targets * y_dim))
    H = np.zeros((num_targets * z_dim, num_targets * y_dim))
    V = np.zeros((num_targets * z_dim, num_targets * z_dim))

    #populate A and W matrices
    robot.tmm.getJacobian(A, W)

    #Allocate innoviation
    innovation = np.zeros(z_dim * num_targets)

    #populate H and V matrices by looping over measurements
    meas_index = 0
    for meas in measurements:
        target = robot.tmm.getTargetByID(meas.getID())
        y_predict = target.predictState(1)
        #print(y_predict)
        #print(target._state)

        H_i = np.zeros((z_dim, y_dim))
        V_i = np.zeros((z_dim, z_dim))

        #print(H_i)
        #print(V_i)
        robot.sensor.getJacobian(H_i, V_i, x_t, y_predict)
        #print(H_i)
        #print(V_i)
        H[meas_index*z_dim:(meas_index*z_dim + z_dim), meas_index*y_dim:(meas_index*y_dim + y_dim)] = H_i
        V[meas_index*z_dim:(meas_index*z_dim + z_dim), meas_index*z_dim:(meas_index*z_dim + z_dim)] = V_i
        z = meas._z
        h_xy = robot.sensor.observationModel(x_t, y_predict)
        innovation[meas_index:meas_index + z_dim] = robot.sensor.computeInnovation(z, h_xy)
        meas_index += 1

        #With multi-target matrices now in place, apply kalman filter to entire system
    result = KalmanFilter(mean_prior, cov_prior, A, W, H, V, innovation, debug=False)

    return result

def KalmanFilterCovAndInnovationCov(cov_prior, A, W, H, V, debug=False):
    """
    function to compute Kalman Filter update on covariance only and also return innovation covariance to be used in
    measurement gating
    :param cov_prior:
    :param A: target state transition matrix
    :param W: target state transition noise covariance
    :param H: linearized target measurement model matrix
    :param V: measurement model covariance
    :param debug: parameter to include verbose debugging
    :return: tuple of updated covariance and innovation covariance
    """

    cov_predict = A @ cov_prior @ A.transpose() + W
    innovation_cov = H @ cov_predict @ H.transpose() + V
    kalman_gain = cov_predict @ H.transpose() @ np.linalg.inv(innovation_cov)
    cov_update = (np.eye(cov_prior.shape[0]) - kalman_gain @ H) @ cov_predict

    return cov_update, innovation_cov