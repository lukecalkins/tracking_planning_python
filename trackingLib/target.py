import numpy as np
from copy import copy, deepcopy
from trackingLib.kalmanFilter import GaussianBelief

class Target:

    def __init__(self, state_init, cov_pos, cov_vel, samp, ID, y_dim, process_noise):
        self._A = self.constructDynamics(samp)
        #self._W = self.constructNoise(cov_pos, cov_vel)
        self._W = self.construct_process_noise_covariance(samp, process_noise)
        self._state = np.array([state_init]).transpose()  # store as column vector
        self._ID = ID
        self._y_dim = y_dim
        self.samp = samp
        self.process_noise = process_noise
        self.trajectory = []

    def constructDynamics(self, samp):

        A = np.eye(4, 4)
        A[0:2, 2:4] = samp * np.eye(2)
        return A

    def constructNoise(self, cov_pos, cov_vel):

        noise = np.zeros((4, 4))
        noise[0:2, 0:2] = cov_pos * np.eye(2)
        noise[2:4, 2:4] = cov_vel * np.eye(2)
        return noise

    def construct_process_noise_covariance(self, samp, process_noise):
        """
        based on random acceleration disturbance
        :param q0: acceleration
        :return:
        """
        noise = np.zeros((4, 4))
        noise[0, 0] = 1/4 * samp ** 4
        noise[0, 2] = 1/2 * samp ** 3
        noise[1, 1] = 1/4 * samp ** 4
        noise[1, 3] = 1/2 * samp ** 3
        noise[2, 0] = 1/2 * samp ** 3
        noise[2, 2] = samp ** 2
        noise[3, 1] = 1/2 * samp ** 3
        noise[3, 3] = samp ** 2

        return process_noise * noise

    def forwardSimulate(self, steps):

        self._state = self._A.dot(self._state) # random noise drawn

    def add_fixed_trajectory(self, traj):
        self.trajectory = traj

    def forward_simulate_fixed_trajectory(self, time_step):
        self._state = self.trajectory[time_step]

    def getState(self):
        return deepcopy(self._state)

    def getPosition(self):
        return np.copy(self._state[0:2])

    def getVelocity(self):
        return np.copy(self._state[2:4])

    def getID(self):
        return self._ID

    def getJacobian(self, dt=None):
        #return np.copy(self._A)
        if dt is None:
            dt = self.samp
        return self.constructDynamics(dt)

    def getNoise(self, dt=None):
        #return np.copy(self._W)
        if dt is None:
            dt = self.samp
        return self.construct_process_noise_covariance(dt, self.process_noise)

##############################################################

class InfoTarget(Target):

    def __init__(self, state_init, cov_pos, cov_vel, process_noise, samp, ID, y_dim, cov_pos_init, cov_vel_init):
        Target.__init__(self, state_init, cov_pos, cov_vel, samp, ID, y_dim, process_noise)
        self.covariance = self.constructNoise(cov_pos_init, cov_vel_init)

    def getCovariance(self):
        return deepcopy(self.covariance)

    def predictState(self, steps):
        current = np.copy(self._state)
        return self._A.dot(current)

    def updateBelief(self, mean, cov):
        self._state = mean
        self.covariance = cov

    def predictMeanAndCovariance(self, T=1):  # todo: add multiple prediction steps
        self.mean_predict = self._A.dot(self._state)
        self.cov_predict = np.matmul(self._A, np.matmul(self.covariance, self._A.transpose())) + self._W

    def set_z_predict_and_innovation_covariance(self, z_predict, H, V):
        self.z_predict = np.array([z_predict])
        self.innovation_cov = np.matmul(H, np.matmul(self.cov_predict, H.transpose())) + V
        self.filter_gain_matrix = self.cov_predict @ H.transpose() @ np.linalg.inv(self.innovation_cov)

    def set_gate_volume(self, level):
        """
        Set the gate volume given the current target prediction
        :param level:
        :return:
        """
        if level == 0.95:
            k_alpha = 3.84
        elif level == 0.99:
            k_alpha = 6.64
        elif level == 0.999:
            k_alpha = 10.83
        else:
            print("Desired gating level not found")

        self.gate_volume = 2 * np.sqrt(k_alpha) * np.sqrt(np.linalg.det(self.innovation_cov))
        print("Gate volume: ", self.gate_volume * 180/np.pi, " degrees")

##############################################################
######## Target Models #######################################
##############################################################

class TargetModel:

    def __init__(self, map_coord=None):
        self.targets = []
        self.targetIDs = []
        self.target_dim = 0
        self.map_coord = map_coord  # list of minimum and maximum coordinates


    def addTarget(self, ID, target):
        self.targets.append(target)
        self.targetIDs.append(ID)
        self.target_dim += target._y_dim

    def getTargets(self):
        return self.targets

    def getTargetState(self):

        result = np.array([])
        index = 0
        for target in self.targets:
            result = np.append(result, target.getState())

        return result

    def num_targets(self):
        return len(self.targets)

    def getSystemMatrix(self):
        result = np.zeros((self.target_dim, self.target_dim))
        index = 0
        for target in self.targets:
            result[index:index + target._y_dim, index:index + target._y_dim] = target.getJacobian()
            index += target._y_dim

        return result

    def getNoiseMatrix(self):
        result = np.zeros((self.target_dim, self.target_dim))
        index = 0
        for target in self.targets:
            result[index:index + target._y_dim, index:index + target._y_dim] = target.getNoise()
            index += target._y_dim

        return result

    def forwardSimulate(self, T=1):

        for target in self.targets:
            target.forwardSimulate(T)

            #check for outside mapmin
            """
            if target.getPosition()[0] <= self.map_coord[0][0] or target.getPosition()[1] <= self.map_coord[0][1]:
                target._state[2:4] = -1 * target._state[2:4]
            elif target.getPosition()[0] >= self.map_coord[1][0] or target.getPosition()[1] >= self.map_coord[1][1]:
                target._state[2:4] = -1 * target._state[2:4]
            """

    def forwardSimulate_fixed_trajectory(self, time_step):
        for target in self.targets:
            target.forward_simulate_fixed_trajectory(time_step)

##############################################################

class InfoTargetModel(TargetModel):

    def __init__(self):
        TargetModel.__init__(self)
        self.samp = None  # will initialize after adding first target

    def addTarget(self, ID, infoTarget):
        if not self.targets:
            self.samp = infoTarget.samp
        self.targets.append(infoTarget)
        self.targetIDs.append(ID)
        self.target_dim += infoTarget._y_dim



    def getCovarianceMatrix(self):

        result = np.zeros((self.target_dim, self.target_dim))
        index = 0
        for target in self.targets:
            result[index:index + target._y_dim, index:index + target._y_dim] = target.getCovariance()
            index += target._y_dim

        return result

    def getJacobian(self, A_, W_):
        """
        sets system wide noise and covariance by passing as reference
        :param A_:
        :param W_:
        :return:
        """
        A_[:, :] = self.getSystemMatrix()
        W_[:, :] = self.getNoiseMatrix()

    def getTargetByID(self, ID):
        return self.targets[ID]

    def updateBelief(self, belief):

        index = 0
        for target in self.targets:
            targ_mean = belief._mean[index:index + target._y_dim]
            targ_cov = belief._cov[index:index + target._y_dim, index:index + target._y_dim]
            target.updateBelief(targ_mean, targ_cov)

            index = index + target._y_dim


    def predictTargetState(self, mean, T):
        """
        function that will take the sytem of targets stored in the info model and precit T steps into the future
        :param T: Number of timesteps
        :return: list of targets states (length T + 1)
        """

        target_history = []
        target_state = mean
        A = self.getSystemMatrix()
        for i in range(T + 1):
            target_history.append(target_state)
            target_state = A @ target_state

        return np.array(target_history)

    def get_system_belief_copy(self):
        """
        returns copy of system (multiple targets) belief (mean and covariance)
        :return:
        """
        mean = self.getTargetState()
        cov = self.getCovarianceMatrix()

        output = GaussianBelief(mean, cov)
        return output





