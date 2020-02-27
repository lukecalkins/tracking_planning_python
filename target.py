import numpy as np

class Target:

    def __init__(self, state_init, cov_pos, cov_vel, samp, ID, y_dim):
        self._A = self.constructDynamics(samp)
        self._W = self.constructNoise(cov_pos, cov_vel)
        self._state = np.array([state_init]).transpose()  # store as column vector
        self._ID = ID
        self._y_dim = y_dim

    def constructDynamics(self, samp):

        A = np.eye(4, 4)
        A[0:2, 2:4] = samp * np.eye(2)
        return A

    def constructNoise(self, cov_pos, cov_vel):

        noise = np.zeros((4, 4))
        noise[0:2, 0:2] = cov_pos * np.eye(2)
        noise[2:4, 2:4] = cov_vel * np.eye(2)
        return noise

    def forwardSimulate(self, steps):

        self._state = self._A.dot(self._state) # + random noise drawn

    def getState(self):
        return np.copy(self._state)

    def getPosition(self):
        return self._state[0:2]

    def getID(self):
        return self._ID

    def getJacobian(self):
        return self._A

    def getNoise(self):
        return self._W

##############################################################

class InfoTarget(Target):

    def __init__(self, state_init, cov_pos, cov_vel, samp, ID, y_dim, cov_pos_init, cov_vel_init):
        Target.__init__(self, state_init, cov_pos, cov_vel, samp, ID, y_dim)
        self.covariance = self.constructNoise(cov_pos_init, cov_vel_init)

    def getCovariance(self):
        return self.covariance

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

##############################################################
######## Target Models #######################################
##############################################################

class TargetModel:

    def __init__(self):
        self.targets = {}
        self.target_dim = 0

    def addTarget(self, ID, target):
        self.targets[ID] = target
        self.target_dim += target._y_dim

    def getTargetState(self):

        result = np.array([])
        index = 0
        for key in self.targets.keys():
            result = np.append(result, self.targets[key].getState())

        return result

    def num_targets(self):
        return len(self.targets)

    def getSystemMatrix(self):
        result = np.zeros((self.target_dim, self.target_dim))
        index = 0
        for key in self.targets.keys():
            target = self.targets[key]
            result[index:index + target._y_dim, index:index + target._y_dim] = target.getJacobian()
            index += target._y_dim

        return result

    def getNoiseMatrix(self):
        result = np.zeros((self.target_dim, self.target_dim))
        index = 0
        for key in self.targets.keys():
            target = self.targets[key]
            result[index:index + target._y_dim, index:index + target._y_dim] = target.getNoise()
            index += target._y_dim

        return result

##############################################################

class InfoTargetModel(TargetModel):

    def __init__(self):
        TargetModel.__init__(self)

    def addTarget(self, ID, infoTarget):
        self.targets[ID] = infoTarget
        self.target_dim += infoTarget._y_dim

    def getCovarianceMatrix(self):

        result = np.zeros((self.target_dim, self.target_dim))
        index = 0
        for key in self.targets.keys():
            target = self.targets[key]
            result[index:index + target._y_dim, index:index + target._y_dim] = target.getCovariance()
            index += target._y_dim

        return result

    def getJacobian(self, A_, W_):
        A_[:, :] = self.getSystemMatrix()
        W_[:, :] = self.getNoiseMatrix()

    def getTargetByID(self, ID):
        return self.targets[ID]

    def updateBelief(self, belief):

        index = 0
        for key in self.targets.keys():
            target = self.targets[key]
            targ_mean = belief._mean[index:index + target._y_dim]
            targ_cov = belief._cov[index:index + target._y_dim, index:index + target._y_dim]
            target.updateBelief(targ_mean, targ_cov)

            index = index + target._y_dim









