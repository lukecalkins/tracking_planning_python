import numpy as np


class Robot:

    def __init__(self, init_state, sensor, info_target_model, samp):
        self._state = init_state
        self.sensor = sensor
        self.tmm = info_target_model

    #def senseTargets(self):

    def applyControl(self, action, dt):
        speed = action[0]
        turn_rate = action[1]
        self._state[0] = self._state[0] + speed * np.cos(self._state[2])*dt
        self._state[1] = self._state[1] + speed * np.sin(self._state[2])*dt
        self._state[2] = self._state[2] + turn_rate * dt

    def getState(self):
        return np.copy(self._state)









