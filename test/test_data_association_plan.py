from dataAssociationPlan import JPDAF_simulate
from sensor import BearingSensor, Measurement
import numpy as np
from kalmanFilter import GaussianBelief

b_sigma = 0.021
Pd = 1.0  # detection probablility
sensor = BearingSensor(0, 5000, -180, 180, b_sigma, Pd)

filter = JPDAF_simulate(sensor, gate_level=0.99, verbose=True)

#create measurements
b1 = 75 * np.pi/180
r1 = 500
b2 = 70 * np.pi/180
r2 = 500
b3 = 81 * np.pi/180
r3 = 500
m1 = Measurement(b1, 0, 1)
m2 = Measurement(b2, 0, 1)
m3 = Measurement(b3, 0, 1)
meas = [m1, m2, m3]

#predicted target beliefs
mean1 = np.array([[r1 * np.cos(b1)], [r1 * np.sin(b1)], [1], [1]])
mean2 = np.array([[r2 * np.cos(b2)], [r2 * np.sin(b2)], [1], [1]])
mean3 = np.array([[r3 * np.cos(b3)], [r3 * np.sin(b3)], [1], [1]])
cov1 = np.eye(4)
cov2 = np.eye(4)
cov3 = np.eye(4)
cov1[0:2, 0:2] = 1000*np.eye(2)
cov2[0:2, 0:2] = 1000*np.eye(2)
cov3[0:2, 0:2] = 1000*np.eye(2)
t1 = GaussianBelief(mean1, cov1)
t2 = GaussianBelief(mean2, cov2)
t3 = GaussianBelief(mean3, cov3)
targ_beliefs = [t1, t2, t3]

ownship = [0,0,0]

filter.filter(meas, targ_beliefs, ownship)


