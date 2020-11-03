import matplotlib.pyplot as plt
import numpy as np

from trackingLib.LRDT import LRDT
from trackingLib.robot import Robot
from trackingLib.target import Target
from trackingLib.params import Parameters
import json

pi = np.pi
twopi = 2 * pi

def sense_bearing_contacts(own_state, target_model):
    # create bearing contacts

if __name__ == '__main__':
    working_directory = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib'
    p = Parameters(working_directory)

    #define grid space and velocities
    x_range = np.arange(60)
    y_range = np.arange(60)
    speeds = np.arange(2, 12, 2)
    num_headings = 24
    velocities = []
    for i in range(num_headings):
        heading = -pi + i / num_headings * twopi
        for speed in speeds:
            x_vel = speed * np.cos(heading)
            y_vel = speed * np.sin(heading)
            velocities.append((x_vel, y_vel))

    prior_prob_targ = 0.25
    lrdt = LRDT(x_range, y_range, velocities, prior_prob_targ)

    sheet_index = 0

    initial_own_state = np.array([0, 0, 0])
    ownship = Robot(initial_own_state, None)  # don't include target motion model
    target_model = p.getWorld()
    sensor = p.getSensor()


    T_max = 15
    for i in range(T_max):
        y = sense_bearing_contacts(ownship.getState(), target_model)


    print("Exiting Main")
