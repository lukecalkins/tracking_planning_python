from robot import *
import numpy as np
from plotting import *
from target import *
from sensor import *
import kalmanFilter as KF
import dataAssociation as DA

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



if __name__ == '__main__':

    mapmin = [-1000, -1000]
    mapmax = [1000, 1000]
    title = "Kalman Filter test"
    plotter = StatePlotter(mapmin, mapmax, title)

    # Timing parameters
    T = 200
    samp = 1.0

    # Target parameters
    target_initial_state = [-500, 500, 5, 0]
    cov_pos = 5
    cov_vel = 0.1
    target_ID = [100]

    #target 2


    # Robot parameters
    initial_pose = np.array([0, 0, 0])
    target_initial_belief = [-500, 500, 5, 0]
    target_init_cov_pos = 1000
    target_init_cov_vel = 0.2
    info_target = InfoTarget(target_initial_belief, cov_pos, cov_vel, samp, target_ID[0], 4,
                             target_init_cov_pos, target_init_cov_vel)

    #Sensor parameters
    sense_min_range = 0
    sense_max_range = 1000
    sense_min_hang = -180
    sense_max_hang = 180
    sense_b_sigma = 0.021 # Wolek2019 set to 0.021
    clutter_density = 2 * 3.183  # 3.183 = 20 per 360 derees ~ 1 every 18 degrees

    sensor = BearingSensor(sense_min_range, sense_max_range, sense_min_hang, sense_max_hang, sense_b_sigma)

    info_target_model = InfoTargetModel()
    info_target_model.addTarget(target_ID[0], info_target)

    robot = Robot(initial_pose, sensor, info_target_model, samp)
    robots = [robot]

    target = Target(target_initial_state, cov_pos, cov_vel, samp, target_ID[0], 4)
    targets = [target]

    JPDA = DA.JPDAF(detection_prob=1, gate_level=0.99)

    for kk in range(T):
        print('True     : ', targets[0].getState())
        print('Estimated: ', robot.tmm.targets[100].getState())

        for i in range(len(robots)):
            measurements = robots[i].sensor.senseTargets(robots[i].getState(), targets)
            add_clutter(measurements, clutter_density)
            JPDA.filter(measurements, robot, target_ID, clutter_density)
            #output = KF.MultiTargetFilter(measurements, robots[i], debug=False)
            #robots[i].tmm.updateBelief(output)

        for robot in robots:
            robot.applyControl([0, 0], 1)

        for target in targets:
            target.forwardSimulate(1)

        plotter.plot_state(robots, targets, robot_size=10, target_size=10)
        plt.pause(0.1)
