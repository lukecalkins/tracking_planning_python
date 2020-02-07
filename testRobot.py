from robot import *
import numpy as np
from plotting import *
from target import *
from sensor import *
import kalmanFilter as KF


if __name__ == '__main__':

    mapmin = [0, 0]
    mapmax = [100, 100]
    title = "Kalman Filter test"
    plotter = StatePlotter(mapmin, mapmax, title)

    T = 81
    samp = 1.0

    # Target parameters
    target_initial_state = [10, 10, 0, 1]
    cov_pos = 5
    cov_vel = 0.1
    target_ID = [100]

    # Robot parameters
    initial_pose = np.array([50, 50, np.pi/2])
    target_initial_belief = [50, 50, 1, 1]
    target_init_cov_pos = 1000
    target_init_cov_vel = 0.2
    info_target = InfoTarget(target_initial_belief, cov_pos, cov_vel, samp, target_ID[0], 4,
                             target_init_cov_pos, target_init_cov_vel)

    #Sensor parameters
    sense_min_range = 0
    sense_max_range = 1000
    sense_min_hang = -180
    sense_max_hang = 180
    sense_b_sigma = 0.0 # Wolek2019 set to 0.021

    sensor = BearingSensor(sense_min_range, sense_max_range, sense_min_hang, sense_max_hang, sense_b_sigma)

    target_model = []
    info_target_model = InfoTargetModel()
    info_target_model.addTarget(target_ID[0], info_target)

    robot = Robot(initial_pose, sensor, info_target_model, samp)
    robots = [robot]

    target = Target(target_initial_state, cov_pos, cov_vel, samp, target_ID[0], 4)
    targets = [target]

    for kk in range(T):

        for i in range(0, len(robots)):
            measurements = robots[i].sensor.senseTargets(robots[i].getState(), targets)
            output = KF.MultiTargetFilter(measurements, robots[i], debug=False)
            robots[i].tmm.updateBelief(output)

            for meas in measurements:
                print(meas._z * 180/np.pi, meas._ID)

        for robot in robots:
            robot.applyControl([0, 0], 1)

        for target in targets:
            target.forwardSimulate(1)

        plotter.plot_state(robots, targets, robot_size=10, target_size=10)
        plt.pause(0.1)
