from robot import *
import numpy as np
from plotting import *
from target import *
from sensor import *
import pdb
import kalmanFilter as KF
import dataAssociation as DA
import dataAssociationPlan as DAP
import planner as plan
from cost_function import *


if __name__ == '__main__':

    mapmin = [-1000, -1000]
    mapmax = [1000, 1000]
    title = "Kalman Filter test"
    plotter = StatePlotter(mapmin, mapmax, title, video=True)

    # Timing parameters
    T = 200
    samp = 1.0

    # Target parameters
    target_initial_state = [0, 0, 1, 1]
    cov_pos = 10
    cov_vel = 0.5
    #target_ID = [100]

    #target 2
    target_initial_state2 = [-750, -250, 5, 5]

    target_ID = [100, 200]


    # Robot parameters
    initial_pose = np.array([500, 0, np.pi/2])
    target_initial_belief = [0, 0, 1, 1]
    target_init_cov_pos = 1000
    target_init_cov_vel = 0.2
    info_target = InfoTarget(target_initial_belief, cov_pos, cov_vel, samp, target_ID[0], 4,
                             target_init_cov_pos, target_init_cov_vel)

    target_initial_belief = [-750, -250, 5, 5]
    info_target2 = InfoTarget(target_initial_belief, cov_pos, cov_vel, samp, target_ID[1], 4,
                             target_init_cov_pos, target_init_cov_vel)

    #Sensor parameters
    detection_prob = 1.0
    sense_min_range = 0
    sense_max_range = 1000
    sense_min_hang = -180
    sense_max_hang = 180
    sense_b_sigma = 0.021 # Wolek2019 set to 0.021
    clutter_density = 0.01 * 3.183  # 3.183 = 20 per 360 degrees ~ 1 every 18 degrees
    proximity = 10  # proximity for target masking in degrees

    sensor = BearingSensor(sense_min_range, sense_max_range, sense_min_hang, sense_max_hang, sense_b_sigma, detection_prob)

    info_target_model = InfoTargetModel()
    info_target_model.addTarget(target_ID[0], info_target)
    info_target_model.addTarget(target_ID[1], info_target2)

    robot = Robot(initial_pose, sensor, info_target_model, samp)
    robots = [robot]

    target1 = Target(target_initial_state, cov_pos, cov_vel, samp, target_ID[0], 4)
    target2 = Target(target_initial_state2, cov_pos, cov_vel, samp, target_ID[1], 4)
    targets = [target1, target2]

    #actions
    turn_radius = 50.
    speed = 10.
    actions = [[speed, 0], [speed, speed/turn_radius], [speed, -speed/turn_radius]]

    JPDA = DA.JPDAF(detection_prob=detection_prob, gate_level=0.99, verbose=False)
    log_det_cost = LogDetCost(y_dim=4)
    gate_cost = GateOverlapCost(y_dim=4, level=0.99)
    delta_bearing_cost = DeltaBearingCost(y_dim=4)
    planner_log = '../log/plan.log'
    targ_log = '../log/targ.log'

    JPDAF_simulator = DAP.JPDAF_simulate(sensor, gate_level=0.99, verbose=True)

    planner = plan.Planner(actions, log_det_cost, JPDAF_simulator, final_cost=True)
    plan_horizon = 10
    n_controls = 5  # number of steps to take before replanning

    np.random.seed(42)
    for kk in range(T):
        #print('True     : ', targets[0].getState())
        #print('Estimated: ', robot.tmm.targets[100].getState())

        for i in range(len(robots)):
            measurements = robots[i].sensor.senseTargets(robots[i].getState(), targets)
            #measurements, masked = robots[i].sensor.senseTargets_interference_2(robots[i].getState(), targets, proximity)
            #DA.add_masked_measurements_2targ(measurements, robot, target_ID, proximity)
            add_clutter(measurements, clutter_density)
            JPDA.filter(measurements, robot, target_ID, clutter_density)
            #output = KF.MultiTargetFilter(measurements, robots[i], debug=False)
            #robots[i].tmm.updateBelief(output)



        if kk % n_controls == 0:
            for robot in robots:
                planner_output = planner.planFVI(robot, plan_horizon)

        print("planner output", planner_output)
        for robot in robots:
            print("plan_output length = ", len(planner_output))
            robot.applyControl(planner_output.pop(0), 1)

        for target in targets:
            target.forwardSimulate(1)

        plotter.plot_state(robots, targets, planner_output,
                           masked=masked, robot_size=50, target_size=10)
        plt.pause(0.1)

        print("Timestep: ", kk)

    plotter.save_video(filename='planning/2_targ_FVI_log_det_masked_10_final_cost_speed_10', fps=5)
