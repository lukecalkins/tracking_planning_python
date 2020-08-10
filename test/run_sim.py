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
from params import Parameters


if __name__ == '__main__':

    yaml_file = 'config/init_info_planner.yaml'
    p = Parameters(yaml_file)

    map_min = p.map_min
    map_max = p.map_max
    title = "Kalman Filter test"
    plotter = StatePlotter(map_min, map_max, title, video=True, track_stats_flag=True)

    planner_output = []  # only utilized when not running planner

    robots = p.getRobots()
    planner = p.getPlanner()
    target_model = p.getWorld()
    JPDAF = p.getEstimator()

    np.random.seed(p.random_seed)

    # Main Loop
    for kk in range(p.Tmax):
        #print('True     : ', targets[0].getState())
        #print('Estimated: ', robot.tmm.targets[100].getState())

        for i in range(len(robots)):
            measurements = robots[i].sensor.senseTargets(robots[i].getState(), target_model.getTargets())
            #measurements, num_targs_seen = robots[i].sensor.senseTargets_interference_n(robots[i].getState(), targets, proximity)
            #print("num targs_seen: ", num_targs_seen)
            #measurements, masked = robots[i].sensor.senseTargets_interference_2(robots[i].getState(), targets, proximity)
            #DA.add_masked_measurements_2targ(measurements, robot, target_ID, proximity)
            add_clutter(measurements, p.clutter_density)
            JPDAF.filter(measurements, robots[i])
            #output = KF.MultiTargetFilter(measurements, robots[i], debug=False)
            #robots[i].tmm.updateBelief(output)

        if kk % p.n_controls == 0:
            for robot in robots:
                planner_output = planner.planFVI(robot, p.horizon)
                #pass

        print("planner output", planner_output)
        for robot in robots:
            if len(planner_output) == 0:
                robot.applyControl([0,  0], 1)
            else:
                print("plan_output length = ", len(planner_output))
                robot.applyControl(planner_output.pop(0), 1)

        target_model.forwardSimulate()

        plotter.plot_state(robots, target_model.getTargets(), planner_output,
                            robot_size=50, target_size=10)

        #track_stats_plotter.plot_stats(robot.tmm.getCovarianceMatrix(), robot.tmm.getTargetState(), targets)
        plt.pause(0.1)

        print("Timestep: ", kk)

    filename = 'planning/JPDAF/4_targ/test_new'
    #filename = 'planning/JPDAF/4_targ/4_targ_no_plan_masked'
    plotter.save_video(filename=filename, fps=5)
    plotter.save_track_stats(filename=filename)
    #track_stats_plotter.save_video(filename='planning/JPDAF/test/2_targ_FVI_log_det_kalman_speed_5_2_stats', fps=5)