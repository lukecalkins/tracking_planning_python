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
import dataAssociation_ambiguity as DA_amb


if __name__ == '__main__':

    yaml_file = 'config/init_info_planner.yaml'
    p = Parameters(yaml_file)

    map_min = p.map_min
    map_max = p.map_max
    title = "Kalman Filter test"
    plotter = StatePlotter(map_min, map_max, title, video=True, track_stats_flag=False, FOV_flag=True,
                           meas_plot_flag=True, plan_plot_flag=True)

    planner_output = []  # only utilized when not running planner

    robots = p.getRobots()
    sensor = p.getSensor()
    planner = p.getPlanner()
    target_model = p.getWorld()
    JPDAF = p.getEstimator()

    JPDAF_merged = DA.JPDAFMerged(sensor, p.unresolved_resolution, p.clutter_density,
                                  p.sequential_resolution_update_flag,
                                  p.gate_level)
    JPDAF_ambiguity = DA_amb.JPDAF_amb(p.detection_prob, p.clutter_density, p.gate_level)

    np.random.seed(p.random_seed)

    # Main Loop
    for kk in range(p.Tmax):
        #print('True     : ', targets[0].getState())
        #print('Estimated: ', robot.tmm.targets[100].getState())

        if kk % p.n_controls == 0:
            for robot in robots:
                planner_output, optimal_node = planner.planFVI(robot)
                steps_into_plan = 0

        print("planner output", planner_output)

        for i in range(len(robots)):
            #measurements, num_target_seen = sensor.senseTargets(robots[i].getState(), target_model.getTargets())
            #measurements, num_targs_seen = sensor.senseTargets_interference_n(robots[i].getState(), targets, proximity)            #print("num targs_seen: ", num_targs_seen)
            #measurements, num_targets_seen = sensor.senseTargets_resolution_model_n(robots[i].getState(), target_model.getTargets(), p.unresolved_resolution)
            # measurements, num_targets_seen = sensor.senseTargets_ambiguity(robots[i].getState(), target_model.getTargets())
            #measurements, num_targets_seen = sensor.senseTargets_FOV(robots[i].getState(), target_model.getTargets())
            measurements, num_targets_seen = sensor.senseTargets_resolution_model_n_FOV(robots[i].getState(), target_model.getTargets(), p.unresolved_resolution)
            #add_clutter(measurements, p.clutter_density)

            #output = KF.MultiTargetFilter(measurements, robots[i], debug=False)
            #robots[i].tmm.updateBelief(output)

            #JPDAF.filter(measurements, robots[i])

            filter_output = JPDAF_merged.filter(measurements, robots[i])
            robots[i].tmm.updateBelief(filter_output)


        for robot in robots:
            if len(planner_output) == 0:
                robot.applyControl([0,  0], 1)
            else:
                print("plan_output length = ", len(planner_output))
                robot.applyControl(planner_output.pop(0), 1)
                #robot.applyControl([20, 0], 1)
                steps_into_plan += 1
                curr_node = optimal_node
                for i in range(p.horizon - steps_into_plan):
                    curr_node = curr_node.parent



        target_model.forwardSimulate()

        plotter.plot_state(robots, target_model.getTargets(), measurements, num_targs_seen=num_targets_seen, timestep=kk,
                           planner_output=planner_output, robot_size=50, target_size=10, fov=p.fov,
                           max_range=p.max_range, plan_node=curr_node)

        #track_stats_plotter.plot_stats(robot.tmm.getCovarianceMatrix(), robot.tmm.getTargetState(), targets)
        plt.pause(0.1)

        print("Timestep: ", kk)

    filename = 'planning/merged/2_targ/JPDAF_merged_FOV_total_cost_plan_first'
    #filename = 'planning/JPDAF/4_targ/4_targ_no_plan_masked'
    plotter.save_video(filename=filename, fps=5)
    #plotter.save_track_stats(filename=filename)
    #track_stats_plotter.save_video(filename='planning/JPDAF/test/2_targ_FVI_log_det_kalman_speed_5_2_stats', fps=5)