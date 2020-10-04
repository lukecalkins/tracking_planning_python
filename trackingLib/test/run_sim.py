from trackingLib.plotting import *
import trackingLib.dataAssociation as DA
from trackingLib.cost_function import *
from trackingLib.params import Parameters
import trackingLib.dataAssociation_ambiguity as DA_amb


if __name__ == '__main__':

    # point to file where config folder is locatedß
    working_directory = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib'

    p = Parameters(working_directory)

    map_min = p.map_min
    map_max = p.map_max
    title = "Kalman Filter test"
    plotter = StatePlotter(map_min, map_max, title, video=True, track_stats_flag=False, FOV_flag=True,
                           meas_plot_flag=True, plan_plot_flag=True, working_directory=working_directory)

    planner_output = []  # only utilized when not running planner

    robots = p.getRobots()
    sensor = p.getSensor()
    planner = p.getPlanner()
    target_model = p.getWorld()
    JPDAF = p.getEstimator()

    JPDAF_merged = DA.JPDAFMerged(sensor, p.unresolved_resolution, p.clutter_density,
                                  p.sequential_resolution_update_flag,
                                  p.gate_level, simulated_time_flag=p.simulated_time_flag)
    JPDAF_ambiguity = DA_amb.JPDAF_amb(p.detection_prob, p.clutter_density, p.gate_level)

    np.random.seed(p.random_seed)

    # Main Loop
    for kk in range(p.Tmax):

        # call plotter before targets robot move to get initial state


        target_model.forwardSimulate()

        for i in range(len(robots)):
            #measurements, num_target_seen = sensor.senseTargets(robots[i].getState(), target_model.getTargets())
            #measurements, num_targs_seen = sensor.senseTargets_interference_n(robots[i].getState(), targets, proximity)            #print("num targs_seen: ", num_targs_seen)
            #measurements, num_targets_seen = sensor.senseTargets_resolution_model_n(robots[i].getState(), target_model.getTargets(), p.unresolved_resolution)
            #measurements, num_targets_seen = sensor.senseTargets_ambiguity(robots[i].getState(), target_model.getTargets())
            #measurements, num_targets_seen = sensor.senseTargets_FOV(robots[i].getState(), target_model.getTargets())
            measurements, num_targets_seen = sensor.senseTargets_resolution_model_n_FOV(robots[i].getState(), target_model.getTargets(), p.unresolved_resolution)
            #add_clutter(measurements, p.clutter_density)

            #output = KF.MultiTargetFilter(measurements, robots[i], debug=False)
            #robots[i].tmm.updateBelief(output)

            #JPDAF.filter(measurements, robots[i])

            filter_output = JPDAF_merged.filter(measurements, robots[i], robots[i].getState())

        if kk % p.n_controls == 0:
            for robot in robots:
                planner_output, optimal_node = planner.planFVI(robot.tmm.get_system_belief_copy(), robot.getState(),
                                                               JPDAF_merged.tracking_iterations)
                steps_into_plan = 0

        print("planner output", planner_output)

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

        plotter.plot_state(robots, robots[0].getState(), target_model.getTargets(), measurements,
                           num_targs_seen=num_targets_seen, timestep=kk,
                           planner_output=planner_output, robot_size=1, target_size=10, fov=p.fov,
                           max_range=p.max_range, plan_node=curr_node)

        #track_stats_plotter.plot_stats(robot.tmm.getCovarianceMatrix(), robot.tmm.getTargetState(), targets)
        #plt.pause(0.1)

        print("Timestep: ", kk)

    #filename = 'planning/merged/2_targ/JPDAF_merged_FOV_final_cost_10_steps_config_2'
    filename = 'planning/merged/exp_3targ/3targ_test_moving_0.1_config2'
    filename = filename + '_seed_' + str(p.random_seed)
    plotter.save_video(filename=filename, fps=5)

    #plotter.save_track_stats(filename=filename)
    #track_stats_plotter.save_video(filename='planning/JPDAF/test/2_targ_FVI_log_det_kalman_speed_5_2_stats', fps=5)
    planner.write_log_file_json(working_directory + '/log/', 'plan_log')