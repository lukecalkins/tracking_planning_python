from trackingLib.plotting import *
import trackingLib.dataAssociation as DA
from trackingLib.cost_function import *
from trackingLib.params import Parameters
import trackingLib.dataAssociation_ambiguity as DA_amb
import sys
import json
import copy


if __name__ == '__main__':

    # point to file where config folder is located
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
                                  p.gate_level, FOV=p.fov, simulated_time_flag=p.simulated_time_flag)
    JPDAF_ambiguity = DA_amb.JPDAF_amb(p.detection_prob, p.clutter_density, p.gate_level)

    nearest_neighbor = DA.NearestNeighborFilter(sensor, simulated_time_flag=p.simulated_time_flag)

    np.random.seed(p.random_seed)
    #seed = int(sys.argv[1])
    json_data = {}

    # Main Loop
    for kk in range(p.Tmax):

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

            #JPDAF.filter(measurements, robots[i], robots[i].getState())

            filter_output = JPDAF_merged.filter(measurements, robots[i], robots[i].getState())

            #nearest_neighbor.filter(measurements, robots[i], robots[i].getState())

        if kk % (p.n_controls * p.plan_dt) == 0:
            for robot in robots:
                planner_output, optimal_node = planner.planFVI(robot.tmm.get_system_belief_copy(), robot.getState(),
                                                               JPDAF_merged.tracking_iterations)
                steps_into_plan = 0
                curr_node = optimal_node
                action_ndx = 0
                time_on_action = 0

        print("planner output", planner_output)

        time_step_data = {}
        time_step_data['contacts'] = [measurement.getZ() for measurement in measurements]
        beliefs = robots[i].tmm.get_system_belief_copy()
        time_step_data['ground_truth'] = target_model.getTargetState().tolist()
        time_step_data['post_mean'] = beliefs._mean.tolist()
        time_step_data['post_covariance'] = beliefs._cov.tolist()
        time_step_data['planner_output'] = copy.deepcopy(planner_output)
        if isinstance(robots[i].getState(), list):
            time_step_data['own_state'] = robots[i].getState()
        else:
            time_step_data['own_state'] = robots[i].getState().tolist()
        json_data[kk] = time_step_data

        plotter.plot_state(robots, robots[0].getState(), target_model.getTargets(), measurements,
                           num_targs_seen=num_targets_seen, timestep=kk,
                           planner_output=planner_output, robot_size=1, target_size=10, fov=p.fov,
                           max_range=p.max_range, plan_node=curr_node, plan_dt=p.plan_dt, action_ndx=action_ndx)

        for robot in robots:
            if len(planner_output) == 0:
                robot.applyControl([0,  0], 1)
                print('Applying no control')
            else:
                print("plan_output length = ", len(planner_output))

                if kk % p.plan_dt == 0:
                    #action = planner_output.pop(0)
                    action = planner_output[action_ndx]
                    steps_into_plan += 1
                    curr_node = optimal_node
                    for i in range(p.horizon - steps_into_plan):
                        curr_node = curr_node.parent
                robot.applyControl(action, 1)
                time_on_action += 1
                if time_on_action == p.plan_dt:
                    action_ndx += 1
                    time_on_action = 0



        #track_stats_plotter.plot_stats(robot.tmm.getCovarianceMatrix(), robot.tmm.getTargetState(), targets)
        #plt.pause(0.1)

        print("Timestep: ", kk)



    filename = 'planning/merged/1_targ/JPDAF_merged_plan_dt'
    #filename = 'planning/merged/101020_sims/4_targ_exp_NN_final_cost_log_det_10_steps_turn_radius_0.6_100T'
    #filename = filename + '_seed_' + str(p.random_seed)
    plotter.save_video(filename=filename, fps=5)

    #plotter.save_track_stats(filename=filename)
    #track_stats_plotter.save_video(filename='planning/JPDAF/test/2_targ_FVI_log_det_kalman_speed_5_2_stats', fps=5)
    #planner.write_log_file_json(working_directory + '/log/', 'plan_log')


    # Write json log data
    """
    location = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib/results/'
    direc = 'sims_icra2021/4targ/NN/'
    filename = location + direc + 'seed_' + str(seed)
    with open(filename + '.json', 'w') as f:
        json.dump(json_data, f, indent=4)

    plotter.save_video(filename=filename, fps=5)
    """