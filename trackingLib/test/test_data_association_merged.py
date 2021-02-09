from trackingLib.utils import restrict_angle, get_custom_trajectory
import numpy as np
import trackingLib.dataAssociation as DA
import trackingLib.dataAssociation_ambiguity as DA_amb
from trackingLib.params import Parameters
import matplotlib.pyplot as plt
from trackingLib.plotting import *
from trackingLib.sensor import add_clutter
from trackingLib.sim_data import DataSaver

pi = np.pi

b0 = 11 * np.pi/12
b1 = -11 * np.pi/12

print(restrict_angle(b1-b0), " radians, ", restrict_angle(b1-b0)*180/np.pi, " degrees")
print(restrict_angle(b0-b1), " radians, ", restrict_angle(b0-b1)*180/np.pi, " degrees")

# point to file where config folder is locatedß
working_directory = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib'

p = Parameters(working_directory)

if False:
    angles = np.linspace(-30, 30)
    angles_radians = angles * pi/180
    bearing_res = p.unresolved_resolution
    R_u = bearing_res ** 2
    P_u = np.exp(-1./2 * angles_radians * 1./R_u * angles_radians)
    fig, ax = plt.subplots()
    ax.plot(angles, P_u)
    ax.set_xlabel('$\Delta$ bearing')
    ax.set_ylabel('Probability unresolved')
    ax.set_title('$alpha_{bearing} = $' + str(bearing_res))
    plt.show()



map_min = p.map_min
map_max = p.map_max
title = "Kalman Filter test"
plotter = StatePlotter(map_min, map_max, title, video=True, track_stats_flag=False, meas_plot_flag=False, FOV_flag=False,
                       working_directory=p.working_directory)
data_saver = DataSaver(p.num_targs)

robots = p.getRobots()
sensor = p.getSensor()
planner = p.getPlanner()
JPDAF = p.getEstimator()
target_model = p.getWorld()

JPDAF_merged = DA.JPDAFMerged(sensor, p.unresolved_resolution, p.clutter_density, p.sequential_resolution_update_flag,
                              p.gate_level, simulated_time_flag=p.simulated_time_flag, log=p.tracking_log_json)

JPDAF_ambiguity = DA_amb.JPDAF_amb(p.detection_prob, p.clutter_density, p.gate_level)

# Add a custom target trajectory
traj = get_custom_trajectory(target_model.getTargets()[0].getState(), p.Tmax + 1, dt=1)
target_model.targets[0].add_fixed_trajectory(traj)

np.random.seed(p.random_seed)

for kk in range(p.Tmax):

    target_model.forwardSimulate()
    #target_model.forwardSimulate_fixed_trajectory(kk + 1)

    for i in range(len(robots)):
        #measurements, num_targets_seen = sensor.senseTargets(robots[i].getState(), target_model.getTargets())
        measurements, num_targets_seen = sensor.senseTargets_interference_n(robots[i].getState(), target_model.getTargets(), p.masking_proximity)
        #measurements, num_targets_seen = sensor.senseTargets_resolution_model_2(robots[i].getState(), target_model.getTargets(), p.unresolved_resolution)
        #measurements, num_targets_seen = sensor.senseTargets_resolution_model_n(robots[i].getState(), target_model.getTargets(), p.unresolved_resolution)
        for meas in measurements:
            print(meas.getZ())
        #measurements, num_targets_seen = sensor.senseTargets_ambiguity(robots[i].getState(), target_model.getTargets())
        #measurements, num_targets_seen = sensor.senseTargets_FOV(robots[i].getState(), target_model.getTargets())
        #add_clutter(measurements, p.clutter_density)

        JPDAF.filter(measurements, robots[i], robots[i].getState())

        #JPDAF_ambiguity.filter(measurements, robots[i])

        #filter_output = JPDAF_merged.filter(measurements, robots[i], robots[i].getState())

        output = robots[0].tmm.get_system_belief_copy()
        #print("mean:  ", output._mean)
        #print("covariance:  ", output._cov)

    plotter.plot_state(robots, robots[0].getState(), target_model.getTargets(), measurements, num_targs_seen=num_targets_seen,
                       robot_size=50, target_size=10, timestep=kk, fov=p.fov, max_range=p.max_range)

    #plt.pause(0.1)

    #data_saver.write_time_instance(target_model, measurements, robots[0])  # todo: make it for multiple robots

    print("Timstep: ", kk)


dir = 'JPDAF/auma_presentation/'
file_name = 'interference'
file_name = file_name + '_seed_' + str(p.random_seed)
plotter.save_video(filename=dir + file_name, fps=5)

#json_filename = '/log/' + file_name
#JPDAF_merged.write_log_file_json(p.working_directory, json_filename)

#data_saver.write_data_to_file(file_name)
#p.write_params_to_file(dir)


