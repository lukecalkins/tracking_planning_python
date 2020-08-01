from utils import restrict_angle
import numpy as np
import dataAssociation as DA
from params import Parameters
import matplotlib.pyplot as plt
from plotting import *
from sensor import add_clutter

pi = np.pi

b0 = 11 * np.pi/12
b1 = -11 * np.pi/12

print(restrict_angle(b1-b0), " radians, ", restrict_angle(b1-b0)*180/np.pi, " degrees")
print(restrict_angle(b0-b1), " radians, ", restrict_angle(b0-b1)*180/np.pi, " degrees")

yaml_file = 'config/init_info_planner.yaml'
p = Parameters(yaml_file)

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
plotter = StatePlotter(map_min, map_max, title, video=True, track_stats_flag=False)


robots = p.getRobots()
planner = p.getPlanner()
JPDAF = p.getEstimator()
target_model = p.getWorld()

gate_level = 0.99
JPDAF_merged = DA.JPDAFMerged(robots[0].sensor, p.unresolved_resolution, p.clutter_density, gate_level)

np.random.seed(p.random_seed)

for kk in range(p.Tmax):

    for i in range(len(robots)):
        measurements, num_targets_seen = robots[0].sensor.senseTargets(robots[0].getState(), target_model.getTargets())
        #measurements, num_targets_seen = robots[i].sensor.senseTargets_interference_n(robots[i].getState(), target_model.getTargets(), p.masking_proximity)
        #measurements, num_targets_seen = robots[i].sensor.senseTargets_resolution_model_2(robots[i].getState(), target_model.getTargets(), p.unresolved_resolution)
        #measurements, num_targets_seen = robots[i].sensor.senseTargets_resolution_model_n(robots[i].getState(), target_model.getTargets(), p.unresolved_resolution)
        add_clutter(measurements, p.clutter_density)

        JPDAF.filter(measurements, robots[i])

        #filter_output = JPDAF_merged.filter(measurements, robots[i])
        #robots[i].tmm.updateBelief(filter_output)

    target_model.forwardSimulate()
    plotter.plot_state(robots, target_model.getTargets(), num_targs_seen=num_targets_seen,
                       robot_size=50, target_size=10, timestep=kk)
    plt.pause(0.1)

    print("Timstep: ", kk)

file_name = 'JPDAF/merged/3_targ/3_targ_fully_resolved_JPDAF_wrapped_gate'
file_name = file_name + '_seed_' + str(p.random_seed)
plotter.save_video(filename=file_name, fps=5)


