from trackingLib.dataAssociation import JPDAFMerged
from trackingLib.params import Parameters
import numpy as np
from trackingLib.sensor import Measurement
from trackingLib.plotting import StatePlotter
import matplotlib.pyplot as plt

if __name__ == '__main__':
    working_directory = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib'

    p = Parameters(working_directory)
    robots = p.getRobots()
    robot = robots[0]
    sensor = p.getSensor()
    planner = p.getPlanner()
    JPDAF = p.getEstimator()
    target_model = p.getWorld()

    tracker = JPDAFMerged(sensor, p.unresolved_resolution, p.clutter_density, p.sequential_resolution_update_flag,
                          p.gate_level, simulated_time_flag=p.simulated_time_flag, log=p.tracking_log_json,
                          console_log=False)

    plotter = StatePlotter(p.map_min, p.map_max, title='', video=True, track_stats_flag=False, meas_plot_flag=True,
                           FOV_flag=True, working_directory=p.working_directory)

    pi = np.pi
    range = 5. # range to targets
    delta_angle_degrees = 5
    angle1_radians = pi/2 + delta_angle_degrees * pi/180
    angle2_radians = pi/2 - delta_angle_degrees * pi/180

    y0 = np.array([[range * np.cos(angle1_radians)], [range * np.sin(angle1_radians)], [1], [0]])
    y1 = np.array([[range * np.cos(angle2_radians)], [range * np.sin(angle2_radians)], [-1], [0]])

    # move targets such that they can be positioned the correct separation before
    y0[0, 0] = y0[0, 0] - 2
    y1[0, 0] = y1[0, 0] + 2

    y = [y0, y1]
    own_state = robots[0].getState()
    measurements = []

    robot.tmm.targets[0]._state = y0
    robot.tmm.targets[1]._state = y1

    target_model.targets[0]._state = y0
    target_model.targets[1]._state = y1

    #move forward onece before traking initial measurements
    target_model.forwardSimulate()
    for targ in target_model.targets:
        measurements.append(sensor.observationModel(own_state, targ.getState()))
    measurements = [Measurement(meas[0], 0, 1) for meas in measurements]
    #apply filter
    tracker.filter(measurements, robot, robot.getState())
    plotter.plot_state(robots, robots[0].getState(), target_model.getTargets(), measurements,
                       robot_size=1, target_size=10, fov=p.fov, max_range=p.max_range)
    plt.show()
    # move targets to desired position and get measurements for analysis
    target_model.forwardSimulate()
    measurements = []
    for targ in target_model.targets:
        measurements.append(sensor.observationModel(own_state, targ.getState()))
    #measurements = [Measurement(meas[0], 0, 1) for meas in measurements]  # create list of Measurement objects
    measurements = [Measurement((measurements[0] + measurements[1])/2, 0, 1)]  # create merged measurement

    # apply filter
    tracker.filter(measurements, robot, robot.getState())
    plotter.plot_state(robots, robots[0].getState(), target_model.getTargets(), measurements,
                       robot_size=1, target_size=10, fov=p.fov, max_range=p.max_range)
    plt.show()

    a = 0


