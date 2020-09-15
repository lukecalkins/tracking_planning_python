import trackingLib.dataAssociation as DA
from trackingLib.params import Parameters
from trackingLib.plotting import *
import numpy as np
from trackingLib.sensor import Measurement
import json

if __name__ == '__main__':
    pi = np.pi

    working_directory = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib'
    p = Parameters(working_directory)

    plotter = StatePlotter(p.map_min, p.map_max, title='', video=True, track_stats_flag=False, meas_plot_flag=True,
                           FOV_flag=True, working_directory=p.working_directory)

    robots = p.getRobots()
    sensor = p.getSensor()

    #exp_data_file = '/Users/william.calkins/Documents/Research/Tracking/python/data/91120BallsResults/static_tests/balls1/balls.json'
    #exp_data_file = '/Users/william.calkins/Documents/Research/Tracking/python/data/91120BallsResults/static_tests/balls01/balls.json'
    #exp_data_file = '/Users/william.calkins/Documents/Research/Tracking/python/data/91120BallsResults/static_tests/balls001/balls.json'
    exp_data_file = '/Users/william.calkins/Documents/Research/Tracking/python/data/91120BallsResults/moving_tests/movingballs1/balls.json'
    #exp_data_file = '/Users/william.calkins/Documents/Research/Tracking/python/data/91120BallsResults/moving_tests/movingballs01/balls.json'



    tracker = DA.JPDAFMerged(sensor, p.unresolved_resolution, p.clutter_density, p.sequential_resolution_update_flag,
                              p.gate_level, simulated_time_flag=p.simulated_time_flag)

    with open(exp_data_file) as f:
        data = json.load(f)

    for kk in range(len(data)):
        time_step_data = data[str(kk)]
        contacts = time_step_data['contacts']
        measurements = [Measurement(contact, 0, 1) for contact in contacts]

        tracker.filter(measurements, robots[0], robots[0].getState())

        plotter.plot_state(robots, robots[0].getState(), measurements=measurements,
                       robot_size=1, target_size=10, timestep=kk, fov=p.fov, max_range=p.max_range)


    dir = 'experiments/091120/'
    filename = dir + 'raw_q0.1'
    plotter.save_video(filename, fps=5 )