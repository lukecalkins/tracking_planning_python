import numpy as np

class DataSaver:

    def __init__(self, num_targs):

        self.target_trajectories = []
        self.measurements = []
        self.targs_mean = []
        self.targs_cov = []
        self.num_targs = num_targs

    def write_time_instance(self, target_model, measurements, robot):

        self.target_trajectories.append(target_model.getTargetState())

        meas_list = []
        for meas in measurements:
            meas_list.append(meas)
        self.measurements.append(meas_list)

        self.targs_mean.append(robot.tmm.getTargetState())
        self.targs_cov.append(robot.tmm.getCovarianceMatrix())

    def write_data_to_file(self, filename):

        np.savez('results/videos/' + filename, targs=self.target_trajectories, meas=self.measurements,
                 beliefs_mean=self.targs_mean, beliefs_cov=self.targs_cov, num_targs=self.num_targs)




