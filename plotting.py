"""
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
import numpy as np
import imageio
"""
# 2D Plotting Tools

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.rcParams['axes.grid'] = True
from matplotlib import patches
import numpy as np
import numpy.linalg as LA
import pdb
import imageio
from matplotlib.lines import Line2D
from utils import *

class StatePlotter:

    def __init__(self, mapmin, mapmax, title, plotNum=1, video=False, track_stats_flag=False, meas_plot_flag=False):

        self.fig = plt.figure(plotNum, figsize=(7,5))
        self.mapmin = mapmin
        self.mapmax = mapmax
        self.title = title
        self.video = video
        self.images = []  # for video only
        self.track_stats = track_stats_flag
        if self.track_stats:
            self.MSE_list = []
            self.log_det_Sigma_list = []
        self.meas_plot_flag = meas_plot_flag
        plt.ion()

    def draw_env(self):

        if self.track_stats:
            self.ax = self.fig.add_subplot(121)
            self.ax_MSE = self.fig.add_subplot(222)
            self.ax_log_det_sig = self.fig.add_subplot(224)
        elif self.meas_plot_flag:
            self.ax = self.fig.add_subplot(121)
            self.ax_meas = self.fig.add_subplot(122)
        else:
            self.ax = self.fig.subplots()

        self.ax.set_xlim(self.mapmin[0], self.mapmax[0])
        self.ax.set_ylim(self.mapmin[1], self.mapmax[1])
        self.ax.set_xlabel('Distance (m)')

        #return self.ax

    def clear_plot(self):
        """
        Clear the figure between timesteps.
        :return: The resulting cleared figure.
        """
        return self.fig.clf()

    def draw_robot(self, pose, clr='b', size=1):

        x = pose[0]
        y = pose[1]

        # point robotrs
        #self.ax.plot(x, y, clr + '.', markersize=size)

        th = pose[2]
        # Construct Triangle Polygon.
        XY = np.array([[x + size * np.cos(th), x + size * np.cos(th - 2.7), x + size * np.cos(th + 2.7)],
                       [y + size * np.sin(th), y + size * np.sin(th - 2.7), y + size * np.sin(th + 2.7)]]).transpose()
        return self.ax.add_patch(patches.Polygon(XY, facecolor=clr, zorder=2, alpha=0.5))

    def draw_target(self, state, clr = 'r', size = 1):

        x = state[0]
        y = state[1]

        self.ax.plot(x, y, clr, marker='o', markersize=size)

    def draw_cov(self, mean, cov, confidence, clr='r'):
        #manual entries for plotter checking
        #cov = np.array([[225, 0], [0, 225]])
        #mean = np.array([50, 50])

        s = -2 * np.log(1 - confidence)
        w, V = np.linalg.eig(s * cov)
        t = np.linspace(0, 2*np.pi, 100)
        D = np.array([[np.sqrt(w[0]), 0], [0, np.sqrt(w[1])]])

        Q = np.matmul(V, D)
        a = np.matmul(Q, np.array([np.cos(t), np.sin(t)]))
        b = 1
        self.ax.plot(a[0, :] + mean[0], a[1, :] + mean[1], clr)

    def draw_planned_path(self, pose, planner_output):
        pos_x = [pose[0]]
        pos_y = [pose[1]]
        for i in range(len(planner_output)):
            pose = propagateOwnshipEuler(pose, planner_output[i][0], planner_output[i][1], 1)
            pos_x.append(pose[0])
            pos_y.append(pose[1])
        self.ax.plot(pos_x, pos_y, 'k-', zorder=3)

    def draw_mse_lds_curves(self, robots, targets):

        robot = robots[0]
        Sigma = robot.tmm.getCovarianceMatrix()
        mean = robot.tmm.getTargetState()

        log_det_Sigma = np.log(np.linalg.det(Sigma))
        self.log_det_Sigma_list.append(log_det_Sigma)

        true_state = np.array([])
        for target in targets:
            true_state = np.append(true_state, target.getState())
        MSE = np.linalg.norm(mean - true_state)
        self.MSE_list.append(MSE)

        self.ax_MSE.plot(self.MSE_list, c='b')
        self.ax_MSE.set_title('MSE')
        self.ax_log_det_sig.plot(self.log_det_Sigma_list, c='b')
        self.ax_log_det_sig.set_title('Log(det($\Sigma$))')
        plt.tight_layout()

    def draw_measurements(self, measurements):

        bearings = np.linspace(0, 2 * np.pi, num=100)
        points_x = np.cos(bearings)
        points_y = np.sin(bearings)
        self.ax_meas.plot(points_x, points_y)
        for meas in measurements:
            value = meas.getZ()
            self.ax_meas.plot(np.cos(value), np.sin(value), c='r', marker='x', markersize=10)

    def plot_state(self, robots, targets, measurements = None, planner_output = None, num_targs_seen = None, masked = False, robot_size=1, target_size=1, timestep=None):

        self.clear_plot()
        self.draw_env()
        self.ax.set_title("Number of targets visible: " + str(num_targs_seen) + "   Timestep: " + str(timestep))
        if masked == True:
            self.ax.set_title("MASKED!")
        clr_list = ['r', 'b', 'g', 'm', 'y']

        for robot in robots:
            pose = robot.getState()
            self.draw_robot(pose, size=robot_size)
            if planner_output != None:
                self.draw_planned_path(pose, planner_output)

            info_ndx = 0
            for target in robot.tmm.targets:
                mean = target.getState()[:2]
                cov = target.getCovariance()[:2, :2]
                self.draw_cov(mean, cov, confidence=0.99, clr=clr_list[info_ndx])
                info_ndx += 1

        target_ndx = 0
        for target in targets:
            target_state = target.getState()
            self.draw_target(target_state, clr=clr_list[target_ndx], size=target_size)
            target_ndx += 1

        if self.track_stats:
            self.draw_mse_lds_curves(robots, targets)

        if self.meas_plot_flag:
            self.draw_measurements(measurements)

        plt.draw()
        if self.video:
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.images.append(image)

    def save_video(self, filename, fps=30):
        """
        save GIF
        :param filename:
        :param fps:
        :return:
        """
        kwargs_write = {'fps': fps, 'quantizer': 'nq'}
        imageio.mimsave('results/videos/' + filename + '.mp4', self.images, fps=fps)

    def save_track_stats(self, filename):
        np.savez('results/videos/' + filename, MSE=self.MSE_list, log_det_Sigma=self.log_det_Sigma_list)


class TrackStatsPlotter:

    def __init__(self, plot_num=1, video=False):

        self.fig = plt.figure(plot_num)
        self.MSE_list = []
        self.log_det_Sigma_list = []
        self.video = video
        self.images = []  # for video only

    def clear_plot(self):

        return self.fig.clf()

    def draw_curves(self):

        self.axes = self.fig.subplots(2, 1)
        self.axes[0].plot(self.MSE_list, c='b')
        self.axes[0].set_title('MSE')
        self.axes[1].plot(self.log_det_Sigma_list, c='b')
        self.axes[1].set_title('Log(det($\Sigma$))')
        plt.tight_layout()

    def plot_stats(self, Sigma, mean, targets):

        self.clear_plot()
        log_det_Sigma = np.log(np.linalg.det(Sigma))
        self.log_det_Sigma_list.append(log_det_Sigma)

        #get ground truth target state
        true_state = np.array([])
        for target in targets:
            true_state = np.append(true_state, target.getState())
        MSE = np.linalg.norm(mean - true_state)
        self.MSE_list.append(MSE)

        self.draw_curves()

        if self.video:
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.images.append(image)

    def save_video(self, filename, fps=30):
        imageio.mimsave('results/videos/' + filename + '.mp4', self.images, fps=fps)



def draw_cov(mean, cov, confidence, ax, clr='r'):
        #manual entries for plotter checking
        #cov = np.array([[225, 0], [0, 225]])
        #mean = np.array([50, 50])

        s = -2 * np.log(1 - confidence)
        w, V = np.linalg.eig(s * cov)
        t = np.linspace(0, 2*np.pi, 100)
        D = np.array([[np.sqrt(w[0]), 0], [0, np.sqrt(w[1])]])

        Q = np.matmul(V, D)
        a = np.matmul(Q, np.array([np.cos(t), np.sin(t)]))
        b = 1
        ax.plot(a[0, :] + mean[0], a[1, :] + mean[1], clr)