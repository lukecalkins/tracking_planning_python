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
import sys, os

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
plt.rcParams['axes.grid'] = True
from matplotlib import patches
import numpy as np
import numpy.linalg as LA
import pdb
import imageio
from matplotlib.lines import Line2D
from trackingLib.utils import *

class StatePlotter:

    def __init__(self, mapmin, mapmax, title, plotNum=1, video=False, track_stats_flag=False, meas_plot_flag=False,
                 FOV_flag=False, plan_plot_flag=False, working_directory=None):

        self.fig = plt.figure(plotNum, figsize=(16, 8))
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
        self.FOV_flag = FOV_flag
        self.plan_plot_flag = plan_plot_flag
        self.working_directory = working_directory
        self.steps_into_action = 0  # need to initialize at one because first action take before plotting
        plt.ion()

    def draw_env(self):

        if self.track_stats:
            self.ax = self.fig.add_subplot(121)
            self.ax_MSE = self.fig.add_subplot(222)
            self.ax_log_det_sig = self.fig.add_subplot(224)
        elif self.meas_plot_flag:
            if self.plan_plot_flag:
                self.ax = self.fig.add_subplot(221)
                self.ax_meas = self.fig.add_subplot(222)
                self.ax_plan = self.fig.add_subplot(223)
                self.ax_plan_meas = self.fig.add_subplot(224)
            else:
                self.ax = self.fig.add_subplot(121)
                self.ax_meas = self.fig.add_subplot(122)
        else:
            self.ax = self.fig.subplots()

        self.ax.set_xlim(self.mapmin[0], self.mapmax[0])
        self.ax.set_ylim(self.mapmin[1], self.mapmax[1])
        if self.plan_plot_flag:
            self.ax_plan.set_xlim(self.mapmin[0], self.mapmax[0])
            self.ax_plan.set_ylim(self.mapmin[1], self.mapmax[1])
        self.ax.set_xlabel('Distance (m)')

        #return self.ax

    def clear_plot(self):
        """
        Clear the figure between timesteps.
        :return: The resulting cleared figure.
        """
        return self.fig.clf()

    def draw_robot(self, pose, clr='b', size=1, ax=None):

        x = pose[0]
        y = pose[1]

        # point robotrs
        #self.ax.plot(x, y, clr + '.', markersize=size)

        th = pose[2]
        # Construct Triangle Polygon.
        XY = np.array([[x + size * np.cos(th), x + size * np.cos(th - 2.7), x + size * np.cos(th + 2.7)],
                       [y + size * np.sin(th), y + size * np.sin(th - 2.7), y + size * np.sin(th + 2.7)]]).transpose()

        if ax:
            return ax.add_patch(patches.Polygon(XY, facecolor=clr, zorder=2, alpha=0.5))
        else:
            return self.ax.add_patch(patches.Polygon(XY, facecolor=clr, zorder=2, alpha=0.5))

    def draw_fov(self, ax, pose, sense_range, fov, clr='c'):

        x = pose[0]
        y = pose[1]
        th = pose[2]
        # convert theta to degrees
        th_deg = 180/np.pi * th
        #create port wedge
        theta1 = th_deg + 90 - fov/2
        theta2 = th_deg + 90 + fov/2
        ax.add_patch(patches.Wedge((x, y), sense_range, theta1, theta2,
                                               fill=False,
                                               edgecolor='b',
                                               linewidth=1.5,
                                               linestyle='--',
                                               facecolor=clr,
                                               alpha=.4, zorder=3))
        #create starboard wedge
        theta1 = th_deg - 90 - fov / 2
        theta2 = th_deg - 90 + fov / 2
        ax.add_patch(patches.Wedge((x, y), sense_range, theta1, theta2,
                                        fill=False,
                                        edgecolor='b',
                                        linewidth=1.5,
                                        linestyle='--',
                                        facecolor=clr,
                                        alpha=.4, zorder=3))

    def draw_target(self, state, clr='r', size=1):

        x = state[0]
        y = state[1]

        self.ax.plot(x, y, clr, marker='o', markersize=size)

    def draw_cov(self, mean, cov, confidence, clr='r', ax=None):
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
        if ax:
            ax.plot(a[0, :] + mean[0], a[1, :] + mean[1], clr)
            ax.plot(mean[0], mean[1], clr, marker='x')
        else:
            self.ax.plot(a[0, :] + mean[0], a[1, :] + mean[1], clr)
            self.ax.plot(mean[0], mean[1], clr, marker='x')

    def draw_planned_path(self, pose, planner_output, plan_dt, action_ndx):
        pos_x = [pose[0]]
        pos_y = [pose[1]]
        #apply current action for necessary amoutn of steps
        for i in range(action_ndx, len(planner_output)):
            if i == action_ndx:
                steps = plan_dt - self.steps_into_action
                self.steps_into_action += 1
                if self.steps_into_action == plan_dt:
                    self.steps_into_action = 0
            else:
                steps = plan_dt
            pose = propagateOwnshipEuler(pose, planner_output[i][0], planner_output[i][1], steps)
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

    def draw_measurements(self, measurements, size=0.1, clr='b'):

        bearings = np.linspace(0, 2 * np.pi, num=100)
        points_x = np.cos(bearings)
        points_y = np.sin(bearings)
        self.ax_meas.plot(points_x, points_y)
        for meas in measurements:
            value = meas.getZ()
            self.ax_meas.plot(np.cos(value), np.sin(value), c='r', marker='x', markersize=10)

        # position robot at [0,0,0] for
        pose = np.array([0, 0, 0])
        x = pose[0]
        y = pose[1]
        th = pose[2]
        XY = np.array([[x + size * np.cos(th), x + size * np.cos(th - 2.7), x + size * np.cos(th + 2.7)],
                       [y + size * np.sin(th), y + size * np.sin(th - 2.7), y + size * np.sin(th + 2.7)]]).transpose()
        return self.ax_meas.add_patch(patches.Polygon(XY, facecolor=clr, zorder=2, alpha=0.5))

    def draw_planner_belief(self, plan_node, max_range=None, fov=None, clr_list=None, size=1):

        pose = plan_node.state.state
        self.draw_robot(pose, size=size, ax=self.ax_plan)
        self.draw_fov(self.ax_plan, pose, max_range, fov)
        y_dim = plan_node.state.y_dim
        num_targs = plan_node.state.targ_state[0].shape[0] // y_dim
        for i in range(num_targs):
            targ_state = plan_node.state.targ_state_at_node
            start = i * y_dim
            stop = start + y_dim
            mean = targ_state[start: start + 2]
            cov = plan_node.state.Sigma[start:stop, start:stop]
            cov = cov[:2, :2]
            self.draw_cov(mean, cov, confidence=0.99, clr=clr_list[i], ax=self.ax_plan)

        bearings = np.linspace(0, 2 * np.pi, num=100)
        points_x = np.cos(bearings)
        points_y = np.sin(bearings)
        self.ax_plan_meas.plot(points_x, points_y)
        measurements = plan_node.state.predicted_meas
        for meas in measurements:
            value = meas
            self.ax_plan_meas.plot(np.cos(value), np.sin(value), c='r', marker='x', markersize=10)




    def plot_state(self, robots, own_state, targets=None, measurements=None, planner_output=None, num_targs_seen=None, masked=False,
                   robot_size=1, target_size=1, timestep=None, fov=None, max_range=None, plan_node=None, plan_dt=None,
                   action_ndx=None):

        self.clear_plot()
        self.draw_env()
        self.ax.set_title("Number of targets visible: " + str(num_targs_seen) + "   Timestep: " + str(timestep))
        if masked == True:
            self.ax.set_title("MASKED!")
        clr_list = ['r', 'b', 'g', 'm', 'y']

        for robot in robots:
            pose = own_state
            self.draw_robot(pose, size=robot_size)
            if self.FOV_flag:
                self.draw_fov(self.ax, pose, max_range, fov)
            if planner_output != None:
                self.draw_planned_path(pose, planner_output, plan_dt,  action_ndx)
            if self.plan_plot_flag:
                self.draw_planner_belief(plan_node, max_range, fov, clr_list, size=robot_size)

            info_ndx = 0
            for target in robot.tmm.targets:
                mean = target.getState()[:2]
                cov = target.getCovariance()[:2, :2]
                self.draw_cov(mean, cov, confidence=0.99, clr=clr_list[info_ndx])
                info_ndx += 1

            if self.meas_plot_flag:
                self.draw_measurements(measurements)

        target_ndx = 0
        if targets:
            for target in targets:
                target_state = target.getState()
                self.draw_target(target_state, clr=clr_list[target_ndx], size=target_size)
                target_ndx += 1

        if self.track_stats:
            self.draw_mse_lds_curves(robots, targets)

        plt.draw()
        #plt.tight_layout()
        if self.video:
            plt.tight_layout()
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
        file_dir = os.path.join(self.working_directory, 'results/videos/')
        full_name = os.path.join(file_dir, filename)
        imageio.mimsave(full_name + '.mp4', self.images, fps=fps)

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