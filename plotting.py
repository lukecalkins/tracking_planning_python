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

class StatePlotter:

    def __init__(self, mapmin, mapmax, title, plotNum=1, video=False):

        self.fig = plt.figure(plotNum)
        self.mapmin = mapmin
        self.mapmax = mapmax
        self.title = title
        self.video = video
        self.images = []
        plt.ion()

    def draw_env(self):

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
        return self.ax.add_patch(patches.Polygon(XY, facecolor=clr, zorder=5))

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


    def plot_state(self, robots, targets, robot_size=1, target_size=1):

        self.clear_plot()
        self.draw_env()
        clr_list = ['r', 'b']

        for robot in robots:
            pose = robot.getState()
            self.draw_robot(pose, size=robot_size)

            info_ndx = 0
            for ID in robot.tmm.targets:
                target = robot.tmm.getTargetByID(ID)
                mean = target.getState()[:2]
                cov = target.getCovariance()[:2, :2]
                self.draw_cov(mean, cov, confidence=0.99, clr=clr_list[info_ndx])
                info_ndx += 1

        target_ndx = 0
        for target in targets:
            target_state = target.getState()
            self.draw_target(target_state, clr=clr_list[target_ndx], size=target_size)
            target_ndx += 1

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
        imageio.mimsave('../results/videos/' + filename + '.gif', self.images, fps=fps)