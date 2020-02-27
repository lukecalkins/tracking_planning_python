import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
import numpy as np

class StatePlotter:

    def __init__(self, mapmin, mapmax, title, plotNum=1, video=False):

        self.fig = plt.figure(plotNum)
        self.mapmin = mapmin
        self.mapmax = mapmax
        self.title = title
        self.video = video
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

        self.ax.plot(x, y, clr + '.', markersize=size)

    def draw_target(self, state, clr = 'r', size = 1):

        x = state[0]
        y = state[1]

        self.ax.plot(x, y, clr, marker='o', markersize=size)

    def draw_cov(self, mean, cov, confidence):
        #manual entries for plotter checking
        #cov = np.array([[225, 0], [0, 225]])
        #mean = np.array([50, 50])

        s = -2 *np.log(1 - confidence)
        w, V = np.linalg.eig(s * cov)
        t = np.linspace(0, 2*np.pi, 100)
        D = np.array([[np.sqrt(w[0]), 0], [0, np.sqrt(w[1])]])

        Q = np.matmul(V, D)
        a = np.matmul(Q, np.array([np.cos(t), np.sin(t)]))
        b = 1
        self.ax.plot(a[0, :] + mean[0], a[1, :] + mean[1], 'r')


    def plot_state(self, robots, targets, robot_size=1, target_size=1):

        self.clear_plot()
        self.draw_env()

        for robot in robots:
            pose = robot.getState()
            self.draw_robot(pose, size=robot_size)

            for ID in robot.tmm.targets:
                target = robot.tmm.getTargetByID(ID)
                mean = target.getState()[:2]
                cov = target.getCovariance()[:2, :2]
                self.draw_cov(mean, cov, confidence=0.95)

        for target in targets:
            target_state = target.getState()
            self.draw_target(target_state, size=target_size)

        plt.draw()
