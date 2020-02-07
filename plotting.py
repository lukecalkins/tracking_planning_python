import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

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

    def plot_state(self, robots, targets, robot_size=1, target_size=1):

        self.clear_plot()
        self.draw_env()

        for robot in robots:
            pose = robot.getState()
            self.draw_robot(pose, size=robot_size)

        for target in targets:
            target_state = target.getState()
            self.draw_target(target_state, size=target_size)

        plt.draw()
