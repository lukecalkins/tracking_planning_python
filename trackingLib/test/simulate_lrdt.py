import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import patches
import imageio
import numpy as np

from trackingLib.LRDT import LRDT, aspect_dependent_b_sigma
from trackingLib.robot import Robot
from trackingLib.target import Target
from trackingLib.params import Parameters
import json
from trackingLib.sensor import Measurement

pi = np.pi
twopi = 2 * pi

class PlotterLRDT:

    def __init__(self, map_min, map_max, video=False):
        self.map_min = map_min
        self.map_max = map_max
        self.fig = plt.figure()
        self.video = video
        self.clr_list = ['r', 'b', 'g', 'm', 'y']
        if self.video:
            self.images = []

    def draw_env(self):
        self.ax_sheet = self.fig.add_subplot(221)
        self.ax_sheet.set_title('$\Lambda(t, s)$')
        self.ax_scene = self.fig.add_subplot(222)
        self.ax_scene.set_aspect('equal')
        self.ax_log_sheet = self.fig.add_subplot(223)
        self.ax_log_sheet.set_title('log($\Lambda(t, s)$)')
        self.ax_meas_sheet = self.fig.add_subplot(224)
        self.ax_meas_sheet.set_title('Measurement likelihood ratio')
        self.ax_scene.set_xlim(self.map_min[0], self.map_max[0])
        self.ax_scene.set_ylim(self.map_min[1], self.map_max[1])

        plt.tight_layout()

    def draw_ownship(self, own_state, clr='b', size=5, ax=None):
        x = own_state[0]
        y = own_state[1]

        # point robotrs
        # self.ax.plot(x, y, clr + '.', markersize=size)

        th = own_state[2]
        # Construct Triangle Polygon.
        XY = np.array([[x + size * np.cos(th), x + size * np.cos(th - 2.7), x + size * np.cos(th + 2.7)],
                       [y + size * np.sin(th), y + size * np.sin(th - 2.7), y + size * np.sin(th + 2.7)]]).transpose()

        self.ax_scene.add_patch(patches.Polygon(XY, facecolor=clr, zorder=2, alpha=0.5))

    def draw_contact_lines(self, own_state, measurements):

        max_range = 100  # range to plot line in meters
        for measurement in measurements:
            contact = measurement.getZ()
            ID = measurement.getID()
            global_contact_heading = own_state[2] + contact
            global_contact_heading_mirror = own_state[2] - contact
            x_contact_point = own_state[0] + max_range * np.cos(global_contact_heading)
            y_contact_point = own_state[1] + max_range * np.sin(global_contact_heading)
            x_contact_point_mirror = own_state[0] + max_range * np.cos(global_contact_heading_mirror)
            y_contact_point_mirror = own_state[1] + max_range * np.sin(global_contact_heading_mirror)
            if ID >= 0:
                color = self.clr_list[ID]
            else:
                color = 'k'
            self.ax_scene.plot([own_state[0], x_contact_point], [own_state[1], y_contact_point],
                               c=color, linestyle='--', alpha=0.5)
            self.ax_scene.plot([own_state[0], x_contact_point_mirror], [own_state[1], y_contact_point_mirror],
                               c=color, linestyle='--', alpha=0.5)

    def add_image(self, integrated_sheet, measurement_lr_sheet, own_state, targets, measurements):
        self.fig.clf()
        self.draw_env()

        # draw integreated lrdt surface
        sheet = self.ax_sheet.imshow(integrated_sheet, origin='lower')
        self.fig.colorbar(sheet, ax=self.ax_sheet)
        self.ax_sheet.set_xlabel('x')
        self.ax_sheet.set_ylabel('y')

        # draw log of integrated lrdt surface
        log_integrated_lr_sheet = np.log(integrated_sheet)
        log_sheet = self.ax_log_sheet.imshow(log_integrated_lr_sheet, origin='lower')
        self.fig.colorbar(log_sheet, ax=self.ax_log_sheet)
        self.ax_log_sheet.set_xlabel('x')
        self.ax_log_sheet.set_ylabel('y')

        #draw measurement likelihood ratio
        meas_lr_sheet = self.ax_meas_sheet.imshow(measurement_lr_sheet, origin='lower')
        self.fig.colorbar(meas_lr_sheet, ax=self.ax_meas_sheet)
        self.ax_meas_sheet.set_xlabel('x')
        self.ax_meas_sheet.set_ylabel('y')
        meas_lr_sheet.set_clim(0, 10)

        # draw ownship and bearing lines
        self.draw_ownship(own_state)
        self.draw_contact_lines(own_state, measurements)

        # draw target
        targ_ndx = 0
        for target in targets:
            targ_state = target.getState()
            self.ax_scene.plot(targ_state[0], targ_state[1], self.clr_list[targ_ndx], marker='.', markersize=10)
            targ_ndx += 1

        # save image for video
        #plt.tight_layout()
        if self.video:
            canvas = FigureCanvas(self.fig)
            canvas.draw()
            width, height = self.fig.get_size_inches() * self.fig.get_dpi()
            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            self.images.append(image)

    def save_video(self, directory, filename, fps):
        imageio.mimsave(directory + filename + '.mp4', self.images, fps=fps)

def sense_bearing_contacts(own_state, target_model, sensor, clutter_density):
    # create bearing contacts
    bearings = []
    measurements, _ = sensor.sense_targets_ambiguity(own_state, target_model.getTargets())

    # add clutter contacts
    fov = pi
    poisson_parameter = fov * clutter_density
    num_clutter_measurements = np.random.poisson(poisson_parameter)
    clutter_measurements = np.random.uniform(0, pi, size=num_clutter_measurements)
    for clutter_meas in clutter_measurements:
        measurements.append(Measurement(clutter_meas, -1, 1))

    return measurements

if __name__ == '__main__':
    working_directory = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib'
    p = Parameters(working_directory)

    plot_measurement_noise = False
    if plot_measurement_noise:
        angles = np.arange(0, 181)
        bearing_sigmas = []
        for angle in angles:
            bearing_sigmas.append(aspect_dependent_b_sigma(angle * np.pi/180))
        bearing_sigmas = np.array(bearing_sigmas)
        fig, ax = plt.subplots()
        ax.plot(angles, bearing_sigmas*180/np.pi)
        ax.set_aspect(1)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 30)
        ax.set_xlabel('Relative bearing')
        ax.set_ylabel('$\sigma_{b}$')
        ax.set_title('Aspect-dependent measurement noise')
        plt.tight_layout()
        plt.show()


    #define grid space and velocities
    x_range = np.arange(100)
    y_range = np.arange(100)
    map_min = [x_range.min(), y_range.min()]
    map_max = [x_range.max(), y_range.max()]
    speeds = np.arange(0.2, 1.0, .2)
    num_headings = 24
    velocities = []
    for i in range(num_headings):
        heading = -pi + i / num_headings * twopi
        for speed in speeds:
            x_vel = speed * np.cos(heading)
            y_vel = speed * np.sin(heading)
            velocities.append((x_vel, y_vel))

    np.random.seed(p.random_seed)

    initial_own_state = np.array([30., 30., 0.])
    ownship = Robot(initial_own_state, None)  # don't include target motion model
    target_model = p.getWorld()
    sensor = p.getSensor()
    clutter_density = p.clutter_density

    prior_prob_targ = 0.25
    lrdt = LRDT(x_range, y_range, velocities, prior_prob_targ, clutter_density, video=True, endfire=True)
    plotter = PlotterLRDT(map_min, map_max, video=True)

    # perform initial measurement update
    y0 = sense_bearing_contacts(ownship.getState(), target_model, sensor, clutter_density)
    lrdt.measurement_update(ownship.getState(), y0, sensor)
    plotter.add_image(lrdt.get_integrated_position(), lrdt.measurement_lr_sheet,
                      ownship.getState(), target_model.getTargets(), y0)
    T_max = 40
    action = [1, 1./50]
    for i in range(T_max):
        target_model.forwardSimulate(1)
        ownship.applyControl(action, 1)
        y = sense_bearing_contacts(ownship.getState(), target_model, sensor, clutter_density)
        lrdt.motion_update(1)
        lrdt.measurement_update(ownship.getState(), y, sensor)
        plotter.add_image(lrdt.get_integrated_position(), lrdt.measurement_lr_sheet,
                          ownship.getState(), target_model.getTargets(), y)
        print("timestep: " + str(i))

    directory = 'results/videos/lrdt/1targ/endfire/'
    filename = 'lrdt_cd_0.3183_bsigma_0.087_Pd_0.5_turning_left_slow'
    plotter.save_video(directory, filename, fps=1)

    print("Exiting Main")
