import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def restrict_angle(phi, min_range=-np.pi, max_range=np.pi):
    x = phi - min_range
    y = max_range - min_range
    return min_range + x - y * np.floor(x / y)

def aspect_dependent_b_sigma(bearing):
    steady_b_sigma = 5.
    max_b_sigma = 25.
    min_angle_degrees = 30.
    max_angle_degrees = 150.
    bearing_degrees = bearing * 180./np.pi
    if bearing_degrees >= min_angle_degrees and bearing_degrees <=  max_angle_degrees:
        return steady_b_sigma * np.pi / 180.
    elif bearing_degrees < min_angle_degrees:
        # point = (0, max_b_sigma)
        slope = (max_b_sigma - steady_b_sigma) / (0. - min_angle_degrees)
        value = max_b_sigma + slope * (bearing_degrees - 0.)
        return value * np.pi / 180.
    else:
        # point (180, max_b_sigma)
        slope = (max_b_sigma - steady_b_sigma) / (180. - max_angle_degrees)
        value = max_b_sigma + slope * (bearing_degrees - 180)
        return value * np.pi / 180.

class LRDT:
    def __init__(self, x_range, y_range, velocities, prior_prob_targ, clutter_density, video=False, endfire=False):
        self.vel_sheets = []
        self.x_range = x_range
        self.y_range = y_range
        self.prior_prob_targ = prior_prob_targ
        self.num_vel = len(velocities)
        self.build_velocity_sheets(velocities)
        self.clutter_density = clutter_density
        self.video = video
        self.endfire = endfire      # if true, use aspect dependent variance in likelihood
        self.fig = plt.figure()
        self.measurement_lr_sheet = None
        if self.video:
            self.images = []

    def build_velocity_sheets(self, velocities):
        total_sheets = len(velocities)
        for vel in velocities:
            self.vel_sheets.append(VelocitySheet(self.x_range, self.y_range, vel, self.prior_prob_targ, self.num_vel))

    def motion_update(self, dt=1):

        for vel_sheet in self.vel_sheets:
            vel_sheet.move(dt)

        return None

    def measurement_update(self, own_state, y, sensor):

        # create likelihood ratio sheet that is save for all velocities sheets
        measurement_lr_sheet = np.zeros((len(self.y_range), len(self.x_range)))
        X, Y = np.meshgrid(self.x_range, self.y_range)

        # need to calculate likelihood function in  every cell
        # restrict_angle(np.arctan2(y[1] - x[1], y[0] - x[0]) - x[2])
        # first get ambiguous bearing measurement in each cell relative to ownstate
        bearing_sheet = np.abs(restrict_angle(np.arctan2(Y - own_state[1], X - own_state[0]) - own_state[2]))

        b_sigma = sensor.get_b_sigma()
        P_d = sensor.get_probability_of_detection()
        for measurement in y:
            bearing = measurement.getZ()
            bearing_diff = bearing - bearing_sheet
            # compute likelihood sheet given the bearing sheet and target bearing
            if self.endfire:
                b_sigma_measurement = aspect_dependent_b_sigma(bearing)
                gaussian_likelihood_sheet_for_measurement = 1./b_sigma_measurement/np.sqrt(2*np.pi) * \
                    np.exp(-1./(2*b_sigma_measurement**2) * bearing_diff ** 2)
            else:
                gaussian_likelihood_sheet_for_measurement = 1./b_sigma/np.sqrt(2*np.pi) * \
                                                        np.exp(-1./(2*b_sigma**2) * bearing_diff ** 2)
            measurement_lr_sheet_for_measurement = P_d / self.clutter_density * gaussian_likelihood_sheet_for_measurement + 1 - P_d
            measurement_lr_sheet = measurement_lr_sheet + measurement_lr_sheet_for_measurement

        self.measurement_lr_sheet = measurement_lr_sheet
        for i in range(len(self.vel_sheets)):
            self.vel_sheets[i].lr = self.vel_sheets[i].lr * measurement_lr_sheet


    def call_detections(self):
        """
        function that thresholds the integrated lrdt surface and calls detections and resets surface accordingly
        :return:
        """

        # get position sheet by integrating over velocity
        integrated_sheet = self.get_integrated_position()

        convolved_sheet = self.convolve_lr_sheet(integrated_sheet)

        return None

    def get_integrated_position(self):
        """
        sums the likelihood ratio over all velocity sheets
        :return:
        """
        integrated_sheet = np.zeros((len(self.y_range), len(self.x_range)))
        for vel_sheet in self.vel_sheets:
            integrated_sheet += vel_sheet.lr

        return integrated_sheet

    def convolve_lr_sheet(self, integrated_sheet):

        # find peak
        X, Y = np.meshgrid(self.x_range, self.y_range)
        map_min = (self.x_range[0], self.y_range[0])
        map_max = (self.x_range[-1], self.y_range[-1])
        ndx = np.unravel_index(integrated_sheet.argmax(), integrated_sheet.shape)
        fig, ax = plt.subplots()
        ax.imshow(integrated_sheet, origin='lower', extent=(map_min[0], map_max[0], map_min[1], map_max[1]))
        peak_position = (X[ndx[0], ndx[1]], Y[ndx[0], ndx[1]])
        ax.plot(X[ndx[0], ndx[1]], Y[ndx[0], ndx[1]], marker='*', c='r')

        #plt.show()

        #use gaussian effectiveness function
        sigma_kernel = 200
        num_pixels = 10
        kernel_x_range = (ndx[0] - num_pixels, ndx[0] + num_pixels + 1)
        kernel_y_range = (ndx[1] - num_pixels, ndx[1] + num_pixels + 1)
        E_ij = -1. / (2 * sigma_kernel ** 2) * ((X - peak_position[0]) ** 2 + (Y - peak_position[1]) ** 2)
        #E_ij = -1. / (2 * sigma_kernel ** 2) * ((X[kernel_x_range[0]:kernel_x_range[1], kernel_y_range[0]:kernel_y_range[1]] - peak_position[0]) ** 2 +
        #                                        (Y[kernel_x_range[0]:kernel_x_range[1], kernel_y_range[0]:kernel_y_range[1]] - peak_position[1]) ** 2)
        E_ij = np.exp(E_ij)
        fig, ax = plt.subplots()
        E_ij_sheet = ax.imshow(E_ij, origin='lower', extent=(map_min[0], map_max[0], map_min[1], map_max[1]))
        #E_ij_sheet = ax.imshow(E_ij, origin='lower', extent=(self.x_range[kernel_x_range[0]], self.x_range[kernel_x_range[1]], self.y_range[kernel_y_range[0]], self.y_range[kernel_y_range[1]]))
        fig.colorbar(E_ij_sheet, ax=ax)
        #convolved_image = ndimage.convolve(integrated_sheet, E_ij, mode='constant', cval=0.0)
        output_image = E_ij * integrated_sheet
        output_sum = np.sum(output_image)

        fig, ax = plt.subplots()
        ax.imshow(output_image, origin='lower', extent=(map_min[0], map_max[0], map_min[1], map_max[1]))
        ax.set_title(str(output_sum))

    def plot_vel_sheet(self, sheet_index, log_sheet=False):
        sheet = self.vel_sheets[sheet_index]
        sheet.plot_sheet(log_sheet)

class VelocitySheet:
    def __init__(self,  x_range, y_range, velocity, prior_prob_targ, num_vel):
        X, Y = np.meshgrid(x_range, y_range)
        self._gridx = X
        self._gridy = Y
        self.vel = velocity
        self.counter_x = 0
        self.counter_y = 0
        self.cell_width_x = x_range[1] - x_range[0]
        self.cell_width_y = y_range[1] - y_range[0]
        self.prior_prob_targ = prior_prob_targ
        self.num_vel = num_vel
        self.lr = self.prior_prob_targ / (1 - self.prior_prob_targ) / self.num_vel / np.prod(self._gridx.shape) / \
                  np.ones(self._gridx.shape)
        self.log_lr = np.log(self.lr)

    def move(self, dt):
        """
        perform motion update for velocity  sheet
        :param dt:
        :return:
        """
        self.counter_x += self.vel[0] * dt
        self.counter_y += self.vel[1] * dt
        translate_x = 0
        translate_y = 0
        if self.counter_x > self.cell_width_x:
            translate_x = np.floor(self.counter_x / self.cell_width_x)
            self.counter_x = self.counter_x - translate_x * self.cell_width_x
        if self.counter_x < -1 * self.cell_width_x:
            translate_x = np.ceil(self.counter_x / self.cell_width_x)
            self.counter_x = self.counter_x - translate_x * self.cell_width_x
        if self.counter_y > self.cell_width_y:
            translate_y = np.floor(self.counter_y / self.cell_width_y)
            self.counter_y = self.counter_y - translate_y * self.cell_width_y
        if self.counter_y < -1 * self.cell_width_y:
            translate_y = np.ceil(self.counter_y / self.cell_width_y)
            self.counter_y = self.counter_y - translate_y * self.cell_width_y

        translate = (int(translate_x), int(translate_y))
        # add or remove rows/columns as necessary
        if translate_x != 0 or translate_y != 0:
            self.translate_cells(translate)

    def translate_cells(self, translate):
        translate_x = translate[0]
        translate_y = translate[1]

        if translate_x != 0:
            num_columns_to_add = np.abs(translate_x)
            columns_to_add = self.prior_prob_targ / (1 - self.prior_prob_targ) / self.num_vel / \
                             np.prod(self._gridx.shape) * np.ones((self._gridx.shape[0], num_columns_to_add))
            if translate_x > 0:
                with_columns = np.hstack((columns_to_add, self.lr))  # add new column(s) on far left side
                self.lr = with_columns[:, :-1 * num_columns_to_add]  # delete columns on far right
            else:

                with_columns = np.hstack((self.lr, columns_to_add))  # add new column(s) to far right side
                self.lr = with_columns[:, num_columns_to_add:]       # delete column(s) on far left

        if translate_y != 0:
            num_rows_to_add = np.abs(translate_y)
            rows_to_add = self.prior_prob_targ / (1 - self.prior_prob_targ) / self.num_vel / \
                             np.prod(self._gridx.shape) * np.ones((num_rows_to_add, self._gridx.shape[1]))
            if translate_y > 0:
                with_rows = np.vstack((self.lr, rows_to_add))        # add new rows to bottom
                self.lr = with_rows[num_rows_to_add:, :]        # delete rows from top
            else:
                with_rows = np.vstack((rows_to_add, self.lr))        # add new rows to bottom
                self.lr = with_rows[:-1 * num_rows_to_add, :]             # delete rows from top

        # take log of result
        self.log_lr = np.log(self.lr)

    def plot_sheet(self, log_sheet):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if log_sheet:
            ax.plot_surface(self._gridx, self._gridy, self.log_lr)
        else:
            ax.plot_surface(self._gridx, self._gridy, self.lr)




