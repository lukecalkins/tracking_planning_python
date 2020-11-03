import numpy as np
import matplotlib.pyplot as plt

class LRDT:
    def __init__(self, x_range, y_range, velocities, prior_prob_targ):
        self.vel_sheets = []
        self.x_range = x_range
        self.y_range = y_range
        self.prior_prob_targ = prior_prob_targ
        self.num_vel = len(velocities)
        self.build_velocity_sheets(velocities)


    def build_velocity_sheets(self, velocities):
        total_sheets = len(velocities)
        for vel in velocities:
            self.vel_sheets.append(VelocitySheet(self.x_range, self.y_range, vel, self.prior_prob_targ, self.num_vel))

    def motion_update(self, dt=1):

        for vel_sheet in self.vel_sheets:
            vel_sheet.move(dt)

        return None

    def measurement_update(self, y):

        return None

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
            self.counter_x = self.counter_x - self.cell_width_x
            translate_x = 1
        if self.counter_x < self.cell_width_x:
            self.counter_x = self.counter_x + self.cell_width_x
            translate_x = -1
        if self.counter_y > self.cell_width_y:
            self.counter_y = self.counter_y - self.cell_width_y
            translate_y = 1
        if self.counter_y < self.cell_width_y:
            self.counter_y = self.counter_y + self.cell_width_y
            translate_y = -1

        translate = (translate_x, translate_y)
        # add or remove rows/columns as necessary
        if translate_x != 0 or translate_y != 0:
            self.translate_cells(translate)

    def translate_cells(self, translate):
        translate_x = translate[0]
        translate_y = translate[1]

        if translate_x != 0:
            num_columns_to_add = np.abs(translate_x)
            columns_to_add = self.prior_prob_targ / (1 - self.prior_prob_targ) / self.num_vel / \
                             np.prod(self._gridx.shape) / np.ones((self._gridx.shape[0], num_columns_to_add))
            if translate_x > 0:
                # add new column(s) on far left side
                with_columns = np.hstack((columns_to_add, self.lr))
                self.lr = with_columns[:, num_columns_to_add:]
            else:
                # add new column(s) to far right side
                with_columns = np.hstack((self.lr, columns_to_add))
                self.lr = with_columns[:, :-1 * num_columns_to_add]

        if translate_y != 0:
            num_rows_to_add = np.abs(translate_y)
            rows_to_add = self.prior_prob_targ / (1 - self.prior_prob_targ) / self.num_vel / \
                             np.prod(self._gridx.shape) / np.ones((num_rows_to_add, self._gridx.shape[1]))
            if translate_y > 0:
                # add new rows to top
                with_rows = np.vstack((rows_to_add, self.lr))
                self.lr = with_rows[:-1 * num_rows_to_add, :]
            else:
                # add new rows to bottom
                with_rows = np.vstack((self.lr, rows_to_add))
                self.lr = with_rows[num_rows_to_add:, :]

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




