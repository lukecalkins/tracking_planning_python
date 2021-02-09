import scipy.io
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import patches
import imageio
import numpy as np
from trackingLib.sensor import Measurement
from trackingLib.LRDT import LRDT, aspect_dependent_b_sigma
from trackingLib.params import Parameters
from trackingLib.utils import restrict_angle

pi = np.pi
twopi = 2 * pi

class PlotterLRDT_exp:

    def __init__(self, map_min, map_max, video=False):
        self.map_min = map_min
        self.map_max = map_max
        self.fig = plt.figure()
        self.video = video
        self.clr_list = ['r', 'b', 'g', 'c', 'y']
        self.meas_index = 1
        if self.video:
            self.images = []

    def draw_env(self, integrated_sheet):
        integrated_likelihood_ratio = np.sum(integrated_sheet)
        self.ax_sheet = self.fig.add_subplot(221)
        self.ax_sheet.set_title('$\Lambda(t, s)$'+ '\n p(s)/p($\phi$) = ' + str(integrated_likelihood_ratio))
        self.ax_scene = self.fig.add_subplot(222)
        self.ax_scene.set_aspect('equal')
        self.ax_log_sheet = self.fig.add_subplot(223)
        self.ax_log_sheet.set_title('log($\Lambda(t, s)$)')
        self.ax_meas_sheet = self.fig.add_subplot(224)
        self.ax_meas_sheet.set_title('Measurement likelihood ratio')
        self.ax_scene.set_xlim(self.map_min[0], self.map_max[0])
        self.ax_scene.set_ylim(self.map_min[1], self.map_max[1])
        self.ax_scene.set_title('Measurement index: ' + str(self.meas_index))
        self.meas_index += 1

        plt.tight_layout()

    def draw_ownship(self, own_state, clr='b', size=20, ax=None):
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

        max_range = 5000  # range to plot line in meters
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
        self.draw_env(integrated_sheet)

        # draw integreated lrdt surface
        sheet = self.ax_sheet.imshow(integrated_sheet, origin='lower',
                                     extent=(self.map_min[0],  self.map_max[0], self.map_min[1], self.map_max[1]))
        self.fig.colorbar(sheet, ax=self.ax_sheet)
        self.ax_sheet.set_xlabel('x')
        self.ax_sheet.set_ylabel('y')

        # draw log of integrated lrdt surface
        log_integrated_lr_sheet = np.log(integrated_sheet)
        log_sheet = self.ax_log_sheet.imshow(log_integrated_lr_sheet, origin='lower',
                                             extent=(self.map_min[0],  self.map_max[0], self.map_min[1], self.map_max[1]))
        self.fig.colorbar(log_sheet, ax=self.ax_log_sheet)
        self.ax_log_sheet.set_xlabel('x')
        self.ax_log_sheet.set_ylabel('y')

        #draw measurement likelihood ratio
        meas_lr_sheet = self.ax_meas_sheet.imshow(measurement_lr_sheet, origin='lower',
                                                  extent=(self.map_min[0],  self.map_max[0], self.map_min[1], self.map_max[1]))
        self.fig.colorbar(meas_lr_sheet, ax=self.ax_meas_sheet)
        self.ax_meas_sheet.set_xlabel('x')
        self.ax_meas_sheet.set_ylabel('y')
        meas_lr_sheet.set_clim(0, 10)

        # draw ownship and bearing lines
        self.draw_ownship(own_state)
        self.draw_contact_lines(own_state, measurements)

        # draw target
        targ_ndx = 0
        for target_position in targets:
            self.ax_scene.plot(target_position[0], target_position[1], self.clr_list[targ_ndx], marker='.', markersize=10)
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

def create_measurements_from_contacts(contacts_vec):
    measurements = []
    for contact in contacts_vec:
        if not np.isnan(contact):
            measurements.append(Measurement(contact, -1, 1))
    return measurements


def create_passive_curve_movie(passive_curves, contacts, fps=5):
    """

    :param passive_curves: rows  are time indices, columns are the beam values
    :return:
    """
    fig = plt.figure(figsize=(8, 8))
    num_curves, num_beams = passive_curves.shape
    max_contacts = contacts.shape[0]
    images = []
    clr_map = ['r', 'b', 'g', 'm', 'c']
    for i in range(num_curves):
        fig.clf()
        ax = fig.add_subplot()
        ax.set_title('Measurement index: ' + str(i))
        curve = passive_curves[i, :]
        # flip the curve before plotting (same  as pi - contact)
        ax.plot(np.flip(curve))
        for k in range(max_contacts):
            if not np.isnan(contacts[k, i]):
                contact = contacts[k, i] * 180 / np.pi
                ax.axvline(x=contact, c=clr_map[k], alpha=0.5)
            else:
                break
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        images.append(image)

    # save images to video
    direc= 'data/matlab/'
    filename = 'sensor_tracks_3_28_no_flip'
    imageio.mimsave(direc + filename + '.mp4', images, fps=fps)

    return None


def angular_dist(ang1, ang2):
    ang1 = np.mod(ang1, 2*np.pi)
    ang2 = np.mod(ang2, 2*np.pi)
    ang_diff = ang2 - ang1
    if ang_diff > np.pi:
        ang_diff = 2*np.pi - ang_diff
    elif -ang_diff > np.pi:
        ang_diff = 2*np.pi + ang_diff
    elif ang_diff < 0:
        ang_diff = -ang_diff

    return ang_diff

def overlay_ground_truth_on_passive_response(passive_curves, contacts, ships, ownship, fps=5):
    fig = plt.figure(figsize=(8, 8))
    num_curves, num_beams = passive_curves.shape
    max_contacts = contacts.shape[0]
    images = []
    clr_map = ['r', 'b', 'g', 'm', 'c', 'y']
    for i in range(num_curves):
        fig.clf()
        ax = fig.add_subplot()
        ax.set_title('Measurement index: ' + str(i))
        curve = passive_curves[i, :]
        # flip the curve before plotting (same  as pi - contact)
        ax.plot(np.flip(curve))
        ships_key = ['res', 'jam', 'ste', 'nat', 'nor', 'bos']

        # overlay ground truth ship's bearing
        for k in range(len(ships)):
            ship = ships[k]
            delE = ship['east'][i] - ownship['east'][i]
            delN = ship['north'][i] - ownship['north'][i]
            bearing = angular_dist(np.arctan2(delE, delN), ownship['heading'][i]) * 180 / np.pi
            ax.axvline(x=bearing, c=clr_map[k], label=ships_key[k])

        # plot called contacts along passive curve line
        beams = np.arange(0, num_beams + 1)
        for j in range(max_contacts):
            if not np.isnan(contacts[j, i]):
                # find closest beam
                contact_beam = contacts[j, i] * 180 / np.pi
                ndx = np.argmin(np.abs(beams - contact_beam))
                ax.plot(ndx, np.flip(curve)[ndx], marker='*', markersize=15, c='k')



        #ax.set_ylim([0, 1.6e11])
        ax.legend()
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        images.append(image)

    # save images to video
    direc = 'data/matlab/'
    filename = 'sensor_ground_truth_tracks_196_213_all_ships'
    imageio.mimsave(direc + filename + '.mp4', images, fps=fps)

    return None

def overlay_ground_truth_on_passive_response_with_freq(passive_curves, passive_curves_freq, contacts, ships, ownship,
                                                       fps=5):
    fig = plt.figure(figsize=(16, 8))
    num_curves, num_beams = passive_curves.shape
    num_freq_bands = 16
    max_contacts = contacts.shape[0]
    images = []
    clr_map = ['r', 'b', 'g', 'c', 'y']
    for i in range(num_curves):
        fig.clf()
        ax_full = fig.add_subplot(121)
        ax_full.set_title('Measurement index: ' + str(i))
        curve = passive_curves[i, :]
        # flip the curve before plotting (same  as pi - contact)
        ax_full.plot(np.flip(curve))
        ships_key = ['res', 'jam', 'ste', 'nor', 'bos']

        freq_curve = passive_curves_freq[i, :]
        freq_matrix = np.zeros((num_freq_bands,  num_beams))
        for band in range(num_freq_bands):
            for beam in range(num_beams):
                freq_matrix[band, beam] = freq_curve[band + beam * num_freq_bands]

        for band in range(num_freq_bands):
            # multiply by total amplitudes curve
            freq_matrix[band, :] = freq_matrix[band, :] * curve

        # normalize
        freq_matrix = freq_matrix / np.max(freq_matrix)

        # plot the bands
        ax_freq = fig.add_subplot(122)
        ax_freq.set_title('Frequency bands')
        for k in range(num_freq_bands):
            ax_freq.plot(np.flip(freq_matrix[k, :]))

        # overlay ground truth ship's bearing
        for k in range(len(ships)):
            ship = ships[k]
            delE = ship['east'][i] - ownship['east'][i]
            delN = ship['north'][i] - ownship['north'][i]
            bearing = angular_dist(np.arctan2(delE, delN), ownship['heading'][i]) * 180 / np.pi
            ax_full.axvline(x=bearing, c=clr_map[k], label=ships_key[k])
            ax_freq.axvline(x=bearing, c=clr_map[k], label=ships_key[k])

        # plot called contacts along passive curve line
        beams = np.arange(0, num_beams + 1)
        for j in range(max_contacts):
            if not np.isnan(contacts[j, i]):
                # find closest beam
                contact_beam = contacts[j, i] * 180 / np.pi
                ndx = np.argmin(np.abs(beams - contact_beam))
                ax_full.plot(ndx, np.flip(curve)[ndx], marker='*', markersize=15, c='k')

        # ax.set_ylim([0, 1.6e11])
        ax_full.legend()
        plt.tight_layout()
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        images.append(image)

    # save images to video
    direc = 'data/matlab/'
    filename = 'sensor_ground_truth_freq_tracks_196_213_all_ships'
    imageio.mimsave(direc + filename + '.mp4', images, fps=fps)


def plot_measurement_likelihood_ratio_function(y, b_sigma, meas_index):

    fig, ax = plt.subplots()
    likelihood_function = np.zeros(181)
    bearing_beams = np.arange(0, 181) * np.pi / 180
    for measurement in y:
        bearing = measurement.getZ()
        bearing_diff = bearing - bearing_beams
        likelihood_function =likelihood_function + 1./b_sigma/np.sqrt(2*np.pi) * \
                                                        np.exp(-1./(2*b_sigma**2) * bearing_diff ** 2)
    ax.plot(np.arange(0, 181), likelihood_function)
    ax.set_title('Measurement Likelihood ratio function, Measurement  index: ' + str(meas_index))
    plt.show()
    return None


if __name__ == '_' \
               '_main__':
    mat = scipy.io.loadmat('data/matlab/aug15_tracks_3_28.mat')
    #mat = scipy.io.loadmat('data/matlab/aug15_tracks_135_166.mat')
    #mat = scipy.io.loadmat('data/matlab/aug15_tracks_196_213.mat')
    contacts = mat['contacts']
    passive_curves = mat['passive_curves']
    passive_curves_freq = mat['passive_curves_freq']
    res = mat['res']
    jam = mat['jam']
    ownship = mat['ownship']

    # turn matlab structs into dicts
    res = {'east': res['east'][0][0][0, :],
           'north': res['north'][0][0][0, :]}
    jam = {'east': jam['east'][0][0][0, :],
           'north': jam['north'][0][0][0, :]}
    ste = None
    nat = None
    nor = None
    bos = None
    if mat.__contains__('ste'):
        ste = mat['ste']
        ste = {'east': ste['east'][0][0][0, :],
               'north': ste['north'][0][0][0, :]}
    if mat.__contains__('nat'):
        nat = mat['nat']
        nat = {'east': nat['east'][0][0][0, :],
               'north': nat['north'][0][0][0, :]}
    if mat.__contains__('nor'):
        nor = mat['nor']
        nor = {'east': nor['east'][0][0][0, :],
               'north': nor['north'][0][0][0, :]}
    if mat.__contains__('bos'):
        bos = mat['bos']
        bos = {'east': bos['east'][0][0][0, :],
               'north': bos['north'][0][0][0, :]}


    ownship = {'east': ownship['east'][0][0][0, :],
               'north': ownship['north'][0][0][0, :],
               'heading': ownship['heading'][0][0][0, :],
               'time': ownship['time'][0][0][0, :]}

    # plot the target and ownship trajectories
    time = ownship['time'] - ownship['time'][0]

    if False:
        fig, ax = plt.subplots()
        ax.plot(res['east'], res['north'], 'r', label='RES')
        ax.plot(res['east'][0], res['north'][0], 'k', marker='o')
        ax.plot(res['east'][-1], res['north'][-1], 'k', marker='s')
        ax.plot(jam['east'], jam['north'], 'b', label='JAM')
        ax.plot(jam['east'][0], jam['north'][0], 'k', marker='o')
        ax.plot(jam['east'][-1], jam['north'][-1], 'k', marker='s')
        ax.plot(ste['east'], ste['north'], 'g', label='STE')
        ax.plot(ste['east'][0], ste['north'][0], 'k', marker='o')
        ax.plot(ste['east'][-1], ste['north'][-1], 'k', marker='s')
        ax.plot(ownship['east'], ownship['north'], 'c', label='AUV')
        ax.plot(ownship['east'][0], ownship['north'][0], 'k', marker='o')
        ax.plot(ownship['east'][-1], ownship['north'][-1], 'k', marker='s')
        if nat is not None:
            ax.plot(nat['east'], nat['north'], 'm', label='NAT')
            ax.plot(nat['east'][0], nat['north'][0], 'k', marker='o')
            ax.plot(nat['east'][-1], nat['north'][-1], 'k', marker='s')
        if nor is not None:
            ax.plot(nor['east'], nor['north'], 'c', label='NOR')
            ax.plot(nor['east'][0], nor['north'][0], 'k', marker='o')
            ax.plot(nor['east'][-1], nor['north'][-1], 'k', marker='s')
        if bos is not None:
            ax.plot(bos['east'], bos['north'], 'y', label='BOS')
            ax.plot(bos['east'][0], bos['north'][0], 'k', marker='o')
            ax.plot(bos['east'][-1], bos['north'][-1], 'k', marker='s')

        ax.legend()
        plt.show()

    # overlay contacts on the full passive curve
    #create_passive_curve_movie(passive_curves, contacts, fps=5)

    # overlay ground truth bearings on passive response
    ships = [res, jam, ste, nor, bos]
    #overlay_ground_truth_on_passive_response(passive_curves, contacts, ships, ownship, fps=5)
    # overlay_ground_truth_on_passive_response_with_freq(passive_curves, passive_curves_freq, contacts, ships, ownship, fps=5)

    working_directory = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib'
    p = Parameters(working_directory)

    # initialize LRDT
    num_x_cells = 100
    num_y_cells = 100
    x_range = np.linspace(-1700, 600, num_x_cells)
    y_range = np.linspace(-1750, 500, num_y_cells)
    map_min = [x_range.min(), y_range.min()]
    map_max = [x_range.max(), y_range.max()]
    num_headings = 24
    speeds = np.arange(0, 5.5, 0.5)
    velocities = []
    for i in range(num_headings):
        heading = -pi + i /num_headings * twopi
        for speed in speeds:
            xvel = speed * np.cos(heading)
            yvel = speed * np.sin(heading)
            vel = (xvel, yvel)
            velocities.append(vel)
    prior_prop_targ = 0.25
    sensor = p.sensor
    clutter_density = p.clutter_density
    lrdt = LRDT(x_range, y_range, velocities, prior_prop_targ, clutter_density, endfire=True)
    plotter = PlotterLRDT_exp(map_min, map_max, video=True)

    np.random.seed(p.random_seed)

    T_max = len(ownship['time'])
    #T_max = 200
    meas_index_to_plot = 4
    for i in range(1, T_max):
        y = create_measurements_from_contacts(contacts[:, i])
        dt = time[i] - time[i - 1]
        lrdt.motion_update(dt)
        own_state = np.array([ownship['east'][i], ownship['north'][i], pi/2 - ownship['heading'][i]])
        lrdt.measurement_update(own_state, y, sensor)
        targets = [[res['east'][i], res['north'][i]], [jam['east'][i], jam['north'][i]], [ste['east'][i], ste['north'][i]]]
        lrdt.call_detections()
        #targets = [[res['east'][i], res['north'][i]], [jam['east'][i], jam['north'][i]], [ste['east'][i], ste['north'][i]], [nor['east'][i], nor['north'][i]], [bos['east'][i], bos['north'][i]]]
        plotter.add_image(lrdt.get_integrated_position(), lrdt.measurement_lr_sheet,
                          own_state, targets, y)
        print("Timestep:  ", str(i))

    directory = 'results/videos/lrdt/aug15_2018/'
    filename = 'tracks_3_28_fps1_100cell'
    plotter.save_video(directory, filename, fps=1)
    filename = 'tracks_3_28_fps5_100cell'
    plotter.save_video(directory,  filename, fps=5)

    print("Exiting")

