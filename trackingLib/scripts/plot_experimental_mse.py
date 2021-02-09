import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from matplotlib.lines import Line2D
import json


if __name__ == '__main__':
    directory = '/Users/william.calkins/Documents/Research/Tracking/fall2020/'


    #configuration 1, 10.11.20
    # successes = 4,6,8,9,10,11,12,13,16
    # failures = 5 (Switched tracks),  or 7

    #configurations used in plots in paper
    #config 1 - 10/12/20 Runs 11 and 12
    #config 2 - 10/14/20 Runs 5 and 23

    run = '101420/run5/'
    jpdam_data = np.load(directory + run + 'mse.npz')
    jpdam_data = jpdam_data['data']
    print("jpdam data shape: ", jpdam_data.shape)
    run = '101420/run23/'
    nn_data = np.load(directory + run + 'mse.npz')
    nn_data = nn_data['data']
    print("nn data shape: ", nn_data.shape)

    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    fig_height = 3
    fig.set_figheight(fig_height)
    num_targs = 2
    clr_list = ['r', 'b']
    for i in range(num_targs):
        ax.plot(jpdam_data[i, :], c=clr_list[i])
        ax.plot(nn_data[i, :], c=clr_list[i], linestyle='dotted')
    legend_elements = [Line2D([0], [0], color='k', lw=1, ls='-', label='Merged Measurement JPDAF'),
                       Line2D([0], [0], color='k', lw=1, ls='dotted', label='Nearest Neighbors')]
    ax.legend(handles=legend_elements, loc='upper left')
    ax.set_xlabel('Tracking Iteration')
    ax.set_ylabel('RMSE (position)')
    ax.set_title('RMS Position Error for Representative Mission Configuration 2')
    plt.tight_layout()
    #plt.show()

    # plot top level of scene beginning to end
    file = directory + run + 'tracker_log.json'
    with open(file) as f:
        data = json.load(f)
    ground_truth_file = directory + run + 'state_log.json'
    with open(ground_truth_file) as f:
        ground_data = json.load(f)

    aion1_x = []
    aion1_y = []
    aion2_x = []
    aion2_y = []
    aion3_x = []
    aion3_y = []
    for i in range(len(data)):
        time_step_data = data[str(i)]
        state_iteration = time_step_data['state_iteration']
        aion1 = ground_data['aion1'][state_iteration]
        aion2 = ground_data['aion2'][state_iteration]
        aion3 = ground_data['aion3'][state_iteration]

        aion1_x.append(aion1['x'])
        aion1_y.append(aion1['y'])
        aion2_x.append(aion2['x'])
        aion2_y.append(aion2['y'])
        aion3_x.append(aion3['x'])
        aion3_y.append(aion3['y'])

    fig, ax = plt.subplots()
    #  trajectories
    #ax.plot(aion1_x, aion1_y, 'k', label='Sensor Start')
    ax.plot(aion2_x, aion2_y, 'r', label='Target 1')
    ax.plot(aion3_x, aion3_y, 'b', label='Target 2')
    # starting points
    ax.plot(aion1_x[0], aion1_y[0], 'k.', markersize=20, label='Sensor Start')
    ax.plot(aion2_x[0], aion2_y[0], 'r.', markersize=20)
    ax.plot(aion3_x[0], aion3_y[0], 'b.', markersize=20)
    ax.legend()
    ax.set_title('Scene 2')
    ax.set_xlabel('x (meters)')
    ax.set_ylabel('y (meters')
    plt.tight_layout()
    plt.show()




    exit(0)


