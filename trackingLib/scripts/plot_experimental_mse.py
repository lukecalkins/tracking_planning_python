import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
import numpy as np
from matplotlib.lines import Line2D


if __name__ == '__main__':
    directory = '/Users/william.calkins/Documents/Research/Tracking/fall2020/'


    #configuration 1, 10.11.20
    # successes = 4,6,8,9,10,11,12,13,16
    # failures = 5 (Switched tracks),  or 7

    #configurations used in plots in paper
    #config 1 - 10/12/20 Runs 11 and 12
    #config 2 - 10/14/20 Runs 5 and 23

    run = '101220/run11/'
    jpdam_data = np.load(directory + run + 'mse.npz')
    jpdam_data = jpdam_data['data']
    print("jpdam data shape: ", jpdam_data.shape)
    run = '101220/run12/'
    nn_data = np.load(directory + run + 'mse.npz')
    nn_data = nn_data['data']
    print("nn data shape: ", nn_data.shape)

    fig, ax = plt.subplots()
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
    ax.set_title('RMS Position Error for Representative Mission Configuration 1')
    plt.tight_layout()
    plt.show()


    exit(0)


