import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
import numpy as np
import json
import sys, os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from trackingLib.utils import propagateOwnshipEuler


if __name__ == '__main__':

    wd = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib/'
    direc = 'results/sims_icra2021/4targ/NN/'

    y_dim = 4   #  dimension of target state
    clr_list = ['r', 'b', 'g', 'm', 'y']
    num_targs = 4
    T_sim = 100    # number of  time steps

    # Loop  through nearest neighbors sims and get statistics
    file = wd + direc
    file_count = 0
    mse_array_nn = np.zeros((num_targs, T_sim))
    lds_array_nn = np.zeros((num_targs, T_sim))
    for file in os.listdir(wd+direc):
        if file.endswith('.json'):
            with open(wd + direc + file) as nn_file:
                nn_data = json.load(nn_file)
            for i in range(len(nn_data)):
                ts_data = nn_data[str(i)]  # particular time step
                for j in range(num_targs):
                    start = j * y_dim
                    stop = start + y_dim
                    targ_est = np.array(ts_data['post_mean'][start:stop])
                    targ_true = np.array(ts_data['ground_truth'][start:stop])
                    post_cov = np.array(ts_data['post_covariance'])[start:stop, start:stop]
                    mse_array_nn[j][i] = mse_array_nn[j, i] + np.linalg.norm(targ_est[0:2] - targ_true[0:2])
                    lds_array_nn[j, i] = lds_array_nn[j, i] + np.log(np.linalg.det(post_cov))
            file_count += 1
    mse_array_nn = 1./file_count * mse_array_nn
    lds_array_nn = 1./file_count * lds_array_nn

    #calculate standard deviation at every time step for error bars
    mse_sd_array_nn = np.zeros((num_targs, T_sim))
    file_count = 0
    for file in os.listdir(wd+direc):
        if file.endswith('.json'):
            with open(wd + direc + file) as nn_file:
                nn_data = json.load(nn_file)
            for i in range(len(nn_data)):
                ts_data = nn_data[str(i)]
                for j in range(num_targs):
                    start = j * y_dim
                    stop = start + y_dim
                    targ_est = np.array(ts_data['post_mean'][start:stop])
                    targ_true = np.array(ts_data['ground_truth'][start:stop])
                    mse = np.linalg.norm(targ_est[0:2] - targ_true[0:2])
                    mse_sd_array_nn[j, i] = (mse_array_nn[j, i] - mse) ** 2
            file_count += 1

    mse_sd_array_nn = np.sqrt(1./file_count * mse_sd_array_nn)

    # Same thing for JPDAM
    direc = 'results/sims_icra2021/4targ/JPDAM/'
    file = wd + direc
    file_count = 0
    mse_array_jpdam = np.zeros((num_targs, T_sim))
    mse_sd_array_jpdam = np.zeros((num_targs, T_sim))
    lds_array_jpdam = np.zeros((num_targs, T_sim))
    for file in os.listdir(wd + direc):
        if file.endswith('.json'):
            with open(wd + direc + file) as jpdam_file:
                jpdam_data = json.load(jpdam_file)
            for i in range(len(jpdam_data)):
                ts_data = jpdam_data[str(i)]  # particular time step
                for j in range(num_targs):
                    start = j * y_dim
                    stop = start + y_dim
                    targ_est = np.array(ts_data['post_mean'][start:stop])
                    targ_true = np.array(ts_data['ground_truth'][start:stop])
                    post_cov = np.array(ts_data['post_covariance'])[start:stop, start:stop]
                    mse_array_jpdam[j, i] = mse_array_nn[j, i] + np.linalg.norm(targ_est[0:2] - targ_true[0:2])
                    lds_array_jpdam[j, i] = lds_array_jpdam[j, i] + np.log(np.linalg.det(post_cov))
            file_count += 1
    mse_array_jpdam = 1. / file_count * mse_array_jpdam
    lds_array_jpdam = 1./ file_count * lds_array_jpdam

    # calculate standard deviation for JPDAM at every time step for error bars
    mse_sd_array_jpdam = np.zeros((num_targs, T_sim))
    file_count = 0
    for file in os.listdir(wd + direc):
        if file.endswith('.json'):
            with open(wd + direc + file) as jpdam_file:
                jpdam_data = json.load(jpdam_file)
            for i in range(len(nn_data)):
                ts_data = jpdam_data[str(i)]
                for j in range(num_targs):
                    start = j * y_dim
                    stop = start + y_dim
                    targ_est = np.array(ts_data['post_mean'][start:stop])
                    targ_true = np.array(ts_data['ground_truth'][start:stop])
                    mse = np.linalg.norm(targ_est[0:2] - targ_true[0:2])
                    mse_sd_array_jpdam[j, i] = (mse_array_jpdam[j, i] - mse) ** 2
            file_count += 1

    mse_sd_array_jpdam = np.sqrt(1. / file_count * mse_sd_array_nn)

    fig, ax = plt.subplots()
    fig_height = 4
    fig.set_figheight(fig_height)
    for i in range(num_targs):
        ax.plot(mse_array_nn[i, :],  c=clr_list[i], linestyle='--')
        #plt.errorbar(np.arange(T_sim), mse_array_nn[i, :], yerr=mse_sd_array_nn[i,  :])
        plt.fill_between(np.arange(T_sim), mse_array_nn[i, :] + mse_sd_array_nn[i, :], mse_array_nn[i, :] - mse_sd_array_nn[i, :], alpha=0.3,  color=clr_list[i])
        ax.plot(mse_array_jpdam[i, :], c=clr_list[i])
        plt.fill_between(np.arange(T_sim), mse_array_jpdam[i, :] + mse_sd_array_jpdam[i, :], mse_array_jpdam[i, :] - mse_sd_array_jpdam[i, :], alpha=0.3, color=clr_list[i])
    ax.set_title('4 Targets, Average RMS Error - 100 Monte-Carlo Runs')
    ax.set_ylabel('RMS Error')
    ax.set_xlabel('Tracking Iteration')
    legend_elements = [Line2D([0], [0], color='k', lw=1, ls='-', label='Merged Measurement JPDAF'),
                   Line2D([0], [0], color='k', lw=1, ls='--', label='Nearest Neighbors')]
    ax.legend(handles=legend_elements, loc='upper left')
    plt.tight_layout()
    plt.savefig('4_targ_mse.png', format='png')

    if False:
        # log det sigma plots
        fig, ax = plt.subplots()
        for i in range(num_targs):
            ax.plot(lds_array_nn[i, :], c=clr_list[i],  linestyle='--')
            ax.plot(lds_array_jpdam[i, :], c=clr_list[i])



    # example sim to plot for JPDAM
    seed = "0"
    direc = 'results/sims_icra2021/4targ/JPDAM/'
    file = wd + direc + "seed_" + seed + '.json'
    num_targs = 4
    len_sim = 100
    targ_trajectories_x = np.zeros((num_targs, len_sim))
    targ_trajectories_y = np.zeros((num_targs, len_sim))
    own_states = np.zeros((2, len_sim))
    curr_own_state = [6.7, 7.22, 3.14]  # initial starting state
    with open(file) as f:
        sim_data = json.load(f)
    #create target trajecotries
    for i in range(len(sim_data)):
        ts_data = sim_data[str(i)]
        for k in range(num_targs):
            start = k * y_dim
            stop = start + y_dim
            targ_trajectories_x[k, i] = ts_data['ground_truth'][start:stop][0]
            targ_trajectories_y[k, i] = ts_data['ground_truth'][start:stop][1]
        action = ts_data['planner_output'][0]
        next_own_state = propagateOwnshipEuler(curr_own_state, action[0], action[1], dt=1)
        own_states[0, i] = next_own_state[0]
        own_states[1, i] = next_own_state[1]
        curr_own_state = next_own_state

    # get trajectory from nearest neighbors
    seed = "0"
    direc = 'results/sims_icra2021/4targ/NN/'
    file = wd + direc + "seed_" + seed + '.json'
    with open(file) as f:
        sim_data_nn = json.load(f)
    own_states_nn = np.zeros((2, len_sim))
    curr_own_state_nn = [6.7, 7.22, 3.14]
    for i in range(len(sim_data_nn)):
        ts_data = sim_data_nn[str(i)]
        action = ts_data['planner_output'][0]
        next_own_state_nn = propagateOwnshipEuler(curr_own_state_nn,  action[0], action[1], dt=1)
        own_states_nn[0, i] = next_own_state_nn[0]
        own_states_nn[1, i] = next_own_state_nn[1]
        curr_own_state_nn = next_own_state_nn


    fig, ax = plt.subplots()
    fig.set_figheight(fig_height)
    for i in range(num_targs):
        ax.plot(targ_trajectories_x[i, :],  targ_trajectories_y[i, :], c=clr_list[i], linestyle='-')
    # plot initial position
    for i in range(num_targs):
        ax.plot(targ_trajectories_x[i, 0],targ_trajectories_y[i, 0], c=clr_list[i], marker='.', markersize=20)
    ax.plot(own_states[0, :], own_states[1, :], c='k', label='Algorithm 3')
    ax.plot(6.7, 7.22, c='k', marker='.',  markersize=20)
    ax.plot(own_states_nn[0, :], own_states_nn[1, :], c='k', linestyle='--', label='FVI')
    ax.legend()
    ax.set_title('Planning Trajectory Comparison')
    plt.tight_layout()
    #plt.show()
    plt.savefig('planning_comparison.png', format='png')
    a = 3




