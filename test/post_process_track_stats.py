import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

if False:
    filename_kalman = '../results/videos/planning/JPDAF/test/2_targ_FVI_log_det_kalman_masked_10_speed_5_2_test.npz'
    filename_jpda = '../results/videos/planning/JPDAF/test/2_targ_FVI_log_det_JPDA_masked_10_speed_5_2_test.npz'
    data_kalman = np.load(filename_kalman)
    data_jpda = np.load(filename_jpda)
    mse_kalman = data_kalman['MSE']
    lds_kalman = data_kalman['log_det_Sigma']
    mse_jpda = data_jpda['MSE']
    lds_jpda = data_jpda['log_det_Sigma']


    fig, ax = plt.subplots(2, 1)
    ax[0].plot(mse_kalman, label='kalman')
    ax[0].plot(mse_jpda, label='jpda')
    ax[0].set_title('MSE')

    ax[1].plot(lds_kalman, label='kalman')
    ax[1].plot(lds_jpda, label='jpda')
    ax[1].set_title('log(det($\Sigma$))')
    plt.legend()
    plt.tight_layout()
    plt.show()

file_name_JPDAF = '../results/videos/JPDAF/merged/2_targ/sim1/separated_resolution_10_JPDAF_seed_5.npz'
data = np.load(file_name_JPDAF, allow_pickle=True)
targs = data['targs']
meas = data['meas']
bel_mean = data['beliefs_mean']
bel_cov = data['beliefs_cov']
num_targs = data['num_targs']
n_steps = targs.shape[0]

mse_JPDAF = []
for i in range(n_steps):
    true = targs[i]
    est = bel_mean[i]
    mse_JPDAF.append(np.linalg.norm(true - est))

file_name_merged = '../results/videos/JPDAF/merged/2_targ/sim1/separated_resolution_10_seed_5.npz'
data = np.load(file_name_merged, allow_pickle=True)
targs = data['targs']
meas = data['meas']
bel_mean = data['beliefs_mean']
bel_cov = data['beliefs_cov']
num_targs = data['num_targs']

mse_merged = []
for i in range(n_steps):
    true = targs[i]
    est = bel_mean[i]
    mse_merged.append(np.linalg.norm(true - est))

fig,ax = plt.subplots()

ax.plot(range(n_steps), mse_JPDAF, c='b', label='JPDAF')
ax.plot(range(n_steps), mse_merged, c='r', label='JPDAFM')
ax.set_ylabel('MOSPA (m)')
ax.set_xlabel('Time (sec)')
ax.legend()

plt.show()
print('file loaded')