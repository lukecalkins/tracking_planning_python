import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

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