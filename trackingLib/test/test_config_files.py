import numpy as np

old = np.load('results/videos/planning/JPDAF/4_targ/test_old.npz')
mse_old = old['MSE']
lds_old = old['log_det_Sigma']
new = np.load('results/videos/planning/JPDAF/4_targ/test_new.npz')
mse_new = new['MSE']
lds_new = new['log_det_Sigma']

a = 3