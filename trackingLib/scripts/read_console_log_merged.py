import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
import json
from trackingLib.utils import draw_cov

working_directory = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib'
#extension = '/log/console/console_2meas.log'
extension = '/log/console/console_2meas_merged.log'
with open(working_directory+extension) as f:
    data = json.load(f)

fig, ax = plt.subplots(2,2)
axes = ax.flatten()

fig2, ax2 = plt.subplots(3,2)
axes2 = ax2.flatten()
confidence = 0.99

norm_constant_meas_updated = 0
for i in range(len(data)):  # looping over data stored as graph data association
    gda = data[str(i)]
    meas_updated_probs = gda['meas_updated_probabilities']
    for j in range(len(meas_updated_probs)):
        norm_constant_meas_updated += meas_updated_probs[j]

norm_constant_res_update = 0
for i in range(len(data)):
    gda = data[str(i)]
    res_updated_probs = gda['resolution_updated_probabilities']
    for j in range(len(res_updated_probs)):
        norm_constant_res_update  += res_updated_probs[j]

for ax in axes:
    ax.set_xlim((-5, 5))
    ax.set_ylim((-10, 10))

for ax in axes2:
    ax.set_xlim((-5, 5))
    ax.set_ylim((-10, 10))

for i in range(len(data)):
    gda = data[str(i)]
    associations = gda['associations']
    associations = [np.array(ass) for ass in associations]
    meas_updated_mean = gda['meas_updated_mean']
    meas_updated_mean = [np.array(mean) for mean in meas_updated_mean]
    meas_updated_cov = gda['meas_updated_cov']
    meas_updated_cov = [np.array(cov) for cov in meas_updated_cov]
    meas_updated_probs = gda['meas_updated_probabilities']
    res_updated_mean = gda['resolution_updated_mean']
    res_updated_mean = [np.array(mean) for mean in res_updated_mean]
    res_updated_cov = gda['resolution_updated_cov']
    res_updated_cov = [np.array(cov) for cov in res_updated_cov]
    res_updated_probs = gda['resolution_updated_probabilities']
    for j in range(len(meas_updated_mean)):
        draw_cov(axes[j], meas_updated_mean[j][0:2], meas_updated_cov[j][0:2, 0:2], confidence=confidence, clr='r')
        draw_cov(axes[j], meas_updated_mean[j][4:6], meas_updated_cov[j][4:6, 4:6], confidence=confidence, clr='b')
        axes[j].text(-5, -2, str(meas_updated_probs[j]/norm_constant_meas_updated), fontsize=12)
    plt.tight_layout()
    for j in range(len(res_updated_mean)):
        draw_cov(axes2[j], res_updated_mean[j][0:2], res_updated_cov[j][0:2, 0:2], confidence=confidence, clr='r')
        draw_cov(axes2[j], res_updated_mean[j][4:6], res_updated_cov[j][4:6, 4:6], confidence=confidence, clr='b')
        axes2[j].text(-5, -2, str(res_updated_probs[j]/norm_constant_res_update))
    plt.tight_layout()
    plt.show()
print("number of graph data association = ", len(data))


print("End program")