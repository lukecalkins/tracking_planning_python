import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
import json
from trackingLib.utils import draw_cov

working_directory = '/Users/william.calkins/Documents/Research/Tracking/python/tracking_lib/trackingLib'
extension = '/log/console/console.log'
with open(working_directory+extension) as f:
    data = json.load(f)

fig, ax = plt.subplots(4,2)
axes = ax.flatten()
confidence = 0.99
for ax in axes:
    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))
for i in range(len(data)):
    gda = data[str(i)]
    associations = gda['associations']
    associations = [np.array(ass) for ass in associations]
    meas_updated_mean = gda['meas_updated_mean']
    meas_updated_mean = [np.array(mean) for mean in meas_updated_mean]
    meas_updated_cov = gda['meas_updated_cov']
    meas_updated_cov = [np.array(cov) for cov in meas_updated_cov]
    meas_updated_probs  = gda['meas_updated_probabilities']
    for j in range(len(meas_updated_mean)):
        draw_cov(axes[j], meas_updated_mean[j][0:2], meas_updated_cov[j][0:2, 0:2], confidence=confidence, clr='r')
        draw_cov(axes[j], meas_updated_mean[j][4:6], meas_updated_cov[j][4:6, 4:6], confidence=confidence, clr='b')
        axes[j].text(-8, -8, str(meas_updated_probs[j]), fontsize=12)
    plt.tight_layout()
    plt.show()
print("number of graph data association = ", len(data))


print("End program")