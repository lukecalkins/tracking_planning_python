from utils import restrict_angle
import numpy as np
import dataAssociation as DA
from params import Parameters
import matplotlib.pyplot as plt

pi = np.pi

b0 = 11 * np.pi/12
b1 = -11 * np.pi/12

print(restrict_angle(b1-b0), " radians, ", restrict_angle(b1-b0)*180/np.pi, " degrees")
print(restrict_angle(b0-b1), " radians, ", restrict_angle(b0-b1)*180/np.pi, " degrees")

if False:
    angles = np.linspace(-30, 30)
    angles_radians = angles * pi/180
    bearing_res = 0.3491
    R_u = 1./(np.sqrt(2*np.log(2))) * bearing_res**2
    P_u = np.exp(-angles_radians * 1./R_u * angles_radians)
    fig, ax = plt.subplots()
    ax.plot(angles, P_u)
    plt.show()

yaml_file = 'config/init_info_planner.yaml'
p = Parameters(yaml_file)
robots = p.getRobots()
planner = p.getPlanner()
JPDAF = p.getEstimator()
target_model = p.getWorld()

gate_level = 0.99
JPDAF_merged = DA.JPDAFMerged(robots[0].sensor, p.unresolved_resolution, p.clutter_density, gate_level)

np.random.seed(42)
measurements = robots[0].sensor.senseTargets(robots[0].getState(), target_model.getTargets())
JPDAF_merged.filter(measurements, robots[0])
a = 0
