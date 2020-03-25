import math
from copy import copy
import numpy as np

def propagateOwnshipEuler(osOld, speed, turnRateCommand, dt):
    osNew = copy(osOld)
    osNew[0] = osOld[0] + speed * np.cos(osOld[2]) * dt
    osNew[1] = osOld[1] + speed * np.sin(osOld[2]) * dt
    osNew[2] = osOld[2] + turnRateCommand * dt

    return osNew

def restrict_angle(phi, min_range=-math.pi, max_range=math.pi):
    x = phi - min_range
    y = max_range - min_range
    return min_range + x - y * math.floor(x / y)
