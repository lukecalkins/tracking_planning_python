import math


def restrict_angle(phi, min_range=-math.pi, max_range=math.pi):
    x = phi - min_range
    y = max_range - min_range
    return min_range + x - y * math.floor(x / y)
