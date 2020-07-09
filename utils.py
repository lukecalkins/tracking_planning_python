import math
from copy import copy
import numpy as np


def kron_delta(i, j):

    if i == j:
        return 1
    else:
        return 0


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

def gaussian_pdf(x, mean, cov):
    """
    returns the value of the gaussian
    :param x: value to evaluate the pdf
    :param mean: n x 1 vector
    :param cov: n x n matrix
    :return: scalar pdf value
    """
    z_dim = mean.shape[0]
    diff = x - mean
    inn_prod = diff @ np.linalg.solve(cov, diff)
    num = np.exp(-1./2 * inn_prod)
    den = ((2*np.pi)**z_dim/2)*(np.sqrt(np.linalg.det(cov)))
    likelihood = num/den
    return likelihood

def draw_cov(ax, mean, cov, confidence, clr='r'):
    #manual entries for plotter checking
    #cov = np.array([[225, 0], [0, 225]])
    #mean = np.array([50, 50])

    s = -2 * np.log(1 - confidence)
    w, V = np.linalg.eig(s * cov)
    t = np.linspace(0, 2*np.pi, 100)
    D = np.array([[np.sqrt(w[0]), 0], [0, np.sqrt(w[1])]])

    Q = np.matmul(V, D)
    a = np.matmul(Q, np.array([np.cos(t), np.sin(t)]))
    b = 1
    ax.plot(a[0, :] + mean[0], a[1, :] + mean[1], clr)