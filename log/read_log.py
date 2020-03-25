import numpy as np
import matplotlib.pyplot as plt
from plotting import draw_cov

f = open("plan.log", 'r')
lines = f.readlines()
n_Sigma = 8
n_mat = int(len(lines)/n_Sigma)
Sigma_list = []
for i in range(n_mat):
    Sigma = []
    for j in range(n_Sigma):
        Sigma_line = [float(digit) for digit in lines[i*n_Sigma + j].split()]
        Sigma.append(Sigma_line)
    Sigma_list.append(np.array(Sigma))
f.close()

#parse target log
f = open("targ.log", 'r')
lines = f.readlines()
y_list = []
for i in range(n_mat):
    y = []
    for j in range(n_Sigma):
        y.append(float(lines[i*n_Sigma + j]))
    y_list.append(np.array(y))
f.close()

fig, ax = plt.subplots()
for i in range(len(Sigma_list)):
    ax.plot(y_list[i][0], y_list[i][1], 'b.')
    ax.plot(y_list[i][4], y_list[i][5], 'r.')
    mean = y_list[i][0:2]
    print("log det Sigma: ", np.log(np.linalg.det(Sigma_list[i])))
    Sigma = Sigma_list[i][0:2, 0:2]
    draw_cov(mean, Sigma, confidence=0.99, ax=ax, clr='b')
    mean = y_list[i][4:6]
    Sigma = Sigma_list[i][4:6, 4:6]
    draw_cov(mean, Sigma, confidence=0.99, ax=ax, clr='r')

plt.show()