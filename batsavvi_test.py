import matplotlib.pyplot as plt
import math
import numpy as np

t = np.linspace(0, 4*np.pi, 100)

y = 0.5 * np.cos(t)**2 + np.sin(t)**2

fig, ax = plt.subplots()
ax.plot(t,y)
plt.show()