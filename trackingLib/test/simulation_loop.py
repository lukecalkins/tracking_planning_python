import numpy as np
import sys, os
import trackingLib.test.run_sim as simulate

print(sys.path)


seeds = np.arange(1, 100)

for seed in seeds:
    string_seed = str(seed)
    try:
        json_data = os.system("python3 run_sim.py " + string_seed)
    except:
        pass

