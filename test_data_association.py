import dataAssociation as DA
import numpy as np

Omega = np.array([[1, 1, 0], [1, 1, 1], [1, 0, 1]])

#Omega = np.array([[1, 0, 1],[1, 1, 1], [1, 1, 0],[1, 1, 0]])

#Omega = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])

detection_prob = 1.0
filter = DA.JPDAF(detection_prob)

events = filter.generate_association_events(Omega)

i = 0
for event in events:
    print(i)
    print(event)
    i += 1

print("Num valid events: " + str(len(events)))