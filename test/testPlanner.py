from planner import *
import dataAssociation as DA
from cost_function import GateOverlapCost
import numpy as np

#actions
turn_radius = 50.
speed = 5.
actions = [[speed, 0], [speed, speed/turn_radius], [speed, -speed/turn_radius]]

gate_cost = GateOverlapCost(y_dim=4, level=0.99)
planner = Planner(actions, gate_cost)
plan_horizon = 10

x0 = [0, 0, np.pi/2]
Sigma0 = np.eye(2)
S0 = SearchState(x0, Sigma0, dt=1)
root_node = SearchNode(S0)
root_node.make_children(actions)
a = 3