from scipy.linalg import sqrtm
import numpy as np
from utils import restrict_angle, kron_delta
import itertools as IT

class Graph:

    def __init__(self, n_vertices, edges, feasible_edges):
        self.n_vertices = n_vertices
        self.edges = edges
        self.feasible_edges = feasible_edges
        self.adjacency = self.build_adjacency()
        self.non_edges = [edge for edge in feasible_edges if edge not in edges]
        self.edge_multipliers = []
        self.edge_multipliers_sign = []
        self.resolution_update_D_matrices = []


    def build_resolution_update_D_matrices(self):
        """
        function that takes each product of P_u terms in the resolution update and contructs the associated D matrix
        that is used in the single step resolution update for that term.
        :return:
        """
        for edge_set in self.edge_multipliers:
            G = np.zeros((self.n_vertices, self.n_vertices))
            for edge in edge_set:
                pi_i_j = get_pi_i_j(edge, self.n_vertices)
                G = G + pi_i_j @ pi_i_j.transpose()
            D = sqrtm(G)
            self.resolution_update_D_matrices.append(np.real(D))

    def build_resolution_update_multipliers(self):

        num_resolved = len(self.non_edges)
        num_unresolved = len(self.edges)
        edge_multipliers = []
        edge_multipliers_sign = []
        for i in range(num_resolved + 1):
            edge_list = list(IT.combinations(self.non_edges, i))
            for edge_set in edge_list:
                edge_multipliers.append(list(edge_set))
                if len(edge_set) % 2 == 0:
                    edge_multipliers_sign.append(1)
                else:
                    edge_multipliers_sign.append(-1)
        if num_unresolved > 0:
            for edge_set in edge_multipliers:
                for connected_edge in self.edges:
                    edge_set.append(connected_edge)

        self.edge_multipliers = edge_multipliers
        self.edge_multipliers_sign = edge_multipliers_sign

    def build_adjacency(self):
        adjacency = np.zeros((self.n_vertices, self.n_vertices))
        for edge in self.edges:
            adjacency[edge[0], edge[1]] = 1
            adjacency[edge[1], edge[0]] = 1
        return adjacency

    def is_connected(self, targ_index):
        """
        returns true if target is connected to any other target in the graph
        :param targ_index:
        :return:
        """
        if self.adjacency[targ_index, :].sum() > 0:
            return True
        else:
            return False

    def get_connected_targets(self, targ_index):
        """
        function that takes a target index (starting at zero and produces a list of target indices that are connected
        to it
        :param targ_index:
        :return:
        """
        row = self.adjacency[targ_index, :]
        visited = {targ_index}
        connected = {targ_index}
        for connected_targ in list(row.nonzero()[0]):  # return as a tuple of length 1
            connected.add(int(connected_targ))

        while visited != connected:
            for targ in connected:
                if targ not in visited:
                    row = self.adjacency[targ, :]
                    for connected_targ in list(row.nonzero()[0]):
                        if connected_targ not in connected:
                            connected.add(connected_targ)
                    visited.add(targ)
                    break

        # only want to return connected indices, not the one target itself
        connected.remove(targ_index)

        connected_list = list(connected)
        correct_target_index_for_Omega = [targ + 1 for targ in connected_list]

        return correct_target_index_for_Omega

    def get_connected_targets_raw_index(self, targ_index):
        """
        function that takes a target index (starting at zero and produces a list of target indices that are connected
        to it
        :param targ_index:
        :return:
        """
        row = self.adjacency[targ_index, :]
        visited = {targ_index}
        connected = {targ_index}
        for connected_targ in list(row.nonzero()[0]):  # return as a tuple of length 1
            connected.add(int(connected_targ))

        while visited != connected:
            for targ in connected:
                if targ not in visited:
                    row = self.adjacency[targ, :]
                    for connected_targ in list(row.nonzero()[0]):
                        if connected_targ not in connected:
                            connected.add(connected_targ)
                    visited.add(targ)
                    break

        # only want to return connected indices, not the one target itself
        connected.remove(targ_index)

        connected_list = list(connected)
        correct_target_index_raw = connected_list

        return correct_target_index_raw

##############################################
############## End Graph #####################
##############################################

def get_pi_i_j(edge_pair, n):
    """
    returns pi_i_j vector from svennson2012multitarget
    :param edge_pair:
    :param n:
    :return:
    """
    pi_i_j = np.zeros((n, 1))
    for i in range(n):
        pi_i_j[i] = kron_delta(i, edge_pair[0]) - kron_delta(i, edge_pair[1])

    return pi_i_j