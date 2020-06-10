import networkx as nx
from itertools import combinations


def rSubset(arr, r):
    # return list of all subsets of length r
    # to deal with duplicate subsets use
    # set(list(combinations(arr, r)))
    return list(combinations(arr, r))
"""
G_roc = nx.generators.community.ring_of_cliques(4, 2)
G_cave = nx.generators.community.caveman_graph(4, 2)
G_cave = nx.generators.community.relaxed_caveman_graph()
print(G_roc.edges)
print(G_cave.edges)
"""


# Can iterate graphs by using
arr = [(0,1), (1,2), (2,3), (3,0)]
r = 1
graphs = rSubset(arr, r)
print("Number of graphs", len(graphs))
print("Edges", graphs)
a=0