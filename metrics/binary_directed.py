"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Graph-theory metrics for binary undirected graphs.
"""
from __future__ import division

import numpy as np
import networkx as nx


def reciprocity(G):
    """Calculate the reciprocity coefficient of a directed graph.

    Reciprocity is defined as (Ndirected - Nundirected)/Nundirected"""

    # get number of directed edges
    Ndirected = len(G.edges())

    # get number of undirected edges
    Nundirected = len(G.to_undirected().edges())

    # return reciprocity
    return (Ndirected - Nundirected) / float(Nundirected)


def efficiency_matrix(G):
    """Calculate the efficiency (the inverse of the length of the shortest
    directed path) for all pairs of nodes in a graph.  Rows correspond to
    source nodes, columns to target nodes.

    Parameters
    ----------
    G : networkX graph

    Returns
    -------
    efficiency : ndarray, shape(G.number_of_nodes(), G.number_of_nodes()
        Matrix containing pairwise efficiency calculations"""

    shortest_path_lengths = nx.shortest_path_length(G)

    efficiency = np.zeros((len(G.nodes()), len(G.nodes())), dtype=float)

    for src in shortest_path_lengths.keys():
        for targ in shortest_path_lengths[src].keys():
            if src is not targ:
                if nx.has_path(G, src, targ):
                    efficiency[src, targ] = 1. / shortest_path_lengths[src][targ]
                else:
                    efficiency[src, targ] = 0.

    return efficiency
