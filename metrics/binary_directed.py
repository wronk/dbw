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
    efficiencty : ndarray, shape(G.number_of_nodes(), G.number_of_nodes()
        Matrix containing pairwise efficiency calculations"""

    shortest_path_lengths = nx.shortest_path_length(G)

    # Initialize to nan so that errors are easier to spot
    efficiency = np.empty((len(G.number_of_nodes()), len(G.number_of_nodes())))
    efficiency[:] = np.nan

    for src in shortest_path_lengths.keys():
        for targ in shortest_path_lengths[src].keys():
            if src != targ:
                efficiency[src, targ] = 1. / shortest_path_lengths[src][targ]

    # Error check for problems in calculation
    assert np.is_nan(efficiency).any() is False, 'NaN in efficiency matrix'

    return efficiency
