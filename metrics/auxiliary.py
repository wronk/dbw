"""
Created on Tue Sep  2 15:10:08 2014

@author: rkp

Auxiliary functions used for graph theory metrics.
"""

from copy import deepcopy
import random

import numpy as np
import networkx as nx


def swap_nodes(D, idx0, idx1):
    """Return distance matrix after randomly swapping a pair of nodes.

       Returns:
           swapped distance matrix, indices of swapped nodes"""
    D_swapped = D.copy()

    # Swap rows & columns of distance matrix
    D_swapped[[idx0, idx1], :] = D_swapped[[idx1, idx0], :]
    D_swapped[:, [idx0, idx1]] = D_swapped[:, [idx1, idx0]]

    return D_swapped


def lesion_graph_randomly(graph, phi):
    """
    Randomly remove vertices from a graph according to occupation probability,
    phi.

    Args:
        graph: NetworkX graph to be lesioned.
        phi: Occupation probability (probability of node remaining intact)

    Returns:
        G: NetworkX graph
        A: Adjacency matrix for graph

    """

    # Error checking
    assert phi >= 0. and phi < 1.0, 'phi must be 0.0 <= phi < 1.0'
    assert graph.order > 0, 'Graph is empty'

    # Get list of nodes and probabilty (uniform random) of executing each one
    node_list = graph.nodes()
    execute_prob = random.random(size=(len(node_list),))

    # Identify random nodes to cut
    cut_nodes = [node_list[i] for i in range(len(node_list))
                 if execute_prob[i] > phi]

    G = deepcopy(graph).remove_nodes_from(cut_nodes)

    return G, nx.adjaceny_matrix(G)


def lesion_graph_degree(graph, num):
    """
    Remove vertices from a graph according to degree.

    Args:
        graph: NetworkX graph to be lesioned.
        num: Number of top degree nodes to remove.

    Returns:
        G: NetworkX graph
        A: Adjacency matrix for graph

    """

    # Error checking
    assert num >= 0 and num < graph.order, 'Attempting to remove too many/few\
        nodes'


    #return G, nx.adjaceny_matrix(G)
