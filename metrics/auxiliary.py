"""
Created on Tue Sep  2 15:10:08 2014

@author: rkp

Auxiliary functions used for graph theory metrics.
"""

from copy import deepcopy

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
    G = deepcopy(graph)
    if phi == 1.:
        return G, nx.adjacency_matrix(G)

    # Error checking
    assert phi > 0. and phi <= 1.0, 'phi must be 0.0 <= phi < 1.0'
    assert graph.order > 0, 'Graph is empty'

    # Get list of nodes and probabilty (uniform random) of executing each one
    node_list = graph.nodes()
    execute_prob = np.random.random((len(node_list),))

    # Identify random nodes to cut
    cut_nodes = [node_list[i] for i in range(len(node_list))
                 if execute_prob[i] > phi]

    G.remove_nodes_from(cut_nodes)

    if G.order() > 0:
        return G, nx.adjacency_matrix(G)
    else:
        print 'Graph completely lesioned.'
        return None, None


def lesion_graph_degree(graph, num_lesions):
    """
    Remove vertices from a graph according to degree.

    Args:
        graph: NetworkX graph to be lesioned.
        num_lesions: Number of top degree nodes to remove.

    Returns:
        G: NetworkX graph
        A: Adjacency matrix for graph

    """
    # Error checking
    G = deepcopy(graph)
    if num_lesions == 0:
        return G, nx.adjacency_matrix(G)

    assert num_lesions >= 0 and num_lesions < graph.order, 'Attempting to\
        remove too many/few nodes'

    for l in range(num_lesions):
        # Identify node to cut
        node_i, node_d = max(G.degree().items(),
                             key=lambda degree: degree[1])
        G.remove_node(node_i)
        print (node_i, node_d)

    if G.order() > 0:
        return G, nx.adjacency_matrix(G)
    else:
        print 'Graph completely lesioned.'
        return None, None


    #return G, nx.adjacency_matrix(G)
