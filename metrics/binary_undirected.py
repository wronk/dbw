"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp, wronk

Graph-theory metrics for binary undirected graphs.
"""

import numpy as np
import scipy.stats as stats
import networkx as nx
from itertools import combinations

import auxiliary as aux


def cxn_length_scale(A, D, bins=50, no_self_cxns=True):
    """Calculate the length scale of a set of connections.

    Args:
        A: adjacency matrix
        D: distance matrix
        bins: how many bins to use to calculate distance histogram
        no_self_cxns: whether or not to include self connections
    Returns:
        Length scale of best fit exponential.
    """
    # Add/remove self-connections
    A_temp = A.copy()
    if no_self_cxns:
        np.fill_diagonal(A_temp, 0)
    else:
        np.fill_diagonal(A_temp, 1)
    # Get vector of connection distances
    D_vec = D[np.triu(A) > 0]
    # Calculate histogram
    prob, bins = np.histogram(D_vec, bins=bins, normed=True)
    bin_centers = .5 * (bins[:-1] + bins[1:])
    log_prob = np.log(prob)
    # Remove -infs
    bin_centers = bin_centers[~np.isinf(log_prob)]
    log_prob = log_prob[~np.isinf(log_prob)]
    # Fit line
    slope, b, r, p, stderr = stats.linregress(bin_centers, log_prob)

    L = -1. / slope

    return L, r, p


def cxn_probability(A, D, bins=50):
    """Calculate the probability of a connection between two nodes given their
    distance

    Args:
        A: adjacency matrix
        D: distance matrix
        bins: distance bins
    Returns:
        probabilities of cxns, distance bins
    """
    # Get vector of distances (without double counting)
    D_vec = D[np.triu(np.ones(D.shape), k=1) == 1]
    # Get distance counts & bins regardless of whether cxn present or not
    dist_cts, dist_bins = np.histogram(D_vec, bins)
    # Get vector of distances only when connections present
    D_cxn_vec = D[np.triu(A, k=1) == 1]
    # Get distance counts using the same bins, but only when cxn present
    dist_cxn_cts, dist_bins = np.histogram(D_cxn_vec, bins)
    # Calculate probability of cxn given distance bin
    cxn_prob = dist_cxn_cts / (dist_cts.astype(float))
    return cxn_prob, dist_bins


def wiring_distance_cost(A, D):
    """Calculate wiring distance cost (sum of edge distances)

    Args:
        A: adjacency matrix
        D: distance matrix
    Returns:
        scalar cost"""
    # Make sure self-connections are set to zero
    np.fill_diagonal(A, 0)
    # Calculate cost by summing weights with distances
    return (np.triu(A) * D).sum()


def swapped_cost_distr(A, D, n_trials=500, percent_change=True):
    """Calculate how much the wiring distance cost changes for a random node
    swap.

    Args:
        A: adjacency matrix
        D: distance matrix
        n_trials: how many times to make a random swap (starting from the
            original configuration)
        percent_change: set to True to return percent changes in cost

    Returns:
        vector of cost changes for random swaps"""
    # Calculate true cost
    true_cost = wiring_distance_cost(A, D)
    # Allocate space for storing amount by which cost changes
    cost_changes = np.zeros((n_trials,), dtype=float)

    # Perform random swaps
    for trial in range(n_trials):
        print 'trial %d' % trial
        # Randomly select two nodes
        idx0, idx1 = np.random.permutation(D.shape[0])[:2]
        # Create new distance matrix
        D_swapped = aux.swap_nodes(D, idx0, idx1)
        # Calculate cost change for swapped-node graph
        cost_changes[trial] = wiring_distance_cost(A, D_swapped) - true_cost

    if percent_change:
        cost_changes /= true_cost
        cost_changes *= 100

    return cost_changes


def efficiency(G, n1, n2):
    """Returns the efficiency of a pair of nodes in a graph.

    Efficiency of a pair of nodes is the inverse of the shortest path each
    node.

    Parameters
    ----------
    G : graph
        An undirected graph to compute the average local efficiency of
    n1, n2: node
        Nodes in the graph G

    Returns
    -------
    float
        efficiency between the node u and v"""

    assert G.is_directed is False, 'Graph can\'t be directed'

    return 1. / nx.shortest_path_length(G, n1, n2)


def local_efficiency(G):
    """Calculate local efficiency for a graph. Local efficiency is the
    average efficiency of all neighbors of the given node.

    Parameters
    ----------
    G : networkX graph

    Returns
    -------
    float : avg local efficiency of the graph
    """

    assert G.is_directed is False, 'Graph can\'t be directed'

    return sum(global_efficiency(G.subgraph[v]) for v in G) / \
        np.float(G.order())


def global_efficiency(G):
    """Calculate global efficiency for a graph. Global efficiency is the
    average efficiency of all neighbors of the given node.

    Parameters
    ----------
    G : networkX graph

    Returns
    -------
    float : avg global efficiency of the graph
    """

    assert G.is_directed is False, 'Graph can\'t be directed'

    n_nodes = G.number_of_nodes()
    den = np.float(n_nodes * (n_nodes - 1))

    return sum(efficiency(G, n1, n2) for n1, n2 in combinations(G, 2)) / den
