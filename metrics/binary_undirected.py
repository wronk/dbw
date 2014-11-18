"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Graph-theory metrics for binary undirected graphs.
"""

import numpy as np
import scipy.stats as stats

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
    
    L = -1./slope
    
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
    cxn_prob = dist_cxn_cts/(dist_cts.astype(float))
    return cxn_prob, dist_bins


def wiring_distance_cost(A, D):
    """Calculate wiring distance cost (sum of edge distances)
    
    Args:
        A: adjacency matrix
        D: distance matrix
    Returns:
        scalar cost"""
    # Make sure self-connections are set to zero
    np.fill_diagonal(A,0)
    # Calculate cost by summing weights with distances
    return (np.triu(A)*D).sum()
    
    
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