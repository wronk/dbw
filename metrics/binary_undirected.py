"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Graph-theory metrics for binary undirected graphs.
"""

import numpy as np
import auxiliary as aux

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