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
    return (A/2.*D).sum()