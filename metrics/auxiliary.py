"""
Created on Tue Sep  2 15:10:08 2014

@author: rkp

Auxiliary functions used for graph theory metrics.
"""

import numpy as np

def swap_nodes(D,idx0,idx1):
    """Return distance matrix after randomly swapping a pair of nodes.
    
    Returns:
        swapped distance matrix, indices of swapped nodes"""
    D_swapped = D.copy()
    
    # Choose random nodes to swap
    pair = 
    
    # Store labels of swapped nodes
    node_pairs += [(row_labels[p[0]],row_labels[p[1]])]
    # Store centroids of swapped nodes
    centroid_pairs += [(centroids[p[0],:],centroids[p[1],:])]
    # Swap rows & columns of distance matrix
    D_swapped[[p[0],p[1]],:] = D_swapped[[p[1],p[0]],:]
    D_swapped[:,[p[0],p[1]]] = D_swapped[:,[p[1],p[0]]]
        
    return D_swapped,p[0],p[1]