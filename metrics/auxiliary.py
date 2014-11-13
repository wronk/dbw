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
    
    # Swap rows & columns of distance matrix
    D_swapped[[idx0,idx1],:] = D_swapped[[idx1,idx0],:]
    D_swapped[:,[idx0,idx1]] = D_swapped[:,[idx1,idx0]]
        
    return D_swapped