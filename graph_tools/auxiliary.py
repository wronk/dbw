"""
Created on Wed Nov 12 11:21:24 2014

@author: rkp

Auxiliary tools for handling graphs.
"""

import numpy as np

def dist_mat(centroids,mm=True):
    """Compute a distance matrix from 3D centroids.
    
    Args:
        centroids: 2D array of centroid coordinates.
        mm: set to True to return in mm (otherwise returned in 100 micron units)
    Returns:
        distance matrix between all centroids"""
    
    D = np.zeros((centroids.shape[0],centroids.shape[0]),dtype=float)
    for r_idx in range(D.shape[0]):
        for c_idx in range(D.shape[1]):
            d = np.sqrt(np.sum((centroids[r_idx,:] - centroids[c_idx])**2))
            D[r_idx,c_idx] = d
    if mm:
        D /= 10.
    return D