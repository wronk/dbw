"""
Created on Wed Nov 12 11:21:24 2014

@author: rkp

Auxiliary tools for handling graphs.
"""

import numpy as np
import networkx as nx

def dist_mat(centroids):
    """Compute a distance matrix (returned in mm) from 3D centroids.
    
    Args:
        centroids: 2D array of centroid coordinates.
    Returns:
        distance matrix between all centroids"""
    
    D = np.zeros((centroids.shape[0],centroids.shape[0]),dtype=float)
    for r_idx in range(D.shape[0]):
        for c_idx in range(D.shape[1]):
            d = np.sqrt(np.sum((centroids[r_idx,:] - centroids[c_idx])**2))
            D[r_idx,c_idx] = d

    return D
    
def swap_node_positions(G, D, idx0=None, idx1=None):
    """Swap two nodes' positions and return the resulting graph and distance
    matrix."""
    
    Gswapped = G.copy()
    
    # randomly pick nodes to swap if not provided
    if idx0 is None and idx1 is None:
        idx0, idx1 = np.random.permutation(len(G.nodes()))[:2]
        
    # create new distance matrix
    Dswapped = D.copy()
    # swap rows
    Dswapped[[idx0,idx1],:] = Dswapped[[idx1,idx0],:]
    # swap columns
    Dswapped[:,[idx0,idx1]] = Dswapped[:,[idx1,idx0]]
    
    # make dictionary of edge distances
    dd = {edge:Dswapped[edge] for edge in G.edges()}
    
    # set edge distances in Gswapped
    nx.set_edge_attributes(Gswapped, 'distance', dd)
    
    return Gswapped, Dswapped