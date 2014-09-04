import pdb
"""
Created on Tue Sep  2 15:10:08 2014

@author: rkp

Functions for calculating distance & weight dependent cost functions for brain
organization.
"""

import numpy as np
import networkx as nx


def dist_mat(centroids):
    """Compute a distance matrix from 3D centroids."""
    
    D = np.zeros((centroids.shape[0],centroids.shape[0]),dtype=float)
    for r_idx in range(D.shape[0]):
        for c_idx in range(D.shape[1]):
            d = np.sqrt(np.sum((centroids[r_idx,:] - centroids[c_idx])**2))
            D[r_idx,c_idx] = d
    return D
    
def cost(D,W,cost_type='dist'):
    """Calculate the "cost" of a network based on the locations of its nodes
    and the connection weights between them.
    
    Args:
        D: N x N symmetric distance matrix for N nodes
        W: N x N connection weight matrix."""
    
    if cost_type == 'dist':
        return (D*W).sum()
    elif cost_type == 'all_paths':
        # Build directed graph
        DG = nx.DiGraph()
        # Add edges
        for from_idx in range(D.shape[0]):
            for to_idx in range(D.shape[1]):
                w = W[from_idx,to_idx]
                if w > 0:
                    d = D[from_idx,to_idx]
                    DG.add_edge(from_idx,to_idx,weight=w,dist=d)
        # Calculate shortest paths
        SPs = nx.shortest_path(DG,weight='dist')
        # Calculate sum of witt metric over all pairs of nodes
        witt = 0
        for from_idx in SPs.keys():
            for to_idx in SPs[from_idx].keys():
                # Get this shortest path
                SP = SPs[from_idx][to_idx]
                if len(SP) > 1:
                    # Loop over all edges in shortest path
                    for idx in range(len(SP)-1):
                        n0 = SP[idx]
                        n1 = SP[idx+1]
                        witt += D[n0,n1]*W[n0,n1]
        return witt
    else:
        return None
    
def swap_nodes(D0,row_labels,centroids,n_swaps=1,sym=False):
    """Randomly swap a pair of nodes.
    
    Returns swapped distance matrix, labels of swapped nodes, & centroids of
    swapped nodes."""
    D_swapped = D0.copy()
    node_pairs = []
    centroid_pairs = []
    swap_pairs = []
    for pair_idx in range(n_swaps):
        pair = tuple(np.random.permutation(len(row_labels))[:2])
        swap_pairs += [pair]
        if sym:
            # Make symmetric pair
            sym_pair = ((pair[0]+len(row_labels)/2)%len(row_labels),
                        (pair[1]+len(row_labels)/2)%len(row_labels))
            swap_pairs += [sym_pair]
    
    for p in swap_pairs:
        # Store labels of swapped nodes
        node_pairs += [(row_labels[p[0]],row_labels[p[1]])]
        # Store centroids of swapped nodes
        centroid_pairs += [(centroids[p[0],:],centroids[p[1],:])]
        # Swap rows & columns of distance matrix
        D_swapped[[p[0],p[1]],:] = D_swapped[[p[1],p[0]],:]
        D_swapped[:,[p[0],p[1]]] = D_swapped[:,[p[1],p[0]]]
        
    return D_swapped,node_pairs,centroid_pairs