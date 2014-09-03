import pdb
"""
Created on Tue Sep  2 15:10:08 2014

@author: rkp

Functions for calculating distance & weight dependent cost functions for brain
organization.
"""

import numpy as np
import matplotlib.pyplot as plt

import network_gen
import area_compute


def dist_mat(centroids):
    """Compute a distance matrix from 3D centroids."""
    
    D = np.zeros(W.shape,dtype=float)
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
    
    if cost_type=='dist':
        return (D*W).sum()
    elif cost_type=='dist2':
        return ((D**2)*W).sum()
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
    

if __name__ == '__main__':
    # Parameters
    sym = False
    n_swaps = 10
    cost_type = 'dist'
    
    W,row_labels,col_labels = network_gen.quick_net()
    centroids = area_compute.get_centroids(row_labels)
    
    D0 = dist_mat(centroids)
    c0 = cost(D0,W,cost_type=cost_type)
    
    # Iterate over random permutations of pairs of nodes (not symmetrically)
    n_permutations = 5000
    c = np.zeros((n_permutations,))
    node_pairs = [None for p_idx in range(n_permutations)]
    centroid_pairs = [None for p_idx in range(n_permutations)]
    for p_idx in range(n_permutations):
        if not (p_idx+1)%50:
            print 'Permutation #%d'%(p_idx+1)
        D_swapped, node_pair, centroid_pair = \
            swap_nodes(D0,row_labels,centroids,n_swaps=n_swaps,sym=sym)
        node_pairs[p_idx] = node_pair
        centroid_pairs[p_idx] = centroid_pair
        c[p_idx] = cost(D_swapped,W,cost_type=cost_type)
    
    fig,ax = plt.subplots(1,1,facecolor='w')
    ax.scatter(np.arange(n_permutations),c,c='r')
    ax.plot(np.arange(n_permutations),c0*np.ones((n_permutations,)),c='b',lw=3)
    ax.set_xlabel('Permutation #')
    ax.set_ylabel('Cost')
    p_value = ((c<c0).sum()/float(n_permutations))
    if sym:
        ax.set_title('%d symmetric swaps, P = %.3f'%(n_swaps,p_value))
    else:
        ax.set_title('%d swaps, P = %.3f'%(n_swaps,p_value))