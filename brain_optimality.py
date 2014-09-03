import pdb
"""
Created on Tue Sep  2 15:10:08 2014

@author: rkp

Functions for calculating distance & weight dependent cost functions for brain
organization.
"""

import numpy as np
import networkx as nx
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
    

if __name__ == '__main__':
    # Parameters
    sym = False
    n_swaps = 1
    n_permutations = 5000
    cost_type = 'dist'
    print_every = 500
    
    W,row_labels,col_labels = network_gen.quick_net()
    centroids = area_compute.get_centroids(row_labels)
    
    # Loop over number of swaps
    n_swaps_vec = np.arange(1,10)
    p = 1
    plot_idxs = np.array([0,1,2,3,4,5,6,7,8,9])
    p_values = np.ones((len(n_swaps_vec),))
    
    for ns_idx,n_swaps in enumerate(n_swaps_vec):
        
        print 'n_swaps = %.1f'%n_swaps
        
        W1 = np.zeros(W.shape,dtype=float)
        for from_idx in range(W.shape[0]):
            num_cxns = int(p*(W[from_idx,:]>0).sum())
            keep_idxs = np.argsort(W[from_idx,:])[::-1][:num_cxns]
            W1[keep_idxs,:] = W[keep_idxs,:]
        
        D0 = dist_mat(centroids)
        c0 = cost(D0,W1,cost_type=cost_type)
        
        # Iterate over random permutations of pairs of nodes (not symmetrically)
        c = np.zeros((n_permutations,))
        node_pairs = [None for p_idx in range(n_permutations)]
        centroid_pairs = [None for p_idx in range(n_permutations)]
        for p_idx in range(n_permutations):
            if not (p_idx+1)%print_every:
                print 'Permutation #%d'%(p_idx+1)
            D_swapped, node_pair, centroid_pair = \
                swap_nodes(D0,row_labels,centroids,n_swaps=n_swaps,sym=sym)
            node_pairs[p_idx] = node_pair
            centroid_pairs[p_idx] = centroid_pair
            c[p_idx] = cost(D_swapped,W1,cost_type=cost_type)
        
        if ns_idx in plot_idxs:
            fig,ax = plt.subplots(1,1,facecolor='w')
            ax.scatter(np.arange(n_permutations),c,c='r')
            ax.plot(np.arange(n_permutations),c0*np.ones((n_permutations,)),c='b',lw=4)
            ax.set_xlabel('Permutation #')
            ax.set_ylabel('Cost')
            p_value = ((c<c0).sum()/float(n_permutations))
            if sym:
                ax.set_title('%d symmetric swaps, P = %.3f, p = %.1f'%(n_swaps,p_value,p))
            else:
                ax.set_title('%d swaps, P = %.3f, p = %.1f'%(n_swaps,p_value,p))
            plt.draw()
        p_values[ns_idx] = p_value
         
    fig,ax = plt.subplots(1,1,facecolor='w')
    ax.plot(n_swaps_vec,p_values,lw=2)
    ax.set_xlabel('n_swaps')
    ax.set_ylabel('p_value')
    
    plt.draw()
    
    # Run for all_paths cost function
    # Parameters
    sym = False
    n_swaps = 1
    n_permutations = 500
    cost_type = 'all_paths'
    
    W,row_labels,col_labels = network_gen.quick_net()
    centroids = area_compute.get_centroids(row_labels)
    
    D0 = dist_mat(centroids)
    c0 = cost(D0,W1,cost_type=cost_type)
    
    # Iterate over random permutations of pairs of nodes (not symmetrically)
    c = np.zeros((n_permutations,))
    node_pairs = [None for p_idx in range(n_permutations)]
    centroid_pairs = [None for p_idx in range(n_permutations)]
    for p_idx in range(n_permutations):
        print 'Permutation #%d'%(p_idx+1)
        D_swapped, node_pair, centroid_pair = \
            swap_nodes(D0,row_labels,centroids,n_swaps=n_swaps,sym=sym)
        node_pairs[p_idx] = node_pair
        centroid_pairs[p_idx] = centroid_pair
        c[p_idx] = cost(D_swapped,W1,cost_type=cost_type)
    
    fig,ax = plt.subplots(1,1,facecolor='w')
    ax.scatter(np.arange(n_permutations),c,c='r')
    ax.plot(np.arange(n_permutations),c0*np.ones((n_permutations,)),c='b',lw=4)
    ax.set_xlabel('Permutation #')
    ax.set_ylabel('Cost')
    p_value = ((c<c0).sum()/float(n_permutations))
    if sym:
        ax.set_title('%d symmetric swaps, P = %.3f, p = %.1f'%(n_swaps,p_value,p))
    else:
        ax.set_title('%d swaps, P = %.3f, p = %.1f'%(n_swaps,p_value,p))
    plt.draw()