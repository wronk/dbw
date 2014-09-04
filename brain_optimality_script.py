import pdb
"""
Created on Wed Sep  3 16:27:45 2014

@author: rkp
"""

import numpy as np
import matplotlib.pyplot as plt; plt.close('all')

import network_gen
import area_compute
import brain_optimality as bropt

CALC_RANDS = False
SHOW_WVD = False
LOOP_OVER_NSWAPS = True
ALL_PATHS_COST = False

# Parameters
sym = False
n_swaps = 1
n_permutations = 5000
cost_type = 'dist'
print_every = 500
y_lim = [84000,95000]

W,row_labels,col_labels = network_gen.quick_net()
centroids = area_compute.get_centroids(row_labels)
D = bropt.dist_mat(centroids)

if SHOW_WVD:
    # Show correlation between distance and log weight
    W_vec = W.flatten()
    D_vec = D.flatten()
    W_vec_nz = W_vec[W_vec>0]
    D_vec_nz = D_vec[W_vec>0]
    fig, ax = plt.subplots(1,1,facecolor='white')
    ax.scatter(D_vec_nz,np.log(W_vec_nz))
    ax.set_xlabel('Distance (100 um)')
    ax.set_ylabel('Log[Weight]')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
    plt.draw()

if CALC_RANDS:
    # Create 3 completely random permutations
    num_rand_perm = 3
    rand_c = np.zeros((num_rand_perm,),dtype=float)
    for rp_idx in range(num_rand_perm):
        # Make swapped distance matrix
        D_swapped,_,_ = bropt.swap_nodes(D,row_labels,centroids,
                                         n_swaps=D.shape[0])
        rand_c[rp_idx] = bropt.cost(D_swapped,W,cost_type=cost_type)
    print 'Costs of randomly shuffled networks'
    print rand_c

if LOOP_OVER_NSWAPS:
    # Loop over number of swaps
    n_swaps_vec = np.arange(1,11)
    plot_idxs = np.array([0,1,4,9])
    p_values = np.ones((len(n_swaps_vec),))
    
    D0 = bropt.dist_mat(centroids)
    c0 = bropt.cost(D0,W,cost_type=cost_type)
        
    for ns_idx,n_swaps in enumerate(n_swaps_vec):
        
        print 'n_swaps = %.1f'%n_swaps
        
        # Iterate over random permutations of pairs of nodes (not symmetrically)
        c = np.zeros((n_permutations,))
        node_pairs = [None for p_idx in range(n_permutations)]
        centroid_pairs = [None for p_idx in range(n_permutations)]
        for p_idx in range(n_permutations):
            if not (p_idx+1)%print_every:
                print 'Permutation #%d'%(p_idx+1)
            D_swapped, node_pair, centroid_pair = \
                bropt.swap_nodes(D0,row_labels,centroids,n_swaps=n_swaps,sym=sym)
            node_pairs[p_idx] = node_pair
            centroid_pairs[p_idx] = centroid_pair
            c[p_idx] = bropt.cost(D_swapped,W,cost_type=cost_type)
        
        p_value = ((c<=c0).sum()/float(n_permutations))
        if ns_idx in plot_idxs:
            fig,ax = plt.subplots(1,1,facecolor='w')
            ax.scatter(np.arange(n_permutations),c,c='r')
            ax.plot(np.arange(n_permutations),c0*np.ones((n_permutations,)),c='b',lw=5)
            ax.set_xlim(0,n_permutations)
            ax.set_ylim(y_lim[0],y_lim[1])
            ax.set_xlabel('Permutation #')
            ax.set_ylabel('Cost')
            if sym:
                ax.set_title('%d symmetric swaps, P = %.3f'%(n_swaps,p_value))
            else:
                ax.set_title('%d swaps, P = %.3f'%(n_swaps,p_value))
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(16)
            plt.draw()
        p_values[ns_idx] = p_value
         
    fig,ax = plt.subplots(1,1,facecolor='w')
    ax.plot(n_swaps_vec,p_values,lw=2)
    ax.set_xlabel('n_swaps')
    ax.set_ylabel('P')
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(16)
    
    plt.draw()

if ALL_PATHS_COST:
    # Run for all_paths cost function
    # Parameters
    sym = False
    n_swaps = 1
    n_permutations = 5000
    cost_type = 'all_paths'
    
    W,row_labels,col_labels = network_gen.quick_net()
    centroids = area_compute.get_centroids(row_labels)
    
    D0 = bropt.dist_mat(centroids)
    c0 = bropt.cost(D0,W,cost_type=cost_type)
    
    # Iterate over random permutations of pairs of nodes (not symmetrically)
    c = np.zeros((n_permutations,))
    node_pairs = [None for p_idx in range(n_permutations)]
    centroid_pairs = [None for p_idx in range(n_permutations)]
    for p_idx in range(n_permutations):
        print 'Permutation #%d'%(p_idx+1)
        D_swapped, node_pair, centroid_pair = \
            bropt.swap_nodes(D0,row_labels,centroids,n_swaps=n_swaps,sym=sym)
        node_pairs[p_idx] = node_pair
        centroid_pairs[p_idx] = centroid_pair
        c[p_idx] = bropt.cost(D_swapped,W,cost_type=cost_type)
    
    fig,ax = plt.subplots(1,1,facecolor='w')
    ax.scatter(np.arange(n_permutations),c,c='r')
    ax.plot(np.arange(n_permutations),c0*np.ones((n_permutations,)),c='b',lw=4)
    ax.set_xlabel('Permutation #')
    ax.set_ylabel('Cost')
    p_value = ((c<c0).sum()/float(n_permutations))
    if sym:
        ax.set_title('%d symmetric swaps, P = %.3f'%(n_swaps,p_value))
    else:
        ax.set_title('%d swaps, P = %.3f'%(n_swaps,p_value))
    plt.draw()