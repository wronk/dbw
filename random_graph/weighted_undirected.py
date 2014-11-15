import pdb
"""
Created on Wed Nov 12 11:18:14 2014

@author: rkp

Functions for generating random weighted undirected graphs not included in 
NetworkX.
"""

import numpy as np
import networkx as nx
import binary_undirected
import extract.brain_graph
import metrics.weighted_undirected

def biophysical_sample_weights(N=426, N_edges=7804, L=1., gamma=1.5, 
                               brain_size=[10.,10,10], d_bins=None, 
                               w_dists=None, use_brain_weights=True):
    """Create a biophysically inspired graph. Connection probabilities depend
    on distance & degree.
    
    Args:
        N: how many nodes
        N_edges: how many edges
        L: length constant
        gamma: power to raise degree to
        brain_size: size of space in which nodes are randomly placed
        d_bins: distance bin edges for distance-dependent weight distr's
        weight_dists: weight distributions, one for each distance bin
        use_brain_weights: whether to calculate distance-dependent weight 
            distributions from mouse connectivity data
    Returns:
        Networkx graph object, weight matrix, distance matrix"""
    # Create binary undirected graph
    G,A,D = binary_undirected.biophysical(N=N, N_edges=N_edges, L=L,
                                          gamma=gamma, brain_size=brain_size)
    
    # Calculate distance-dependent weight distributions if necessary
    if use_brain_weights:
        # Load weight & distance matrices from mouse connectivity
        G_brain, W_brain, _, _ = extract.brain_graph.weighted_undirected()
        D_brain, _ = extract.brain_graph.distance_matrix()
        # Calculate distributions
        d_bins, w_dists = \
        metrics.weighted_undirected.weight_distr_dist_binned(W_brain, 
                                                             D_brain, d_bins=70)
    # Make sure d_bins includes everything so we don't skip any edges
    d_bins[0], d_bins[-1] = 0., np.inf
    # Assign weights to edges by sampling them according to distance
    W_triu = np.triu(A)
    D_triu = np.triu(D)
    for bin_idx, w_dist in enumerate(w_dists):
        d_lower, d_upper = d_bins[bin_idx], d_bins[bin_idx + 1]
        # Draw samples for this distance bin
        n_samples = ((D_triu >= d_lower)*(D_triu < d_upper)*(W_triu > 0)).sum()
        sample_weights = np.random.choice(w_dist,n_samples)
        W_triu[(D_triu >= d_lower)*(D_triu < d_upper)*(W_triu > 0)] = sample_weights
    
    # Add weights as edge attributes to G
    keys = map(tuple,np.transpose(W_triu.nonzero())) # Convert to list of tuples
    values = W_triu[W_triu.nonzero()]
    nx.set_edge_attributes(G,'weight',dict(zip(keys,values)))
    
    # Make weight matrix symmetric
    W = W_triu + W_triu.T
    
    return G,W,D