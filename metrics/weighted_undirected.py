"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Graph-theory metrics for weighted undirected graphs.
"""

import numpy as np

def weight_distr_dist_binned(W,D,d_bins=50):
    """Calculate weight distributions sorted by distance.
    
    Note: if W is not symmetrical, it is assumed to be directed and is made
    symmetrical by adding it to its transpose
    
    Args:
        W: weight matrix
        D: distance matrix
        bins: how many bins/bin vector
    Returns:
        Distance bin edges, weight distributions."""
    if not (W == W.T).all():
        W_sym = W + W.T
    else:
        W_sym = W.copy()
    # Set all weights below diagonal to zero so we don't double count
    W_sym = np.triu(W_sym)
    # Remove all zero-weight cxns
    w = W_sym[W_sym > 0]
    d = D[W_sym > 0]
    # Get distance bins
    d_hist, d_bins = np.histogram(d,d_bins)
    
    # Calculate distance-dependent weight distributions
    weight_dists = [None for bin_idx in range(len(d_bins) - 1)]
    for bin_idx in range(len(weight_dists)):
        d_lower = d_bins[bin_idx]
        d_upper = d_bins[bin_idx+1]
        
        weight_dists[bin_idx] = w[(d >= d_lower) * (d < d_upper)]
        
    return d_bins, weight_dists