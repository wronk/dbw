"""
Created on Wed Nov 12 12:11:43 2014

@author: rkp

Graph-theory metrics for weighted undirected graphs.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import auxiliary as aux

def weight_distr_dist_binned(W, D, d_bins=50):
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
    
def fit_log_weight_vs_dist(W, D, d_bins=50, plot=False, ax_mu=None, ax_s=None):
    """Fit mean & std of log-weights to distance."""
    # Get weight distributions
    d_bins, weight_dists = weight_distr_dist_binned(W, D, d_bins=d_bins)
    d_bin_centers = .5 * (d_bins[:-1] + d_bins[1:])
    # Calculate logarithms of weights
    log_weight_dists = [np.log(weights) for weights in weight_dists]
    # Get means & stds
    mean_log_weights = [np.mean(log_weight) for log_weight in log_weight_dists]
    std_log_weights = [np.std(log_weight) for log_weight in log_weight_dists]
    # Fit means to distances
    mu_vs_d, intcpt_mu, r, p, stderr = stats.linregress(d_bin_centers,
                                                        mean_log_weights)
    # Fit stds to distances
    s_vs_d, intcpt_s, r, p, stderr = stats.linregress(d_bin_centers,
                                                      std_log_weights)
    # Plot if desired
    if plot:
        ax_mu.scatter(d_bin_centers, mean_log_weights)
        ax_mu.plot(d_bins, mu_vs_d * d_bins + intcpt_mu, lw=3, color='r')
        ax_mu.set_xlabel('Distance (mm)')
        ax_mu.set_ylabel('Log[weight]')
        ax_s.scatter(d_bin_centers, mean_log_weights)
        ax_s.plot(d_bins, s_vs_d * d_bins + intcpt_s, lw=3, color='r')
        ax_s.set_xlabel('Distance (mm)')
        ax_s.set_ylabel('Log[weight]')
    
    return mu_vs_d, intcpt_mu, s_vs_d, intcpt_s
    
def axon_volume_cost(W, D):
    """Calculate axon volume cost of network.
    
    Args:
        W: weight matrix
        D: distance matrix
    Returns:
        axon volume cost"""
    # Make sure self-weights are set to zero
    np.fill_diagonal(W,0)
    # Calculate cost by summing weights with distances
    return (np.triu(W)*D).sum()
    
def swapped_cost_distr(W, D, n_trials=500, percent_change=True):
    """Calculate how much the axon volume cost changes for a random node swap.
    
    Args:
        W: weight matrix
        D: distance matrix
        n_trials: how many times to make a random swap (starting from the 
            original configuration)
        percent_change: set to True to return percent changes in cost
    Returns:
        vector of cost changes for random swaps"""
    # Calculate true cost
    true_cost = axon_volume_cost(W, D)
    # Allocate space for storing amount by which cost changes
    cost_changes = np.zeros((n_trials,), dtype=float)
    
    # Perform random swaps
    for trial in range(n_trials):
        # Randomly select two nodes
        idx0, idx1 = np.random.permutation(D.shape[0])[:2]
        # Create new distance matrix
        D_swapped = aux.swap_nodes(D, idx0, idx1)
        # Calculate cost change for swapped-node graph
        cost_changes[trial] = axon_volume_cost(W, D_swapped) - true_cost
        
    if percent_change:
        cost_changes /= true_cost
        cost_changes *= 100
    
    return cost_changes