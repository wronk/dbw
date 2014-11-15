"""
Created on Wed Aug 27 23:16:17 2014

@author: rkp

Calculate and plot weight distributions binned by distance.
"""

import extract.brain_graph
import numpy as np
import matplotlib.pyplot as plt
import metrics.weighted_undirected

# Load graph & distance matrix
kwargs = {'p_th':.01,'w_th':0}
G, W, rows, cols = extract.brain_graph.weighted_undirected(p_th=.01, w_th=0)
D, _ = extract.brain_graph.distance_matrix()

# Calculate distance-dependent weight distributions
d_bins, weight_dists = \
metrics.weighted_undirected.weight_distr_dist_binned(W,D,d_bins=20)

# Plot all of the histogram
fig,ax = plt.subplots(1,1,facecolor='white')
for bin_idx, weight_dist in enumerate(weight_dists):
    d_lower, d_upper = d_bins[bin_idx], d_bins[bin_idx + 1]
    n_cxns = len(weight_dist)
    ax.cla()
    ax.hist(np.log(weight_dist),normed=True)
    ax.set_xlabel('Log(weight)')
    ax.set_ylabel('Probability')
    ax.set_title('%.2f mm < d < %.2f mm; %d cxns'%(d_lower,d_upper,n_cxns))
    plt.draw()
    raw_input()
    
