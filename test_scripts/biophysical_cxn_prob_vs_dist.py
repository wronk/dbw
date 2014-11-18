"""
Created on Mon Nov 17 11:30:45 2014

@author: rkp
"""

# ANALYSIS PARAMETERS
N_BIO_GRAPHS = 1
L = 1.3 # Length scale parameter
GAMMA = 1.7 # Preferential attachment parameter
BRAIN_SIZE = [10., 10, 10] # Size of volume in which brain regions are distributed

import extract.brain_graph
import metrics.binary_undirected
import random_graph.binary_undirected as rg

import numpy as np
import matplotlib.pyplot as plt

# Load brain network
G_brain, A_brain, _, _ = extract.brain_graph.binary_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
n_nodes = D_brain.shape[0]
n_edges = np.triu(A_brain, k=1).sum()
# Calculate connection probability vs. distance
cxn_prob, dist_bins = metrics.binary_undirected.cxn_probability(A_brain, 
                                                                D_brain)
dist_bin_centers = .5 * (dist_bins[:-1] + dist_bins[1:])
bin_width = dist_bins[1] - dist_bins[0]
                                            
# Build model
G, A, D = rg.biophysical(n_nodes, n_edges, L=L, gamma=GAMMA, 
                         brain_size=BRAIN_SIZE)
# Calculate connection probability vs. distance
cxn_prob_model, dist_bins_model = \
metrics.binary_undirected.cxn_probability(A, D)
dist_bin_centers_model = .5 * (dist_bins_model[:-1] + dist_bins_model[1:])
bin_width_model = dist_bins_model[1] - dist_bins_model[0]

# Plot both
fig, axs = plt.subplots(2, 1, facecolor='white')
axs[0].bar(dist_bin_centers, cxn_prob, width=bin_width)
axs[1].bar(dist_bin_centers_model, cxn_prob_model, width=bin_width_model)