"""
Created on Mon Nov 17 19:10:04 2014

@author: rkp

Plot a histogram of edge distances in cortex graph.
"""

# PLOT PARAMETERS
FACECOLOR = 'white'
FONTSIZE = 20
BINS = 50

# ANALYSIS PARAMETERS
L = 2.3
GAMMA = 1.
CORTEX_SIZE = [7., 7, 7]

import extract.brain_subgraph
import random_graph.binary_undirected as rg

import numpy as np
import matplotlib.pyplot as plt

# Load brain graph & get distance matrix
G_cortex, A_cortex, _ = extract.brain_subgraph.cortex_binary_undirected()
D_cortex, _ = extract.brain_subgraph.cortex_distance_matrix()
N_nodes = len(G_cortex.nodes())
N_edges = len(G_cortex.edges())

# Get all pairwise distances in brain
pairwise_dists_cortex = D_cortex[np.triu(D_cortex, k=1) > 0]
mean_pairwise_dist_cortex = np.mean(pairwise_dists_cortex)
med_pairwise_dist_cortex = np.median(pairwise_dists_cortex)
std_pairwise_dist_cortex = np.std(pairwise_dists_cortex)

# Get all edge distances in brain
edge_dists_cortex = D_cortex[np.triu(A_cortex, k=1) > 0]
mean_edge_dist_cortex = np.mean(edge_dists_cortex)
med_edge_dist_cortex = np.median(edge_dists_cortex)
std_edge_dist_cortex = np.std(edge_dists_cortex)

# Build model
print 'gen...'
G_model, A_model, D_model = rg.biophysical(N_nodes, N_edges, L, GAMMA, 
                                           CORTEX_SIZE)

# Get all pairwise distances in model
pairwise_dists_model = D_model[np.triu(D_model, k=1) > 0]
mean_pairwise_dist_model = np.mean(pairwise_dists_model)
med_pairwise_dist_model = np.median(pairwise_dists_model)
std_pairwise_dist_model = np.std(pairwise_dists_model)

# Get all edge distances in model
edge_dists_model = D_model[np.triu(A_model, k=1) > 0]
mean_edge_dist_model = np.mean(edge_dists_model)
med_edge_dist_model = np.median(edge_dists_model)
std_edge_dist_model = np.std(edge_dists_model)

# Plot histogram of pairwise distances
fig, axs = plt.subplots(2, 1, facecolor=FACECOLOR)
axs[0].hist(pairwise_dists_cortex, bins=BINS)
axs[1].hist(pairwise_dists_model, bins=BINS)

# Set labels
for row_idx, ax in enumerate(axs):
    ax.set_ylabel('Counts', fontsize=FONTSIZE)
    if row_idx == 0:
        ax.set_title('Pairwise distance histogram (cortex)', fontsize=FONTSIZE)
    else:
        ax.set_title('Pairwise distance histogram (model)', fontsize=FONTSIZE)
    if row_idx:
        ax.set_xlabel('Pairwise distance (mm)', fontsize=FONTSIZE)
    ax.set_xlim(0,12)
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)
    
plt.draw()
plt.tight_layout()

# Print statistics
print 'Mean pairwise distance (cortex): %.2f mm' % mean_pairwise_dist_cortex
print 'Median pairwise distance (cortex): %.2f mm' % med_pairwise_dist_cortex
print 'Std pairwise distance (cortex): %.2f mm' % std_pairwise_dist_cortex
print 'Mean pairwise distance (model): %.2f mm' % mean_pairwise_dist_model
print 'Median pairwise distance (model): %.2f mm' % med_pairwise_dist_model
print 'Std pairwise distance (model): %.2f mm' % std_pairwise_dist_model


# Plot histogram of edge distances
fig, axs = plt.subplots(2, 1, facecolor=FACECOLOR)
axs[0].hist(edge_dists_cortex, bins=BINS)
axs[1].hist(edge_dists_model, bins=BINS)

# Set labels
for row_idx, ax in enumerate(axs):
    ax.set_ylabel('Counts', fontsize=FONTSIZE)
    if row_idx == 0:
        ax.set_title('Edge distance histogram (cortex)', fontsize=FONTSIZE)
    else:
        ax.set_title('Edge distance histogram (model)', fontsize=FONTSIZE)
    if row_idx:
        ax.set_xlabel('Edge distance (mm)', fontsize=FONTSIZE)
    ax.set_xlim(0,12)
    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)
    
plt.draw()
plt.tight_layout()

# Print statistics
print 'Mean edge distance (brain): %.2f mm' % mean_edge_dist_cortex
print 'Median edge distance (brain): %.2f mm' % med_edge_dist_cortex
print 'Std edge distance (brain): %.2f mm' % std_edge_dist_cortex
print 'Mean edge distance (model): %.2f mm' % mean_edge_dist_model
print 'Median edge distance (model): %.2f mm' % med_edge_dist_model
print 'Std edge distance (model): %.2f mm' % std_edge_dist_model