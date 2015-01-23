"""
Created on Mon Nov 17 19:10:04 2014

@author: rkp

Plot a histogram of edge distances.
"""

# PLOT PARAMETERS
FACECOLOR = 'white'
FONTSIZE = 20
BINS = 50

# ANALYSIS PARAMETERS
L = 1.7
GAMMA = 2.0
BRAIN_SIZE = [7., 7., 7.]

import extract.brain_graph
import random_graph.binary_undirected as rg

import numpy as np
import matplotlib.pyplot as plt

# Load brain graph & get distance matrix
G_brain, A_brain, _ = extract.brain_graph.binary_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
N_nodes = len(G_brain.nodes())
N_edges = len(G_brain.edges())

# Get all pairwise distances in brain
pairwise_dists_brain = D_brain[np.triu(D_brain, k=1) > 0]
mean_pairwise_dist_brain = np.mean(pairwise_dists_brain)
med_pairwise_dist_brain = np.median(pairwise_dists_brain)
std_pairwise_dist_brain = np.std(pairwise_dists_brain)

# Get all edge distances in brain
edge_dists_brain = D_brain[np.triu(A_brain, k=1) > 0]
mean_edge_dist_brain = np.mean(edge_dists_brain)
med_edge_dist_brain = np.median(edge_dists_brain)
std_edge_dist_brain = np.std(edge_dists_brain)

# Build model
print 'gen...'
G_model, A_model, D_model = rg.biophysical(N_nodes, N_edges, L, GAMMA, 
                                           BRAIN_SIZE)

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
axs[0].hist(pairwise_dists_brain, bins=BINS)
axs[1].hist(pairwise_dists_model, bins=BINS)

# Set labels
for row_idx, ax in enumerate(axs):
    ax.set_ylabel('Counts', fontsize=FONTSIZE)
    if row_idx == 0:
        ax.set_title('Pairwise distance histogram (brain)', fontsize=FONTSIZE)
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
print 'Mean pairwise distance (brain): %.2f mm' % mean_pairwise_dist_brain
print 'Median pairwise distance (brain): %.2f mm' % med_pairwise_dist_brain
print 'Std pairwise distance (brain): %.2f mm' % std_pairwise_dist_brain
print 'Mean pairwise distance (model): %.2f mm' % mean_pairwise_dist_model
print 'Median pairwise distance (model): %.2f mm' % med_pairwise_dist_model
print 'Std pairwise distance (model): %.2f mm' % std_pairwise_dist_model


# Plot histogram of edge distances
fig, axs = plt.subplots(2, 1, facecolor=FACECOLOR)
axs[0].hist(edge_dists_brain, bins=BINS)
axs[1].hist(edge_dists_model, bins=BINS)

# Set labels
for row_idx, ax in enumerate(axs):
    ax.set_ylabel('Counts', fontsize=FONTSIZE)
    if row_idx == 0:
        ax.set_title('Edge distance histogram (brain)', fontsize=FONTSIZE)
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
print 'Mean edge distance (brain): %.2f mm' % mean_edge_dist_brain
print 'Median edge distance (brain): %.2f mm' % med_edge_dist_brain
print 'Std edge distance (brain): %.2f mm' % std_edge_dist_brain
print 'Mean edge distance (model): %.2f mm' % mean_edge_dist_model
print 'Median edge distance (model): %.2f mm' % med_edge_dist_model
print 'Std edge distance (model): %.2f mm' % std_edge_dist_model