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
L = 1.3
GAMMA = 1.
BRAIN_SIZE = [10., 10, 10]

import extract.brain_graph
import random_graph.binary_undirected as rg

import numpy as np
import matplotlib.pyplot as plt

# Load brain graph & get distance matrix
G_brain, A_brain, _, _ = extract.brain_graph.binary_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
N_nodes = len(G_brain.nodes())
N_edges = len(G_brain.edges())

# Get distances of all brain edges
dists_brain = D_brain[np.triu(A_brain, k=1) > 0]
mean_dist_brain = np.mean(dists_brain)
med_dist_brain = np.median(dists_brain)
std_dist_brain = np.std(dists_brain)

# Build model
print 'gen...'
G_model, A_model, D_model = rg.biophysical(N_nodes, N_edges, L, GAMMA, 
                                           BRAIN_SIZE)

# Get distances of all model edges
dists_model = D_model[np.triu(A_model, k=1) > 0]
mean_dist_model = np.mean(dists_model)
med_dist_model = np.median(dists_model)
std_dist_model = np.std(dists_model)

# Plot histogram of cxn distances
fig, axs = plt.subplots(2, 1, facecolor=FACECOLOR)
axs[0].hist(dists_brain, bins=BINS)
axs[1].hist(dists_model, bins=BINS)

# Set labels
for row_idx, ax in enumerate(axs):
    ax.set_ylabel('Counts', fontsize=FONTSIZE)
    if row_idx == 0:
        ax.set_title('Edge distance histogram (brain)', fontsize=FONTSIZE)
    else:
        ax.set_title('Edge distance histogram (model)', fontsize=FONTSIZE)
    if row_idx:
        ax.set_xlabel('Edge distance (mm)', fontsize=FONTSIZE)

    for text in ax.get_xticklabels() + ax.get_yticklabels():
        text.set_fontsize(FONTSIZE)
    
plt.draw()
plt.tight_layout()

# Print statistics
print 'Mean edge distance (brain): %.2f mm' % mean_dist_brain
print 'Median edge distance (brain): %.2f mm' % med_dist_brain
print 'Std edge distance (brain): %.2f mm' % std_dist_brain
print 'Mean edge distance (model): %.2f mm' % mean_dist_model
print 'Median edge distance (model): %.2f mm' % med_dist_model
print 'Std edge distance (model): %.2f mm' % std_dist_model