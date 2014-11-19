"""
Created on Wed Aug 27 23:16:17 2014

@author: rkp

Test creation of random weighted undirected graphs.
"""

# PLOTTING PARAMETERS
FACECOLOR = 'white'

import numpy as np
import matplotlib.pyplot as plt

import extract.brain_graph
import metrics.weighted_undirected
import random_graph.weighted_undirected as rg

# Get mouse connectivity graph & distance
G_brain, W_brain, _ = extract.brain_graph.weighted_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
N_brain = W_brain.shape[0]
N_edges_brain = (W_brain > 0).sum()/2
L = 1.3
GAMMA = 1.7
BRAIN_SIZE = [9., 9, 9]

# Calculate swapped-cost distribution for graph
cost_changes = metrics.weighted_undirected.swapped_cost_distr(W_brain, D_brain)
positive_cost_changes = float((cost_changes > 0).sum()) / len(cost_changes)

# Create random biophysical graph with properly sampled weights
G,W,D = rg.biophysical_sample_weights(N=N_brain, N_edges=N_edges_brain, L=L, 
                                      gamma=GAMMA, brain_size=BRAIN_SIZE,
                                      use_brain_weights=True)
                                      
# Calculate swapped-cost distribution for graph
cost_changes_random = metrics.weighted_undirected.swapped_cost_distr(W, D)
positive_cost_changes_random = float((cost_changes_random > 0).sum()) / \
len(cost_changes_random)

fig, axs = plt.subplots(2, 1, facecolor=FACECOLOR)
axs[0].hist(cost_changes, bins=20, normed=True)
axs[1].hist(cost_changes_random, bins=20, normed=True)

for ax_idx, ax in enumerate(axs):
    if ax_idx == 1:
        ax.set_xlabel('Change in cost (per cent)')
    ax.set_xlim(-1,10)
    ax.set_ylabel('Probability')
    
axs[0].set_title('Mouse brain')
axs[1].set_title('Weighted biophysical model')

print 'Real brain cost changes percent > 0: %.3f' % positive_cost_changes
print 'Simulated cost changes percent > 0: %.3f' % positive_cost_changes_random