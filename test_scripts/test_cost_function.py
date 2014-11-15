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
G_brain, W_brain, _, _ = extract.brain_graph.weighted_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
N_brain = W_brain.shape[0]
N_edges_brain = (W_brain > 0).sum()/2
L = 1.
gamma = 1.5

# Calculate swapped-cost distribution for graph
cost_changes = metrics.weighted_undirected.swapped_cost_distr(W_brain, D_brain)

# Create random biophysical graph with properly sampled weights
brain_size = np.array([10.,10.,10.])
G,W,D = rg.biophysical_sample_weights(N=N_brain, N_edges=N_edges_brain, L=L, 
                                      gamma=gamma, brain_size=brain_size,
                                      use_brain_weights=True)
                                      
# Calculate swapped-cost distribution for graph
cost_changes_random = metrics.weighted_undirected.swapped_cost_distr(W, D)

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