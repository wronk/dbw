"""
Created on Mon Nov 17 09:26:35 2014

@author: rkp

Calculate the binary cost metric for real graph, biophysical, & random model.
"""

# PLOTTING PARAMETERS
FACECOLOR = 'white'

import numpy as np
import matplotlib.pyplot as plt

import extract.brain_graph
import metrics.binary_undirected
import random_graph.binary_undirected as rg

# Get mouse connectivity graph & distance
G_brain, A_brain, _ = extract.brain_graph.binary_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
n_nodes = A_brain.shape[0]
n_edges = (np.triu(A_brain)).sum()
p_edge = float(n_edges) / ((n_nodes * (n_nodes - 1)) / 2)
L = 2.2
GAMMA = 1.6
BRAIN_SIZE = [7., 7, 7]

# Calculate swapped-cost distribution for graph
cost_changes = metrics.binary_undirected.swapped_cost_distr(A_brain, D_brain)
positive_cost_changes = float((cost_changes > 0).sum()) / len(cost_changes)

# Create random biophysical graph with properly sampled weights
G_model, A_model, D_model = rg.biophysical(N=n_nodes, N_edges=n_edges, L=L, 
                                           gamma=GAMMA, brain_size=BRAIN_SIZE)
# Calculate swapped-cost distribution for graph
cost_changes_model = metrics.binary_undirected.swapped_cost_distr(A_model, 
                                                                   D_model)
positive_cost_changes_model = float((cost_changes_model > 0).sum()) / \
len(cost_changes_model)

# Create ER graph with distance matrix
G_ERD, A_ERD, D_ERD = rg.ER_distance(n_nodes, p_edge, brain_size=BRAIN_SIZE)
# Calculate swapped-cost distribution for graph
cost_changes_ERD = metrics.binary_undirected.swapped_cost_distr(A_ERD, D_ERD)
positive_cost_changes_ERD = float((cost_changes_ERD > 0).sum()) / \
len(cost_changes_ERD)

fig, axs = plt.subplots(3, 1, facecolor=FACECOLOR)
axs[0].hist(cost_changes, bins=20, normed=True)
axs[1].hist(cost_changes_model, bins=20, normed=True)
axs[2].hist(cost_changes_ERD, bins=20, normed=True)

for ax_idx, ax in enumerate(axs):
    ax.set_xlabel('Change in cost (per cent)')
    ax.set_xlim(-1,2)
    ax.set_ylabel('Probability')
    
axs[0].set_title('Mouse brain')
axs[1].set_title('Biophysical model')
axs[2].set_title('Erdos Renyi with distance')
plt.draw()
plt.tight_layout()

print 'Real brain cost changes percent > 0: %.3f' % positive_cost_changes
print 'Model cost changes percent > 0: %.3f' % positive_cost_changes_model
print 'Erdos-Renyi cost changes percent > 0: %.3f' % positive_cost_changes_ERD