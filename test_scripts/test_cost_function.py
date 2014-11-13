"""
Created on Wed Aug 27 23:16:17 2014

@author: rkp

Test creation of random weighted undirected graphs.
"""

import numpy as np
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