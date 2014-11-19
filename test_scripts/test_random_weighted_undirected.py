"""
Created on Wed Aug 27 23:16:17 2014

@author: rkp

Test creation of random weighted undirected graphs.
"""

import numpy as np
import extract.brain_graph
import random_graph.weighted_undirected as rg

# Load weight matrix from mouse connectivity to get model parameters
G_brain, W_brain, _ = extract.brain_graph.weighted_undirected()
N_brain = W_brain.shape[0]
N_edges_brain = (W_brain > 0).sum()/2
L = 1.
gamma = 1.5

# Create random biophysical graph with properly sampled weights
brain_size = np.array([10.,10.,10.])
G,W,D = rg.biophysical_sample_weights(N=N_brain, N_edges=N_edges_brain, L=L, 
                                      gamma=gamma, brain_size=brain_size,
                                      use_brain_weights=True)