"""
Created on Mon Nov 17 10:01:22 2014

@author: rkp

Calculate the length scale of the real brain & our biophysical model
"""

# PLOT PARAMETERS
FACECOLOR = 'white'

# ANALYSIS PARAMETERS
L = 1.5
GAMMA = 1.
BRAIN_SIZE = [9., 9, 9]
BINS = 50

import numpy as np
import matplotlib.pyplot as plt

import extract.brain_graph
import metrics.binary_undirected
import random_graph.binary_undirected as rg

# Load brain
G_brain, A_brain, _, _ = extract.brain_graph.binary_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())

# Calculate length scale
L_brain, r, p = metrics.binary_undirected.cxn_length_scale(A_brain, D_brain,
                                                           bins=BINS)

# Create biophysical model
G, A, D = rg.biophysical(n_nodes, n_edges, L=L, gamma=GAMMA, 
                         brain_size=BRAIN_SIZE)

# Plot connection distance histograms
fig, axs = plt.subplots(2, 1, facecolor=FACECOLOR)

prob_brain, bins_brain = np.histogram(D_brain[np.triu(A_brain) > 0], bins=BINS, 
                                      normed=True)
bin_centers_brain = .5 * (bins_brain[:-1] + bins_brain[1:])
bin_width_brain = bins_brain[1] - bins_brain[0]
axs[0].bar(bin_centers_brain, prob_brain, width=bin_width_brain)

prob_model, bins_model = np.histogram(D[np.triu(A) > 0], bins=BINS, normed=True)
bin_centers_model = .5 * (bins_model[:-1] + bins_model[1:])
bin_width_model = bins_model[1] - bins_model[0]
axs[1].bar(bin_centers_model, prob_model, width=bin_width_model)