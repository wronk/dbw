"""
Created on Mon Nov 17 10:01:22 2014

@author: rkp

Calculate the length scale of the real brain & our biophysical model
"""

# PLOT PARAMETERS
FACECOLOR = 'white'

# ANALYSIS PARAMETERS
GAMMA = 1.5
BINS = 50

import numpy as np
import matplotlib.pyplot as plt

import extract.brain_graph
import metrics.binary_undirected
import random_graph.binary_undirected as rg

# Load brain
G_brain, A_brain, _ = extract.brain_graph.binary_undirected()
D_brain, _ = extract.brain_graph.distance_matrix()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())

# Calculate length scale
L_brain, r, p = metrics.binary_undirected.cxn_length_scale(A_brain, D_brain,
                                                           bins=BINS)

# Create biophysical model
G, A, D = rg.biophysical(n_nodes, n_edges, L_brain, GAMMA, 
                         brain_size=[10.,10,10])

# Calculate length scale of biophysical model
L, r, p = metrics.binary_undirected.cxn_length_scale(A, D, bins=BINS)

# Plot length histograms plus fitted exponentials 
fig, axs = plt.subplots(2, 1, facecolor=FACECOLOR)

prob_brain, bins_brain = np.histogram(D_brain[np.triu(A_brain) > 0], bins=BINS, 
                                      normed=True)
bin_centers_brain = .5 * (bins_brain[:-1] + bins_brain[1:])
axs[0].bar(bin_centers_brain, prob_brain)
exp_fit_x = np.linspace(0, D.max(), 1000)
exp_fit_y = (1. / L) * np.exp(-exp_fit_x / L)
axs[0].plot(exp_fit_x, exp_fit_y, color='k', lw=3)
