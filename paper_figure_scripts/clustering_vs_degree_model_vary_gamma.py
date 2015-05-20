"""
Created on Mon Nov 24 09:17:11 2014

@author: rkp

Plot the clustering vs. degree for binary undirected biophysical model at
multiple gammas.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
import random_graph.binary_undirected as rg

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

# PLOTTING PARAMETERS
FACECOLOR = 'w'
FIGSIZE = (12, 3.5)
TEXTCOLOR = 'k'
FONTSIZE = 18
MODEL_COLOR = 'c'
DEG_MAX = 250
DEG_TICKS = [0, 125, 250]
CC_TICKS = [0, .2, .4, .6, .8, 1.0]

LS = [np.inf, np.inf, np.inf, np.inf]  # Length scale parameters all infinite
GAMMAS = [1., 1.333, 1.667, 2.0]  # Preferential attachment parameters
BRAIN_SIZE = [7., 7., 7.]  # Size brain region volume to distribute nodes

# Load mouse connectivity graph
G_brain, W_brain, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())
p_edge = float(n_edges) / ((n_nodes * (n_nodes - 1)) / 2.)

# Loop through model graphs with different gamma
model_degrees = [None for gamma in GAMMAS]
model_clusterings = [None for gamma in GAMMAS]

for gamma_idx, gamma in enumerate(GAMMAS):
    L = LS[gamma_idx]
    print 'Generating model graph for gamma = %.2f' % gamma
    G_model, A_model, D_model = rg.biophysical(n_nodes, n_edges, L, gamma,
                                               BRAIN_SIZE)
    model_degree = nx.degree(G_model).values()
    model_clustering = nx.clustering(G_model).values()

    # Store examples
    model_degrees[gamma_idx] = model_degree
    model_clusterings[gamma_idx] = model_clustering

# Make clustering vs. degree plots
fig, axs = plt.subplots(1, 4, facecolor=FACECOLOR, figsize=FIGSIZE, tight_layout=True)

# Model graphs w/ different gammas
for gamma_idx, gamma in enumerate(GAMMAS):
    degree = model_degrees[gamma_idx]
    clustering = model_clusterings[gamma_idx]
    axs[gamma_idx].scatter(degree, clustering, color=MODEL_COLOR)

# Set axis limits and ticks, and label subplots
labels = ('a', 'b', 'c', 'd')
plt.ion()
for ax_idx, ax in enumerate(axs.flat):
    ax.set_xlim(0, DEG_MAX)
    ax.set_ylim(0, 1)
    ax.set_xticks(DEG_TICKS)
    ax.set_yticks(CC_TICKS)
    ax.text(10, .88, labels[ax_idx], color=TEXTCOLOR, fontsize=FONTSIZE,
            fontweight='bold')

# Hide x ticklabels in top row & y ticklabels in right columns
for row in axs.flatten()[1:]:
    ax.set_yticklabels('')

# Set titles
for gamma_idx, gamma in enumerate(GAMMAS):
    title = r'$\gamma$ = %.2f' % gamma
    axs[gamma_idx].set_title(title)

# Set xlabels
for ax in axs:
    ax.set_xlabel('Degree')
axs[0].set_ylabel('Clustering\ncoefficient')

# Set all fontsizes and axis colors
for ax in axs.flatten():
    set_all_text_fontsizes(ax, FONTSIZE)
    set_all_colors(ax, TEXTCOLOR)

plt.draw()
