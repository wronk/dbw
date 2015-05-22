"""
Created on Mon Nov 24 09:17:11 2014

@author: rkp, wronk

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
MODEL_COLOR = ['c'] * 4
DEG_MAX = 250
DEG_TICKS = [0, 125, DEG_MAX]
CC_TICKS = [0, .2, .4, .6, .8, 1.0]
ax_labels = ('a', 'b', 'c', 'd')

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
    # Store examples
    model_degrees[gamma_idx] = nx.degree(G_model).values()
    model_clusterings[gamma_idx] = nx.clustering(G_model).values()

#################################
# Plotting
#################################
plt.ion()
plt.close('all')
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42

fig, axs = plt.subplots(1, 4, facecolor=FACECOLOR, figsize=FIGSIZE,
                        sharey=True, tight_layout=True)

# Convert to list if not 1 x n
if type(axs) == np.ndarray:
    axs = list(axs.ravel())

# Make clustering vs. degree plots w/ different gammas
for gamma_idx, gamma in enumerate(GAMMAS):
    axs[gamma_idx].scatter(model_degrees[gamma_idx],
                           model_clusterings[gamma_idx],
                           color=MODEL_COLOR[gamma_idx])

# Set axis limits and ticks, and label subplots
for ax_ind, (ax, label) in enumerate(zip(axs, ax_labels)):
    ax.set_xlim(0, DEG_MAX)
    ax.set_ylim(0, 1)
    ax.set_xticks(DEG_TICKS)
    ax.set_yticks(CC_TICKS)

    ax.text(19, .88, label, color=TEXTCOLOR, fontsize=FONTSIZE,
            fontweight='bold')

    set_all_text_fontsizes(ax, FONTSIZE)
    set_all_colors(ax, TEXTCOLOR)

    # Set titles/labels
    ax.set_xlabel('Degree')
    ax.set_title(r'$\gamma$ = %.2f' % GAMMAS[ax_ind],
                 fontsize=FONTSIZE)

    # Hide axis labels if not first axis
    #if ax_ind != 0:
        #ax.set_yticklabels('')

axs[0].set_ylabel('Clustering\ncoefficient')

fig.subplots_adjust(wspace=0.18)

plt.plot()
