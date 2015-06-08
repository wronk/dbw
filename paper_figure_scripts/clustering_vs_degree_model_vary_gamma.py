"""
Created on Mon Nov 24 09:17:11 2014

@author: rkp, wronk

Plot the clustering vs. degree for binary undirected biophysical model at
multiple gammas.
"""

import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt
import extract.brain_graph
import random_graph.binary_undirected as rg
import config

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

assert mpl.__version__ == '1.4.3', 'Wrong matplotlib version, need 1.4.3'

# PLOTTING PARAMETERS
FACE_COLOR = config.FACE_COLOR
FIGSIZE = (12.5, 4.5)
TEXTCOLOR = 'k'
FONT_SIZE = config.FONT_SIZE
DEG_MAX = 150
DEG_TICKS = [0, 50, 100, DEG_MAX]
CC_TICKS = [0, .2, .4, .6, .8, 1.0]
ax_labels = ['c', 'd', 'e']

GAMMAS = [1.5, 1.75, 2.0]  # Preferential attachment parameters
LS = [np.inf] * len(GAMMAS)  # Length scale parameters all infinite
BRAIN_SIZE = [7., 7., 7.]  # Size brain region volume to distribute nodes
MODEL_COLOR = ['k'] * len(GAMMAS)

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
plt.rcParams['ps.fonttype'] = 42  # Set for text in Adobe Illustrator
plt.rcParams['pdf.fonttype'] = 42

fig, axs = plt.subplots(1, len(GAMMAS), facecolor=FACE_COLOR, figsize=FIGSIZE,
                        sharey=True, tight_layout=True)

# Convert to list if not 1 x n
if type(axs) == np.ndarray:
    axs = list(axs.ravel())

# Make clustering vs. degree plots w/ different gammas
for gamma_idx, gamma in enumerate(GAMMAS):
    axs[gamma_idx].scatter(model_degrees[gamma_idx],
                           model_clusterings[gamma_idx],
                           color=MODEL_COLOR[gamma_idx],
                           alpha=0.5)

# Set axis limits and ticks, and label subplots
for ax_ind, (ax, label) in enumerate(zip(axs, ax_labels)):
    ax.set_xlim(0, DEG_MAX)
    ax.set_ylim(0, 1)
    ax.set_xticks(DEG_TICKS)
    ax.set_yticks(CC_TICKS)

    ax.text(20, .87, label, color=TEXTCOLOR, fontsize=FONT_SIZE,
            fontweight='bold')

    set_all_text_fontsizes(ax, FONT_SIZE)
    set_all_colors(ax, TEXTCOLOR)

    # Set titles/labels
    ax.set_xlabel('Degree')
    ax.set_title(r'$\gamma$ = %.2f' % GAMMAS[ax_ind], fontsize=FONT_SIZE)

axs[0].set_ylabel('Clustering\ncoefficient')

#fig.subplots_adjust(wspace=0.18)

plt.plot()
