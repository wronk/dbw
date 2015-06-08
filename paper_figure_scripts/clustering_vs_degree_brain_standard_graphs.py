"""
Created on Mon Nov 24 09:17:11 2014

@author: rkp, wronk

Plot clustering vs. degree for mouse connectome and standard random graphs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
import config

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

# PLOTTING PARAMETERS
FACECOLOR = config.FACE_COLOR
FIGSIZE = (12, 3.5)
FONT_SIZE = config.FONT_SIZE
BRAIN_COLOR = config.COLORS['brain']
RAND_COLOR = config.COLORS['configuration']
WS_COLOR = config.COLORS['small-world']
BA_COLOR = config.COLORS['scale-free']
DEG_MAX = 150
DEG_TICKS = [0, 50, 100, 150]
CC_TICKS = [0, .2, .4, .6, .8, 1.0]
graph_names = ['Mouse', 'Random', 'Small-world', 'Scale-free']
labels = ['c', 'd', 'e', 'f']  # Upper corner labels for each plot
plt.ion()
plt.close('all')

########################################################

# Load mouse connectivity graph
G_brain, W_brain, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())
p_edge = float(n_edges) / ((n_nodes * (n_nodes - 1)) / 2.)

# Calculate degree & clustering coefficient distribution
brain_degree = nx.degree(G_brain).values()
brain_clustering = nx.clustering(G_brain).values()
brain_degree_mean = np.mean(brain_degree)

# Build standard graphs & get their degree & clustering coefficient
# Configuration model (random with fixed degree sequence)
G_CM = nx.random_degree_sequence_graph(sequence=brain_degree, tries=100)
CM_degree = nx.degree(G_CM).values()
CM_clustering = nx.clustering(G_CM).values()

# Watts-Strogatz
G_WS = nx.watts_strogatz_graph(n_nodes, int(round(brain_degree_mean)), 0.159)
WS_degree = nx.degree(G_WS).values()
WS_clustering = nx.clustering(G_WS).values()

# Barabasi-Albert
G_BA = nx.barabasi_albert_graph(n_nodes, int(round(brain_degree_mean / 2.)))
BA_degree = nx.degree(G_BA).values()
BA_clustering = nx.clustering(G_BA).values()

############
# Plot
############
# Make clustering vs. degree plots
fig, axs = plt.subplots(1, 4, facecolor=FACECOLOR, figsize=FIGSIZE,
                        tight_layout=True)

# Brain
axs[0].scatter(brain_degree, brain_clustering, color=BRAIN_COLOR)

# Standard random graphs
axs[1].scatter(CM_degree, CM_clustering, color=RAND_COLOR)
axs[2].scatter(WS_degree, WS_clustering, color=WS_COLOR)
axs[3].scatter(BA_degree, BA_clustering, color=BA_COLOR)

# Set axis limits and ticks, and label subplots
for ax_idx, ax in enumerate(axs.flatten()):
    ax.set_xlim(0, DEG_MAX)
    ax.set_ylim(0, 1)
    ax.set_xticks(DEG_TICKS)
    ax.set_yticks(CC_TICKS)
    ax.text(.08, .87, labels[ax_idx], color='k', fontsize=FONT_SIZE,
            fontweight='bold', transform=ax.transAxes)

    ax.set_xlabel('Degree')
    set_all_text_fontsizes(ax, FONT_SIZE)
    set_all_colors(ax, 'k')

    # Set titles
    ax.set_title(graph_names[ax_idx], fontsize=FONT_SIZE)

# Hide x ticklabels in top row & y ticklabels in right columns
for ax in axs[1:]:
    ax.set_yticklabels('')

axs[0].set_ylabel('Clustering\ncoefficient')

fig.subplots_adjust(wspace=0.18)

plt.draw()
