"""
Created on Mon Nov 24 09:17:11 2014

@author: rkp

Plot clustering vs. degree for mouse connectome and standard random graphs.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
import random_graph.binary_undirected as rg

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

# PLOTTING PARAMETERS
FACECOLOR = 'black'
FONTSIZE = 24
BRAIN_COLOR = 'k'
ER_COLOR = 'r'
WS_COLOR = 'g'
BA_COLOR = 'b'
MODEL_COLOR = 'c'
DEG_MAX = 250
DEG_TICKS = [0, 125, 250]
CC_TICKS = [0, .2, .4, .6, .8, 1.0]

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
# Erdos-Renyi
G_ER = nx.erdos_renyi_graph(n_nodes, p_edge)
ER_degree = nx.degree(G_ER).values()
ER_clustering = nx.clustering(G_ER).values()

# Watts-Strogatz
G_WS = nx.watts_strogatz_graph(n_nodes, int(round(brain_degree_mean)), 0.159)
WS_degree = nx.degree(G_WS).values()
WS_clustering = nx.clustering(G_WS).values()

# Barabasi-Albert
G_BA = nx.barabasi_albert_graph(n_nodes, int(round(brain_degree_mean / 2.)))
BA_degree = nx.degree(G_BA).values()
BA_clustering = nx.clustering(G_BA).values()

# Make 8 clustering vs. degree plots
fig, axs = plt.subplots(1, 4, facecolor=FACECOLOR)

# Brain
axs[0].scatter(brain_degree, brain_clustering, color=BRAIN_COLOR)

# Standard random graphs
axs[1].scatter(ER_degree, ER_clustering, color=ER_COLOR)
axs[2].scatter(WS_degree, WS_clustering, color=WS_COLOR)
axs[3].scatter(BA_degree, BA_clustering, color=BA_COLOR)

# Set axis limits and ticks, and label subplots
labels = ('a', 'b', 'c', 'd')
for ax_idx, ax in enumerate(axs.flatten()):
    ax.set_xlim(0, DEG_MAX)
    ax.set_ylim(0, 1)
    ax.set_xticks(DEG_TICKS)
    ax.set_yticks(CC_TICKS)
    ax.text(10, .88, labels[ax_idx], color='w', fontsize=FONTSIZE, fontweight='bold')

# Hide x ticklabels in top row & y ticklabels in right columns
for ax in axs.flatten()[1:]:
    ax.set_yticklabels('')

# Set titles
axs[0].set_title('Mouse')
axs[1].set_title('Erdos-Renyi')
axs[2].set_title('Small-World')
axs[3].set_title('Scale-Free')

# Set xlabels
for ax in axs:
    ax.set_xlabel('Degree')
axs[0].set_ylabel('Clustering\ncoefficient')

# Set all fontsizes and axis colors
for ax in axs.flatten():
    set_all_text_fontsizes(ax, FONTSIZE)
    set_all_colors(ax, 'w')


plt.draw()
