"""
Created on Wed May 13th 2015

@author: wronk

Plot undirected degree distributions for brain and standard graphs
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

from networkx import erdos_renyi_graph as er
from networkx import barabasi_albert_graph as ba
from networkx import watts_strogatz_graph as ws

###############################################
# Plotting params
###############################################
FACECOLOR = 'black'
FONTSIZE = 24

BRAIN_COLOR = 'm'
ER_COLOR = 'r'
WS_COLOR = 'g'
BA_COLOR = 'b'
MODEL_COLOR = 'c'

n_bins = 60
plt.ion()
plt.close('all')

repeats = 100
###############################################
# Calculate degree distributions for all graphs
###############################################
# Load mouse connectivity graph
G_brain, W_brain, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())
edge_density = float(n_edges) / ((n_nodes * (n_nodes - 1)) / 2.)

# Mouse connectome
brain_degree = nx.degree(G_brain).values()
brain_degree_mean = np.mean(brain_degree)

# Initialize repetition matrices for standard graphs
ER_deg_mat = -1 * np.ones((repeats, n_nodes))
WS_deg_mat = -1 * np.ones((repeats, n_nodes))
BA_deg_mat = -1 * np.ones((repeats, n_nodes))

# XXX: Should these be reseeded for each repeat?
for r in np.arange(repeats):
    # Erdos-Renyi
    ER_deg_mat[r, :] = er(n_nodes, edge_density).degree().values()

    # Watts-Strogatz
    WS_deg_mat[r, :] = ws(n_nodes, int(round(brain_degree_mean)),
                          0.159).degree().values()

    # Barabasi-Albert
    BA_deg_mat[r, :] = ba(n_nodes,
                          int(round(brain_degree_mean / 2.))).degree().values()

deg_dists = [WS_deg_mat.flatten(), brain_degree,
             ER_deg_mat.flatten(), BA_deg_mat.flatten()]
colors = [WS_COLOR, BRAIN_COLOR, ER_COLOR, BA_COLOR]
labels = ['Small-world', 'Mouse Connectome', 'Random', 'Scale-free']
histtype = ['stepfilled', 'step', 'stepfilled', 'stepfilled']

##################
# Plot
#################

figsize = (4, 4)
fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor=FACECOLOR)

ax.hist(deg_dists, n_bins, normed=1, histtype='step', color=colors,
        label=labels, lw=2, alpha=1.0)

# Set axis limits and ticks, and label subplots
labels = ('a', 'b', 'c', 'd')
#ax.set_xlim([0, 125])
ax.set_ylim([0, .15])
ax.locator_params(axis='x', nbins=4)
ax.locator_params(axis='y', nbins=4)
#ax.text(10, .88, labels[ax_idx], color='w', fontsize=FONTSIZE,
#        fontweight='bold')
ax.legend(loc='best')

# Hide x ticklabels in top row & y ticklabels in right columns
#ax.set_yticklabels('')

# Set title
ax.set_title('Degree Distributions')

# Set xlabels
ax.set_xlabel('Degree')
ax.set_ylabel('P(k)')

# Set all fontsizes and axis colors
set_all_text_fontsizes(ax, FONTSIZE)
set_all_colors(ax, 'w')

plt.draw()
