"""
Created on Wed May 13th 2015

@author: wronk

Plot undirected degree distributions for brain and standard graphs
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph

from network_plot.change_settings import set_all_text_fontsizes

from networkx import erdos_renyi_graph as er
from networkx import barabasi_albert_graph as ba
from networkx import watts_strogatz_graph as ws

###############################################
# Plotting params
###############################################
FACECOLOR = 'white'
FONTSIZE = 24

BRAIN_COLOR = 'm'
ER_COLOR = 'r'
WS_COLOR = 'g'
BA_COLOR = 'b'
MODEL_COLOR = 'c'

plt.ion()
plt.close('all')
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42

n_bins = 70
repeats = 200
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

deg_dists = [WS_deg_mat.flatten(), ER_deg_mat.flatten(), BA_deg_mat.flatten()]
colors = [WS_COLOR, ER_COLOR, BA_COLOR]
labels = ['Small-world', 'Random', 'Scale-free']

label_brain = 'Mouse Connectome'

###################################################
# Plot semilogy (looking for exponential solutions)
###################################################
lin_bins = np.linspace(0, 200, n_bins)

figsize = (8, 6)
fig, ax = plt.subplots(1, 1, figsize=figsize)

for deg, col, lab in zip(deg_dists, colors, labels):
    hist, plt_bins = np.histogram(deg, lin_bins, normed=True)
    ax.plot(plt_bins[:-1], hist, lw=2, color=col, label=lab)

hist_brain, plt_bins = np.histogram(brain_degree, lin_bins, normed=True)
ax.plot(plt_bins[:-1], hist_brain, lw=2, color=BRAIN_COLOR,
        label=label_brain)

# Set axis limits and ticks, and label subplots
ax.set_xlim([0, 200])
#ax.set_ylim([0, .15])
ax.locator_params(axis='x', nbins=5)
#ax.locator_params(axis='y', nbins=5)
ax.legend(loc='best', fontsize=FONTSIZE - 8)

# Set title
ax.set_title('Degree Distributions')

# Set xlabels
ax.set_xlabel('Degree')
ax.set_ylabel('P(k)')

# Set all fontsizes and axis colors
set_all_text_fontsizes(ax, FONTSIZE)
#set_all_colors(ax, 'w')

plt.tight_layout()

###################################################
# Plot semilogy (looking for exponential solutions)
###################################################
raw_input('Press key')
ax.set_yscale('log')
ax.set_ylabel('Log[P(k)]')
plt.draw()

###################################################
# Plot on log scale (looking for power-law
###################################################

raw_input('Press key')
ax.set_xscale('log')
ax.set_xlim([1, 200])
ax.set_xlabel('Log[Degree]')
ax.set_ylabel('Log[P(k)]')
plt.draw()
