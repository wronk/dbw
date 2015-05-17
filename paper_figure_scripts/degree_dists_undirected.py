"""
Created on Wed May 13th 2015

@author: wronk

Plot undirected degree distributions for brain and standard graphs
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
from copy import copy
from copy import deepcopy

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

plt.ion()
plt.close('all')
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42

n_bins = 50
repeats = 1000

###############################################
# Histogram plot function
###############################################
# Function to plot a list of histograms on a single axis. Allows code to be
# reused for similar plots that will have just a change in x or y scale (log)


def hist_plot(ax, deg_dists, colors, labels):
    for deg, col, lab in zip(deg_dists, colors, labels):
        hist, plt_bins = np.histogram(deg, lin_bins, normed=True)
        ax.plot(plt_bins[:-1], hist, lw=2, color=col, label=lab)

    # Set axis limits and ticks, and label subplots
    ax.set_xlim([0, 150])
    #ax.set_ylim([0, .15])
    ax.locator_params(axis='x', nbins=5)
    #a.locator_params(axis='y', nbins=5)
    ax.legend(loc='best', fontsize=FONTSIZE - 10)

    # Set title
    ax.set_title('Degree Distributions')

    # Set xlabels
    ax.set_xlabel('Degree')
    ax.set_ylabel('P(k)')

    # Set all fontsizes and axis colors
    set_all_text_fontsizes(ax, FONTSIZE)
    #set_all_colors(ax, 'w')

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

label_brain = 'Mouse\nConnectome'

###################################################
# Plot semilogy (looking for exponential solutions)
###################################################
lin_bins = np.linspace(0, 150, n_bins)

figsize = (16, 5)
fig, axs = plt.subplots(1, 3, figsize=figsize)

hist_plot(axs[0], deg_dists, colors, labels)
hist_plot(axs[0], [brain_degree], [BRAIN_COLOR], [label_brain])
axs[0].locator_params(axis='y', nbins=5)

###################################################
# Plot semilogy (looking for exponential solutions)
###################################################
hist_plot(axs[1], deg_dists, colors, labels)
hist_plot(axs[1], [brain_degree], [BRAIN_COLOR], [label_brain])

axs[1].set_ylim([10E-4, 1])
axs[1].set_yscale('log')
axs[1].set_ylabel('Log[P(k)]')
axs[1].legend_.remove()

###################################################
# Plot on log scale (looking for power-law
###################################################
hist_plot(axs[2], deg_dists, colors, labels)
hist_plot(axs[2], [brain_degree], [BRAIN_COLOR], [label_brain])

axs[2].set_xlim([1, 150])
axs[2].set_ylim([10E-4, 1])
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_xlabel('Log[Degree]')
axs[2].set_ylabel('Log[P(k)]')
axs[2].legend_.remove()

plt.tight_layout()
fig.subplots_adjust(wspace=0.325, top=.925, bottom=.17)
plt.draw()
