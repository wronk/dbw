"""
Created on Wed May 13th 2015

@author: wronk
degree_dists_undirected.py

Plot undirected degree distributions for brain and standard graphs
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
import config

from network_plot.change_settings import set_all_text_fontsizes

#from networkx import erdos_renyi_graph as er
from networkx import barabasi_albert_graph as ba
from networkx import watts_strogatz_graph as ws

###############################################
# Plotting params
###############################################
FACECOLOR = config.FACE_COLOR
FONT_SIZE = config.FONT_SIZE

BRAIN_COLOR = config.COLORS['brain']
RAND_COLOR = config.COLORS['configuration']
WS_COLOR = config.COLORS['small-world']
BA_COLOR = config.COLORS['scale-free']
labels = ['a', 'b']

plt.ion()
plt.close('all')
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42

n_bins = 50
repeats = 1

hist_brain = True  # Plot brain as histogram bars? Otherwise, as line
###############################################
# Histogram plot function
###############################################
# Function to plot a list of histograms on a single axis. Allows code to be
# reused for similar plots that will have just a change in x or y scale (log)


def hist_plot(ax, deg_dists, colors, graph_names, graph_ls):

    # Loop through all desired graphs
    for deg, col, lab, ls in zip(deg_dists, colors, graph_names, graph_ls):
        hist, plt_bins = np.histogram(deg, lin_bins, normed=True)
        # Correct x-coords to center them on histogram bins
        plt_bins_mid = plt_bins + np.diff(plt_bins)[0] * 0.5

        ax.plot(plt_bins_mid[:-1], hist, ls=ls, lw=3, color=col, label=lab)

    # Set axis limits and ticks, and label subplots
    ax.set_xlim([0, 150])
    #ax.set_ylim([0, .15])
    ax.locator_params(axis='x', nbins=5)
    #a.locator_params(axis='y', nbins=5)

    # Set title
    ax.set_title('Degree Distributions')

    # Set xlabels
    ax.set_xlabel('Degree')
    ax.set_ylabel('P(k)')

    # Set all fontsizes and axis colors
    set_all_text_fontsizes(ax, FONT_SIZE)
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
#ER_deg_mat = -1 * np.ones((repeats, n_nodes))
RAND_deg_mat = -1 * np.ones((repeats, n_nodes))
WS_deg_mat = -1 * np.ones((repeats, n_nodes))
BA_deg_mat = -1 * np.ones((repeats, n_nodes))

for r in np.arange(repeats):
    # Erdos-Renyi (pure random)
    #ER_deg_mat[r, :] = er(n_nodes, edge_density).degree().values()

    # Degree controlled random
    RAND_deg_mat[r, :] = nx.random_degree_sequence_graph(
        brain_degree, tries=100).degree().values()

    # Watts-Strogatz
    WS_deg_mat[r, :] = ws(n_nodes, int(round(brain_degree_mean)),
                          0.23).degree().values()

    # Barabasi-Albert
    BA_deg_mat[r, :] = ba(n_nodes,
                          int(round(brain_degree_mean / 2.))).degree().values()
    print 'Finished repeat: ' + str(r)

deg_dists = [RAND_deg_mat.flatten(), WS_deg_mat.flatten(),
             BA_deg_mat.flatten()]
colors = [RAND_COLOR, WS_COLOR, BA_COLOR]
graph_names = ['Random', 'Small-world', 'Scale-free']
graph_ls = ['-', '-', '-']

brain_label = 'Connectome'
brain_lw = 0.5
brain_alpha = 0.7

###################################################
# Make all plots
###################################################
lin_bins = np.linspace(0, 150, n_bins)
figsize = (12, 5)
fig, axs = plt.subplots(1, 2, figsize=figsize)

for ax_i, ax in enumerate(axs):

    # Plot brain histogram depending on if hist bars are wanted or not
    if hist_brain:
        ax.hist(brain_degree, lin_bins, normed=True, lw=brain_lw,
                color=BRAIN_COLOR, label=brain_label,
                alpha=brain_alpha)
    else:
        hist_plot(ax, [brain_degree], [BRAIN_COLOR], [brain_label])

    # Plot all std graphs
    hist_plot(ax, deg_dists, colors, graph_names, graph_ls)

    ax.legend(loc='best', fontsize=FONT_SIZE - 6)
    ax.text(.04, .92, labels[ax_i], color='k', fontsize=FONT_SIZE,
            fontweight='bold', transform=ax.transAxes)

###################################################
# Change scale for all plots
###################################################
# Plot linear brain degrees
axs[0].locator_params(axis='y', nbins=5)

# Plot semilogy (looking for exponential solutions)
axs[1].set_ylim([10E-4, 1])
axs[1].set_yscale('log')
axs[1].legend_.remove()

# Put mouse connectome legend entry on top
handles, legends = axs[0].get_legend_handles_labels()
axs[0].legend(handles[::-1], legends[::-1])

'''
# Plot on log scale (looking for power-law solutions)
axs[2].set_xlim([1, 150])
axs[2].set_ylim([10E-4, 1])
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].legend_.remove()
'''

fig.set_tight_layout(True)
fig.subplots_adjust(top=0.925, bottom=0.17, left=0.12, wspace=0.325)
plt.draw()
