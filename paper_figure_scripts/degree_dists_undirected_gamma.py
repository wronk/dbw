"""
Created on Fri May 15th 2015

@author: wronk

Plot undirected degree dists for undirected model with different gammas
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import extract.brain_graph
import random_graph.binary_undirected as rg

from network_plot.change_settings import set_all_text_fontsizes

###############################################
# Plotting params
###############################################
FACECOLOR = 'white'
FONTSIZE = 24

GAMMAS = [1., 1.33, 1.67, 2.0]  # Preferential attachment parameters
#GAMMAS = [1.33, 1.67]  # Preferential attachment parameters
BRAIN_SIZE = [7., 7., 7.]  # Size brain region volume to distribute nodes
colors = ['0.1', '0.25', '0.5', '0.75']
alphas = [0.55, 0.7, 0.85, 1.]
BRAIN_COLOR = 'm'

plt.ion()
plt.close('all')
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['pdf.fonttype'] = 42

n_bins = 50
repeats = 1

###############################################
# Histogram plot function
###############################################
# Function to plot a list of histograms on a single axis. Allows code to be
# reused for similar plots that will have just a change in x or y scale (log)


def hist_plot(ax, deg_dists, colors, gammas, alphas):
    for deg, col, g, alpha in zip(deg_dists, colors, gammas, alphas):
        hist, plt_bins = np.histogram(deg, lin_bins, normed=True)
        ax.plot(plt_bins[:-1], hist, lw=2, color=col, label=str(g),
                alpha=alpha)

    # Set axis limits and ticks, and label subplots
    ax.set_xlim([0, 150])
    #ax.set_ylim([0, .15])
    ax.locator_params(axis='x', nbins=5)
    #a.locator_params(axis='y', nbins=5)
    ax.legend(loc='best', fontsize=FONTSIZE - 10)

    # Set title
    ax.set_title('Model Degree\nDistributions')

    # Set xlabels
    ax.set_xlabel('Degree')
    ax.set_ylabel('P(k)')

    # Set all fontsizes and axis colors
    set_all_text_fontsizes(ax, FONTSIZE)
    #set_all_colors(ax, 'w')

###############################################
# Calculate degree distributions for all graphs
###############################################

G_brain, _, _ = extract.brain_graph.binary_undirected()
n_nodes = len(G_brain.nodes())
n_edges = len(G_brain.edges())
brain_degree = G_brain.degree().values()

# Initialize repetition matrices for standard graphs
gamma_mat = -1 * np.ones((len(GAMMAS), repeats, n_nodes))

for gamma_idx, gamma in enumerate(GAMMAS):
    for r in np.arange(repeats):
        G_model, _, _ = rg.biophysical(n_nodes, n_edges, np.inf, gamma,
                                       BRAIN_SIZE)

        gamma_mat[gamma_idx, r, :] = G_model.degree().values()

gamma_dists = gamma_mat.reshape((len(GAMMAS), -1))

###################################################
# Plot all panels that will have axis scales changes
###################################################
figsize = (16, 5)
fig, axs = plt.subplots(1, 3, figsize=figsize)
lin_bins = np.linspace(0, 150, n_bins)

for ax in axs:
    ax.hist(brain_degree, lin_bins, color=BRAIN_COLOR, normed=True,
            label='Mouse\nConnectome', lw=0.5, alpha=.8)
    hist_plot(ax, gamma_dists, ['c'] * 4, GAMMAS, alphas)

###################################################
# Plot specific scales
###################################################

# Plot semilogy (looking for exponential solutions)
axs[0].locator_params(axis='y', nbins=6)

# Plot semilogy (looking for exponential solutions)
axs[1].set_ylim([10E-5, 1])
axs[1].set_yscale('log')
axs[1].set_ylabel('P(k)')
axs[1].legend_.remove()

# Plot on log scale (looking for power-law
axs[2].set_xlim([1, 150])
axs[2].set_ylim([10E-5, 1])
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_xlabel('Degree')
axs[2].set_ylabel('P(k)')
axs[2].legend_.remove()

plt.tight_layout()
fig.subplots_adjust(wspace=0.325, top=.85, bottom=.17)
plt.draw()
