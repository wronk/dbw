"""
Created on Fri May 15th 2015

@author: wronk

Plot undirected degree dists for undirected model with different gammas
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import extract.brain_graph
import random_graph.binary_undirected as rg

from network_plot.change_settings import set_all_text_fontsizes

assert mpl.__version__ == '1.4.3', 'Matplotlib version must be 1.4.3!'

###############################################
# Plotting params
###############################################
FACECOLOR = 'white'
FONTSIZE = 24

GAMMAS = [1.5, 1.75, 2.0]  # Preferential attachment parameters

# Preferential attachment parameters
BRAIN_SIZE = [7., 7., 7.]  # Size brain region volume to distribute nodes
colors = ['0.25', '0.5', '0.75']  # Colors for each gamma
alphas = [0.5, 0.75, 1.]
alphas = [1., 1., 1.]
labels = ['a', 'b', 'c']
BRAIN_COLOR = 'm'
titles = ['Degree Distributions', 'Degree Distributions']

plt.ion()
plt.close('all')
plt.rcParams['ps.fonttype'] = 42  # For adobe illustrator
plt.rcParams['pdf.fonttype'] = 42

n_bins = 50
repeats = 100

###############################################
# Histogram plot function
###############################################
# Helper function to plot a list of histograms on a single axis. Allows code
# reuse for similar plots that will have just a change in x or y scale (log)


def hist_plot(ax, deg_dists, colors, gammas, alphas, bins):
    for deg, col, g, alpha in zip(deg_dists, colors, gammas, alphas):
        hist, plt_bins = np.histogram(deg, bins, normed=True)
        ax.plot(plt_bins[:-1], hist, lw=2, color=col, label=str(g),
                alpha=alpha)

    # Set axis limits and ticks, and label subplots
    ax.set_xlim([0, 150])
    #ax.set_ylim([0, .15])
    ax.locator_params(axis='x', nbins=5)
    #a.locator_params(axis='y', nbins=5)
    ax.legend(loc='best', fontsize=FONTSIZE - 10)

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

for r in np.arange(repeats):
    for gamma_idx, gamma in enumerate(GAMMAS):
        G_model, _, _ = rg.biophysical(n_nodes, n_edges, np.inf, gamma,
                                       BRAIN_SIZE)

        gamma_mat[gamma_idx, r, :] = G_model.degree().values()
    print 'Finished repeat: ' + str(r)

gamma_dists = gamma_mat.reshape((len(GAMMAS), -1))

###################################################
# Plot all panels that will have axis scales changes
###################################################
figsize = (11, 5)
fig, axs = plt.subplots(1, 2, figsize=figsize)
lin_bins = np.linspace(0, 150, n_bins)

for ax_i, ax in enumerate(axs):
    ax.hist(brain_degree, lin_bins, color=BRAIN_COLOR, normed=True,
            label='Mouse\nConnectome', lw=0, alpha=.4, histtype='stepfilled')
    #hist_plot(ax, gamma_dists, ['c'] * 4, GAMMAS, alphas, lin_bins)
    hist_plot(ax, gamma_dists, colors, GAMMAS, [1] * 4, lin_bins)

    # Set title
    ax.set_title(titles[ax_i])
    ax.text(0.05, .95, labels[ax_i], color='k', fontsize=FONTSIZE - 2,
            fontweight='bold', transform=ax.transAxes)


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

'''
# Plot on log scale (looking for power-law
axs[2].set_xlim([1, 150])
axs[2].set_ylim([10E-5, 1])
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_xlabel('Degree')
axs[2].set_ylabel('P(k)')
axs[2].legend_.remove()
'''

plt.tight_layout()
fig.subplots_adjust(wspace=0.325, top=.85, bottom=.17)
plt.draw()
