"""
Created on Fri Jan 23 13:11:36 2015

@author: rkp

Plot the in- vs. out-degree distribution for the reverse outdegree model.
"""

import numpy as np
import matplotlib.pyplot as plt

from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network_plot.network_viz import plot_scatterAndMarginal

# IMPORT PLOT PARAMETERS
import in_out_plot_config as cf

plt.ion()

# PLOT PARAMETERS
FACECOLOR = 'black'
MARKERCOLOR = 'c'
FONTSIZE = 16
NBINS = 15

# create model graph
G, A, D = biophysical_model(N=bc.num_brain_nodes,
                            N_edges=bc.num_brain_edges_directed,
                            L=.725,
                            gamma=1.)

# Get in & out degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg

# Calculate proportion in degree
percent_indeg = indeg / deg.astype(float)

# Create figure
fig = plt.figure(figsize=cf.FIGSIZE, facecolor=cf.FACECOLOR, tight_layout=True)
ax0 = plt.subplot2grid(cf.SUBPLOT_DIVISIONS, cf.AX0_LOCATION,
                       colspan=cf.AX0_COLSPAN)
ax1 = plt.subplot2grid(cf.SUBPLOT_DIVISIONS, cf.AX1_LOCATION,
                       colspan=cf.AX1_COLSPAN)

# Add new axes for histograms in margins
divider0 = make_axes_locatable(ax0)

ax0_histTop = divider0.append_axes('top', 1.0, pad=0.3, sharex=ax0)
ax0_histRight = divider0.append_axes('right', 2.0, pad=0.3, sharey=ax0)

##########################################################################
# Call plotting function for scatter/marginal histograms (LEFT SIDE)
plot_scatterAndMarginal(ax0, ax0_histTop, ax0_histRight, indeg, outdeg,
                        bin_width=cf.BINWIDTH, marker_size=cf.MARKERSIZE,
                        marker_color=MARKERCOLOR,
                        indegree_bins=cf.INDEGREE_BINS,
                        outdegree_bins=cf.OUTDEGREE_BINS)

ax0.set_xlabel('In-degree')
ax0.set_ylabel('Out-degree')
ax0.set_xlim(*cf.IN_OUT_SCATTER_XLIM)
ax0.set_ylim(*cf.IN_OUT_SCATTER_YLIM)
ax0.set_aspect('auto')
ax0.set_yticks(np.arange(0, 121, 30))

ax0_histTop.set_ylabel('# Nodes')
ax0_histRight.set_xlabel('# Nodes')

##########################################################################
# Plot percent_indeg vs. degree (RIGHT SIDE)
ax1.scatter(deg, percent_indeg, s=cf.MARKERSIZE, lw=0, c=MARKERCOLOR)
ax1.set_xlabel('Total degree (in + out)')
ax1.set_ylabel('Proportion in-degree')
ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
ax1.set_yticks(np.arange(0, 1.1, .2))
ax1.set_xlim([0, 150])
ax1.set_ylim([0., 1.05])
ax1.set_xticks(np.arange(0, 151, 50))

##########################################################################
# Set background color and text size for all spines/ticks
for temp_ax in [ax0, ax0_histRight, ax0_histTop, ax1]:
    set_all_text_fontsizes(temp_ax, cf.FONTSIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)
    #temp_ax.patch.set_facecolor(FACECOLOR)  # Set color of plot area
    temp_ax.tick_params(width=cf.TICKSIZE)

plt.show(block=True)
