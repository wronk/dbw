"""
Created on Fri Jan 23 13:11:36 2015

@author: rkp

Plot the in- vs. outdegree distribution for the Allen Brain mouse connectome.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from extract.brain_graph import binary_directed as brain_graph

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors
from brain_constants import *

import brain_constants as bc
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network_plot.network_viz import plot_scatterAndMarginal

# IMPORT PLOT PARAMETERS
import in_out_plot_config as cf

#plt.close('all')
plt.ion()

# PLOT PARAMETERS
FACECOLOR = 'black'
FONTSIZE = 16
NBINS = 15

# load brain graph, adjacency matrix, and labels
G, A, labels = brain_graph()

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
divider = make_axes_locatable(ax0)

ax0_histTop = divider.append_axes('top', 1.0, pad=0.3, sharex=ax0)
ax0_histRight = divider.append_axes('right', 2.0, pad=0.3, sharey=ax0)


##########################################################################
# Call plotting function for scatter/marginal histograms (LEFT SIDE)
plot_scatterAndMarginal(ax0, ax0_histTop, ax0_histRight, indeg, outdeg,
                        bin_width=BINWIDTH, marker_size=MARKERSIZE,
                        marker_color='m', indegree_bins=cf.INDEGREE_BINS,
                        outdegree_bins=cf.OUTDEGREE_BINS)

ax0.set_xlabel('Indegree')
ax0.set_ylabel('Outdegree')
ax0_histTop.set_title('In- vs. Outdegree', fontsize=FONTSIZE + 2,
                      va='bottom')
ax0.set_xlim(*cf.IN_OUT_SCATTER_XLIM)
ax0.set_ylim(*cf.IN_OUT_SCATTER_YLIM)
ax0.set_aspect('auto')

##########################################################################
# Plot percent_indeg vs. degree (RIGHT SIDE)
ax1.scatter(deg, percent_indeg, s=MARKERSIZE, lw=0, c='m')
ax1.set_xlabel('Total degree (in + out)')
ax1.set_ylabel('Proportion in-degree')
ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
ax1.set_yticks(np.arange(0, 1.1, .2))
ax1.set_title('Proportion of Edges that are Incoming\nvs. Degree',
              fontsize=cf.FONTSIZE + 2, va='bottom')
ax1.set_ylim([0., 1.05])

##########################################################################
# Set background color and text size for all spines/ticks
for temp_ax in [ax0, ax0_histRight, ax0_histTop, ax1]:
    set_all_text_fontsizes(temp_ax, FONTSIZE)
    set_all_colors(temp_ax, LABELCOLOR)
    #temp_ax.patch.set_facecolor(FACECOLOR)  # Set color of plot area
    temp_ax.tick_params(width=TICKSIZE)

#fig.savefig('/home/wronk/Builds/fig_save.png', transparent=True)
plt.show()