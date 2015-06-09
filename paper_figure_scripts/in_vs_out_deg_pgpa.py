"""
Created on Fri Jan 23 13:11:36 2015

@author: rkp

Plot the in- vs. out-degree distribution for the reverse outdegree model.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ion()

from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc
from config.graph_parameters import LENGTH_SCALE
from mpl_toolkits.axes_grid1 import make_axes_locatable

from network_plot.network_viz import plot_scatterAndMarginal

# IMPORT PLOT PARAMETERS
from config import in_vs_out_all_plots as cf
from config import COLORS, FACE_COLOR, FONT_SIZE


# create model graph
G, A, D = biophysical_model(N=bc.num_brain_nodes,
                            N_edges=bc.num_brain_edges_directed,
                            L=LENGTH_SCALE,
                            gamma=1.)

# Get in & out degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg

# Calculate proportion in degree
percent_indeg = indeg / deg.astype(float)

# Create figure
fig = plt.figure(figsize=cf.FIG_SIZE, facecolor=FACE_COLOR, tight_layout=True)
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
                        bin_width=cf.BIN_WIDTH, marker_size=cf.MARKER_SIZE,
                        marker_color=COLORS['pgpa'],
                        indegree_bins=cf.IN_DEGREE_BINS,
                        outdegree_bins=cf.OUT_DEGREE_BINS,
                        normed=False)

ax0_histTop.set_ylim(cf.IN_DEGREE_COUNTS_LIM)
ax0_histTop.set_yticklabels(cf.IN_DEGREE_COUNTS_TICKS)

ax0_histRight.set_xlim(cf.OUT_DEGREE_COUNTS_LIM)
ax0_histRight.set_xticklabels(cf.OUT_DEGREE_COUNTS_TICKS)

ax0.set_xlabel('In-degree')
ax0.set_ylabel('Out-degree')
ax0.set_xlim(cf.IN_DEGREE_LIM)
ax0.set_xticklabels(cf.IN_DEGREE_TICKS)
ax0.set_ylim(cf.OUT_DEGREE_LIM)
ax0.set_yticklabels(cf.OUT_DEGREE_TICKS)
ax0.set_aspect('auto')

ax0_histTop.set_ylabel('# Nodes')
ax0_histRight.set_xlabel('# Nodes')

##########################################################################
# Plot percent_indeg vs. degree (RIGHT SIDE)
ax1.scatter(deg, percent_indeg, s=cf.MARKER_SIZE, lw=0, c=COLORS['pgpa'])
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
    set_all_text_fontsizes(temp_ax, FONT_SIZE)

plt.show(block=True)
