"""
Created on Sun May 17, 2015

@author: wronk

Plot the in- vs. outdegree distribution for the Allen Brain mouse connectome
for paper draft.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from extract.brain_graph import binary_directed

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from network_plot.network_viz import plot_scatterAndMarginal

# IMPORT PLOT PARAMETERS
import in_out_plot_config as cf

plt.close('all')
plt.ion()

# PLOT PARAMETERS
FONTSIZE = 24
NBINS = 15
MARKERCOLOR = 'm'

# load brain graph, adjacency matrix, and labels
G, A, labels = binary_directed()

# Get in & out degree
indeg = G.in_degree().values()
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg

# Calculate proportion in degree
percent_indeg = indeg / deg.astype(float)

# Create figure
fig = plt.figure(figsize=cf.FIGSIZE, facecolor=cf.FACECOLOR, tight_layout=True)
ax0 = plt.subplot2grid((1, 1), (0, 0))
#ax0 = plt.subplot2grid(cf.SUBPLOT_DIVISIONS, cf.AX0_LOCATION,
#                       colspan=cf.AX0_COLSPAN)

# Add new axes for histograms in margins
divider0 = make_axes_locatable(ax0)

ax0_histTop = divider0.append_axes('top', size=2.0, pad=0.3, sharex=ax0)
ax0_histRight = divider0.append_axes('right', size=2.0, pad=0.3, sharey=ax0)

'''
ax1 = plt.subplot2grid(cf.SUBPLOT_DIVISIONS, cf.AX1_LOCATION,
                       colspan=cf.AX1_COLSPAN)
divider1 = make_axes_locatable(ax1)
ax1_right = divider1.append_axes('right', 1.0, pad=0.3, sharey=ax1)
'''
##########################################################################
# Call plotting function for scatter/marginal histograms (LEFT SIDE)
plot_scatterAndMarginal(ax0, ax0_histTop, ax0_histRight, indeg, outdeg,
                        bin_width=cf.BINWIDTH, marker_size=cf.MARKERSIZE,
                        marker_color=MARKERCOLOR,
                        indegree_bins=cf.INDEGREE_BINS,
                        outdegree_bins=cf.OUTDEGREE_BINS, log_probs=True)

ax0.set_xlabel('Indegree')
ax0.set_ylabel('Outdegree')
#ax0_histTop.set_title('In- vs. Outdegree', va='bottom')
ax0.set_xlim(*cf.IN_OUT_SCATTER_XLIM)
ax0.set_ylim(*cf.IN_OUT_SCATTER_YLIM)
ax0.set_aspect('auto')
ax0.set_yticks(np.arange(0, 121, 30))

ax0_histTop.set_ylabel('P(k_in)')
ax0_histRight.set_xlabel('P(k_out)')
ax0_histRight.set_xlim([5E-4, 1.])
ax0_histTop.set_ylim([5E-4, 1.])

'''
##########################################################################
# Plot percent_indeg vs. degree (RIGHT SIDE)
ax1.scatter(deg, percent_indeg, s=cf.MARKERSIZE, lw=0, c=MARKERCOLOR)
ax1.set_xlabel('Total degree (in + out)')
ax1.set_ylabel('Proportion indegree')
ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
ax1.set_yticks(np.arange(0, 1.1, .2))
#ax1.set_title('Incoming edge proportion vs. degree',
#              fontsize=cf.FONTSIZE + 2, va='bottom')
ax1.set_ylim([0., 1.05])
ax1.set_xticks(np.arange(0,121,30))

ax1_right.hist(percent_indeg, orientation='horizontal', color=MARKERCOLOR)
ax1_right.set_xticks([0,30,60])
plt.setp(ax1_right.get_yticklabels(), visible=False)
ax1_right.set_xlabel('# Nodes')
'''
##########################################################################
# Set background color and text size for all spines/ticks
for temp_ax in [ax0, ax0_histRight, ax0_histTop]:
    set_all_text_fontsizes(temp_ax, cf.FONTSIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)
    #temp_ax.patch.set_facecolor(FACECOLOR)  # Set color of plot area
    temp_ax.tick_params(width=cf.TICKSIZE)

#fig.savefig('/Users/richpang/Desktop/brain_in_out.png', transparent=True)
plt.show()
