"""
Created on Sun Apr 5 17:11:50 2015

@author: wronk

Plot the in- vs. outdegree distribution for the directed ER plot
that has matched in/out degree with the brain.
"""

import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

from mpl_toolkits.axes_grid1 import make_axes_locatable

from network_plot.network_viz import plot_scatterAndMarginal
import extract.brain_graph

from random_graph import binary_directed

###############################
# Parameters
###############################
import in_out_plot_config as cf  # plot parameters

# SET YOUR SAVE DIRECTORY
save_dir = '/home/wronk/Documents/dbw_figs/'

plt.close('all')
plt.ion()

# PLOT PARAMETERS
FACECOLOR = 'white'
MARKERCOLOR = 'r'
FONTSIZE = 16
NBINS = 15

###############################
# Create graph/ compute metrics
###############################

# generate erdos-renyi graph
G_brain, _, _ = extract.brain_graph.binary_directed()

# Error check that keys all match (in_seq is in the same order as out_seq)
assert G_brain.in_degree().keys() == G_brain.out_degree().keys()

# Get in and out degree for each node and construct constrained random graph
brain_indeg = G_brain.in_degree().values()
brain_outdeg = G_brain.out_degree().values()
G, _, _ = binary_directed.random_directed_deg_seq(brain_indeg, brain_outdeg,
                                                  simplify=True)

# Get in & out degree of random directed graph (both should match brain)
indeg = np.array(G.in_degree().values())
outdeg = np.array(G.out_degree().values())
deg = indeg + outdeg

# Calculate proportion in degree
proportion_indeg = indeg / deg.astype(float)

###########
# Plotting
###########

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

divider1 = make_axes_locatable(ax1)
ax1_right = divider1.append_axes('right', 1.0, pad=0.3, sharey=ax1)

##########################################################################
# Call plotting function for scatter/marginal histograms (LEFT SIDE)
plot_scatterAndMarginal(ax0, ax0_histTop, ax0_histRight, indeg, outdeg,
                        bin_width=cf.BINWIDTH, marker_size=cf.MARKERSIZE,
                        marker_color=MARKERCOLOR,
                        indegree_bins=cf.INDEGREE_BINS,
                        outdegree_bins=cf.OUTDEGREE_BINS)

ax0.set_xlabel('Indegree')
ax0.set_ylabel('Outdegree')
#ax0_histTop.set_title('In- vs. Outdegree', va='bottom')
ax0.set_xlim(*cf.IN_OUT_SCATTER_XLIM)
ax0.set_ylim(*cf.IN_OUT_SCATTER_YLIM)
ax0.set_aspect('auto')
ax0.set_yticks(np.arange(0, 121, 30))

ax0_histTop.set_ylabel('# Nodes')
ax0_histRight.set_xlabel('# Nodes')

##########################################################################
# Plot proportion vs. degree (RIGHT SIDE)
ax1.scatter(deg, proportion_indeg, s=cf.MARKERSIZE, lw=0, c=MARKERCOLOR)
ax1.set_xlabel('Total degree (in + out)')
ax1.set_ylabel('Proportion indegree')
ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
ax1.set_yticks(np.arange(0, 1.1, .2))
#ax1.set_title('Incoming edge proportion vs. degree',
#              fontsize=cf.FONTSIZE + 2, va='bottom')
ax1.set_xlim([0, 150])
ax1.set_ylim([0., 1.05])
ax1.set_xticks(np.arange(0, 121, 30))

ax1_right.hist(proportion_indeg, orientation='horizontal', color=MARKERCOLOR)
ax1_right.set_xticks([0, 40, 80])
plt.setp(ax1_right.get_yticklabels(), visible=False)
ax1_right.set_xlabel('# Nodes')
##########################################################################
# Set background color and text size for all spines/ticks
for temp_ax in [ax0, ax0_histRight, ax0_histTop, ax1, ax1_right]:
    set_all_text_fontsizes(temp_ax, cf.FONTSIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)
    #temp_ax.patch.set_facecolor(FACECOLOR)  # Set color of plot area
    temp_ax.tick_params(width=cf.TICKSIZE)

fig.savefig(op.join(save_dir, 'ER_degConstrained_in_vs_out_simplified.png'))
plt.show()
