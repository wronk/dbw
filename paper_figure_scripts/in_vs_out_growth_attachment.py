"""
Created on Fri Jan 23 13:11:36 2015

@author: rkp, wronk

Plot the in- vs. out-degree distribution for the pref attachment and pref
growth models in the same plot.
"""

import numpy as np
import matplotlib.pyplot as plt

from random_graph.binary_directed import biophysical_indegree, biophysical_reverse_outdegree

from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

import brain_constants as bc
from mpl_toolkits.axes_grid1 import make_axes_locatable

# IMPORT PLOT PARAMETERS
import in_out_plot_config as cf

plt.close('all')
plt.ion()

# PLOT PARAMETERS
FACECOLOR = 'black'
ATTACHMENTCOLOR = 'y'
GROWTHCOLOR = 'c'
FONTSIZE = 24
NBINS = 15

# create attachment and growht models
Gattachment, _, _ = biophysical_indegree(N=bc.num_brain_nodes,
                                         N_edges=bc.num_brain_edges_directed,
                                         L=np.inf, gamma=1.)

Ggrowth, _, _ = biophysical_reverse_outdegree(N=bc.num_brain_nodes,
                                              N_edges=bc.num_brain_edges_directed,
                                              L=np.inf, gamma=1.)

# Get in- & out-degree
indeg_attachment = np.array([Gattachment.in_degree()[node]
                             for node in Gattachment])
outdeg_attachment = np.array([Gattachment.out_degree()[node]
                              for node in Gattachment])
deg_attachment = indeg_attachment + outdeg_attachment

indeg_growth = np.array([Ggrowth.in_degree()[node] for node in Ggrowth])
outdeg_growth = np.array([Ggrowth.out_degree()[node] for node in Ggrowth])
deg_growth = indeg_growth + outdeg_growth

# Calculate proportion in degree
percent_indeg_attachment = indeg_attachment / deg_attachment.astype(float)
percent_indeg_growth = indeg_growth / deg_growth.astype(float)

# Create figure
fig = plt.figure(figsize=(12, 6.5), facecolor=cf.FACECOLOR, tight_layout=True)
ax0 = plt.subplot2grid(cf.SUBPLOT_DIVISIONS, cf.AX0_LOCATION,
                       colspan=cf.AX0_COLSPAN)
ax1 = plt.subplot2grid(cf.SUBPLOT_DIVISIONS, cf.AX1_LOCATION,
                       colspan=cf.AX1_COLSPAN)

# Add new axes for histograms in margins
divider0 = make_axes_locatable(ax0)

ax0_histTop = divider0.append_axes('top', 1.0, pad=0.4, sharex=ax0)
ax0_histRight = divider0.append_axes('right', 2.0, pad=0.4, sharey=ax0)

##########################################################################
# Call plotting function for scatter/marginal histograms (LEFT SIDE)
al = .6
legend = ['Preferential\nattachment', 'Preferential\ngrowth']

ax0.scatter(indeg_attachment, outdeg_attachment, c=ATTACHMENTCOLOR,
            s=cf.MARKERSIZE, lw=0, alpha=al, label=legend[0])
ax0.scatter(indeg_growth, outdeg_growth, c=GROWTHCOLOR, s=cf.MARKERSIZE, lw=0,
            alpha=al, label=legend[1])

ax0.set_xlabel('In-degree')
ax0.set_ylabel('Out-degree')

#ax0_histTop.set_title('In- vs. Out-degree', va='bottom')
ax0.set_xlim([0, 100])
ax0.set_ylim([0, 100])
ax0.set_aspect('auto')
ax0.set_xticks(np.arange(0, 101, 25))
ax0.set_yticks(np.arange(0, 101, 25))
ax0.legend(loc='best')

############
# Marginals
############
ax0_histTop.hist(indeg_attachment, bins=cf.OUTDEGREE_BINS,
                 histtype='stepfilled', color=ATTACHMENTCOLOR, alpha=al)
ax0_histTop.hist(indeg_growth, bins=cf.OUTDEGREE_BINS, histtype='stepfilled',
                 color=GROWTHCOLOR, alpha=al)

ax0_histRight.hist(outdeg_growth, bins=cf.OUTDEGREE_BINS,
                   orientation='horizontal', histtype='stepfilled',
                   color=GROWTHCOLOR, alpha=al)
ax0_histRight.hist(outdeg_attachment, bins=cf.OUTDEGREE_BINS,
                   orientation='horizontal', histtype='stepfilled',
                   color=ATTACHMENTCOLOR, alpha=al)


plt.setp(ax0_histTop.get_xticklabels() + ax0_histRight.get_yticklabels(),
         visible=False)

ax0_histTop.set_ylabel('Count')
ax0_histRight.set_xlabel('Count')

ax0_histTop.set_yticks(np.arange(0, 301, 100))
ax0_histRight.set_xticks(np.arange(0, 301, 100))

##########################################################################
# Plot percent_indeg vs. degree (RIGHT SIDE)
ax1.scatter(deg_attachment, percent_indeg_attachment, s=cf.MARKERSIZE, lw=0,
            c=ATTACHMENTCOLOR, alpha=al)
ax1.scatter(deg_growth, percent_indeg_growth, s=cf.MARKERSIZE, lw=0,
            c=GROWTHCOLOR, alpha=al)
ax1.set_xlabel('Total degree (in + out)')
ax1.set_ylabel('Proportion in-degree')
ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
ax1.set_yticks(np.arange(0, 1.1, .2))
ax1.set_ylim([0., 1.05])
ax1.set_xticks(np.arange(0, 151, 50))

##########################################################################
# Set background color and text size for all spines/ticks
for temp_ax in [ax0, ax0_histRight, ax0_histTop, ax1]:
    set_all_text_fontsizes(temp_ax, cf.FONTSIZE)
    set_all_colors(temp_ax, cf.LABELCOLOR)
    #temp_ax.patch.set_facecolor(FACECOLOR)  # Set color of plot area
    temp_ax.tick_params(width=cf.TICKSIZE)

plt.show()
