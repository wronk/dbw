"""
Created on Fri Jan 23 13:11:36 2015

@author: rkp

Plot the in- vs. outdegree distribution for the Allen Brain mouse connectome.
"""

import numpy as np
import matplotlib.pyplot as plt

from extract.brain_graph import binary_directed as brain_graph
from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

# PLOT PARAMETERS
FACECOLOR = 'black'
FONTSIZE = 16
NBINS = 15

# load brain graph, adjacency matrix, and labels
G, A, labels = brain_graph()

# get in & out degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg
deg_diff = outdeg - indeg

# calculate percent in & percent out degree
percent_indeg = indeg/deg.astype(float)
percent_outdeg = outdeg/deg.astype(float)

# open figure
fig, axs = plt.subplots(2, 3, facecolor=FACECOLOR, tight_layout=True)

# plot out vs. in-degree scatter
axs[0,0].scatter(indeg, outdeg, lw=0)
axs[0,0].set_xlabel('indegree')
axs[0,0].set_ylabel('outdegree')

# plot out & in-degree distributions
axs[0,1].hist(outdeg, bins=NBINS, orientation='horizontal')
axs[0,1].set_ylabel('outdegree')
axs[0,1].set_xlabel('# nodes')

axs[1,0].hist(indeg, bins=NBINS)
axs[1,0].set_xlabel('indegree')
axs[1,0].set_ylabel('# nodes')

# plot percent_indeg & percent_outdeg distributions
axs[1,1].hist(percent_indeg, bins=NBINS)
axs[1,1].set_xlabel('% indegree')
axs[1,1].set_ylabel('# nodes')

# plot scatter
axs[0,2].scatter(deg, deg_diff, lw=0)
axs[0,2].set_xlabel('outdegree + indegree')
axs[0,2].set_ylabel('outdegree - indegree')

# plot percent_indeg vs. degree
axs[1,2].scatter(deg, percent_indeg, lw=0)
axs[1,2].set_xlabel('degree')
axs[1,2].set_ylabel('% indegree')

for ax in axs.flatten():
    set_all_text_fontsizes(ax, FONTSIZE)
    set_all_colors(ax, 'white')