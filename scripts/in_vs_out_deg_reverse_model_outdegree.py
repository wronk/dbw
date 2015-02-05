"""
Created on Fri Jan 23 13:11:36 2015

@author: rkp

Plot the in- vs. outdegree distribution for the Allen Brain mouse connectome.
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import concatenate as cc

from random_graph.binary_directed import biophysical_reverse_outdegree_nonspatial as biophysical_model
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

from brain_constants import *

# PLOT PARAMETERS
FACECOLOR = 'white'
FONTSIZE = 14
NBINS = 15

# create model graph
G = biophysical_model(N=num_brain_nodes,
                      N_edges=num_brain_edges_directed,
                      gamma=1.)

# get in & out degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg
deg_diff = outdeg - indeg

# calculate percent in & percent out degree
percent_indeg = indeg/deg.astype(float)
percent_outdeg = outdeg/deg.astype(float)

# open figure
fig, axs = plt.subplots(2, 2, facecolor=FACECOLOR, tight_layout=True)

# plot out vs. in-degree scatter
axs[0,0].scatter(indeg, outdeg)
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
axs[1,1].hist(cc([percent_indeg[None,:], percent_outdeg[None,:]]).T, bins=NBINS)
axs[1,1].set_xlabel('% indegree (blue), % outdegree (green)')
axs[1,1].set_ylabel('# nodes')
[set_fontsize(ax, FONTSIZE) for ax in axs.flatten()]


# plot total degree, vs in-out degree difference
fig, axs = plt.subplots(2, 2, facecolor=FACECOLOR, tight_layout=True)

# plot scatter
axs[0,0].scatter(deg, deg_diff)
axs[0,0].set_xlabel('degree')
axs[0,0].set_ylabel('outdegree - indegree')

# plot distributions
axs[0,1].hist(deg_diff, bins=NBINS, orientation='horizontal')
axs[0,1].set_ylabel('outdegree - indegree')
axs[0,1].set_xlabel('# nodes')

axs[1,0].hist(deg, bins=NBINS)
axs[1,0].set_xlabel('degree')
axs[1,0].set_ylabel('# nodes')

# plot percent_indeg vs. degree
axs[1,1].scatter(deg, percent_indeg)
axs[1,1].set_xlabel('degree')
axs[1,1].set_ylabel('% indegree')
[set_fontsize(ax, FONTSIZE) for ax in axs.flatten()]

[set_fontsize(ax, FONTSIZE) for ax in axs.flatten()]