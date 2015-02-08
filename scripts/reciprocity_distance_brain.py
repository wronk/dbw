from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import graph_tools.auxiliary as aux_tools

from numpy import concatenate as cc

from extract.brain_graph import binary_directed as brain_graph
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

import aux_random_graphs


# load brain graph, adjacency matrix, and labels
G, A, labels = brain_graph()

centroids = aux_random_graphs.get_coords()

# get in & out degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg
deg_diff = outdeg - indeg

edges = {} # distances

for i in centroids:
    for j in centroids:
        current = tuple(sorted([i,j]))
        edges[current] = np.sqrt(np.sum((centroids[i] - centroids[j])**2))



bins = np.linspace(0,130,40)
binnedDistances = np.histogram(edges.values(),bins)

A2 = A.copy()
np.fill_diagonal(A2,False)
recip = A2 * A2.T

i,j = np.where(recip)
centroidNames = labels
recipNames = [tuple(sorted([centroidNames[i[k]],centroidNames[j[k]]])) for k in range(len(i))]

recipDistances = [edges[k] for k in recipNames]

binnedRecipDistances = np.histogram(recipDistances,bins,normed=True)

dx = bins[1]-bins[0]
proportionRecip = binnedRecipDistances[0]
#/(dx*binnedDistances[0]*2)
#proportionRecip = binnedRecipDistances[0]/(dx*np.sum(binnedRecipDistances[0]))

yticks = np.linspace(0,0.035,8)
fig,axs = plt.subplots(1,facecolor='white')
axs.hist(edges.values(),bins,normed=True)
axs.plot(bins[0:len(bins)-1],proportionRecip,linewidth=3)
axs.set_xticklabels(np.round(bins*0.1,decimals=2),fontsize=24)
axs.set_yticks(yticks)
axs.set_yticklabels(yticks,fontsize=24)
axs.set_xlabel('Distance (mm)', fontsize=28)
axs.set_ylabel('Density', fontsize=28)


leg =axs.legend(['Reciprocal edges', 'All edges'],prop={'size':24})
#fig,axs = plt.subplots(1,facecolor='white')
plt.show(block=False)
