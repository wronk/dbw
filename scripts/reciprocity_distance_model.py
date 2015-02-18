from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import graph_tools.auxiliary as aux_tools

from numpy import concatenate as cc

#from extract.brain_graph import binary_directed as brain_graph
from random_graph.binary_directed import biophysical_out as biophysical_model
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

import aux_random_graphs

from brain_constants import *

from network_compute import reciprocity

# load brain graph, adjacency matrix, and labels

G, A, labels = biophysical_model(N=num_brain_nodes,
                                 N_edges=num_brain_edges_directed,
                                 L=.7,
                                 gamma=1.)

centroids = G.centroids

# get in & out degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg
deg_diff = outdeg - indeg

edges = {} # distances

for i in range(len(centroids)):
    for j in range(len(centroids)):
        current = tuple(sorted([i,j]))
        
        edges[current] = np.sqrt(np.sum((centroids[i] - centroids[j])**2))


bins = np.linspace(0,10,40)
binnedDistances = np.histogram(edges.values(),bins)

A2 = A.copy()
np.fill_diagonal(A2,False)
recip = A2 * A2.T

i,j = np.where(recip)
centroidNames = np.arange(len(centroids))
recipNames = [tuple(sorted([centroidNames[i[k]],centroidNames[j[k]]])) for k in range(len(i))]

recipDistances = [edges[k] for k in recipNames]

binnedRecipDistances = np.histogram(recipDistances,bins,normed=True)
dx = bins[1]-bins[0]
# Every reciprocal connection is counted twice, so divide by 2
#proportionRecip = binnedRecipDistances[0]/(np.sum(binnedRecipDistances[0])*dx)
proportionRecip = binnedRecipDistances[0]
#/(binnedDistances[0]*2*dx)

yticks = np.linspace(0,0.7,8)

fig,axs = plt.subplots(1,facecolor='white')
axs.hist(edges.values(),bins,normed=True)
axs.plot(bins[0:len(bins)-1],proportionRecip,linewidth=3)
axs.set_xticklabels(np.round(bins*0.1,decimals=2),fontsize=24)
axs.set_yticks(yticks)
axs.set_yticklabels(yticks,fontsize=24)
axs.set_xlabel('Distance', fontsize=28)
axs.set_ylabel('Count', fontsize=28)

axs.legend(['Reciprocal edges', 'All edges'],prop={'size':24})

#fig,axs = plt.subplots(1,facecolor='white')
plt.show(block=False)
