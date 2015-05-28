from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import graph_tools.auxiliary as aux_tools
import networkx as nx

from numpy import concatenate as cc

from extract.brain_graph import binary_directed as brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

import aux_random_graphs

from brain_constants import *

from network_compute import reciprocity

# load brain graph, adjacency matrix, and labels


labelsize = 24
ticksize = 20
legendsize = 18


### First we do stuff for the brain
# load brain graph, adjacency matrix, and labels
G, A, labels = brain_graph()
label_mapping = {k:labels[k] for k in range(len(labels))}
G = nx.relabel_nodes(G,label_mapping)

centroidsUncorrected = aux_random_graphs.get_coords()
centroids = {k:centroidsUncorrected[k]/10 for k in centroidsUncorrected} # This is because the coordinates are off by a factor of 10

# get in & out degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg
deg_diff = outdeg - indeg

edges = {} # distances
actualEdges = G.edges()

for edge in actualEdges:
    edges[edge] =  np.sqrt(np.sum((centroids[edge[0]] - centroids[edge[1]])**2))

bins = np.linspace(0,10,40)
binnedDistances = np.histogram(edges.values(),bins)

A2 = A.copy()
np.fill_diagonal(A2,False)
recip = A2 * A2.T

i,j = np.where(recip)
centroidNames = labels
recipNames = [tuple([centroidNames[i[k]],centroidNames[j[k]]]) for k in range(len(i))]

recipDistances = [edges[k] for k in recipNames]

binnedRecipDistances = np.histogram(recipDistances,bins,normed=True)

dx = bins[1]-bins[0]
proportionRecip = binnedRecipDistances[0]
#/(dx*binnedDistances[0]*2)
#proportionRecip = binnedRecipDistances[0]/(dx*np.sum(binnedRecipDistances[0]))

yticks = [0,0.15,0.3,0.45,0.6,0.75]
xticks = [0,2.5,5.0,7.5,10.0]

fig,axs = plt.subplots(nrows=1,ncols=2,facecolor='white')
#fig.tight_layout(pad=5)
axs[0].hist(edges.values(),bins,normed=True,facecolor=[0.8,0.1,0.15])
axs[0].plot(bins[0:len(bins)-1],proportionRecip,linewidth=3,color=[0.1,0.1,0.8],linestyle='-')
axs[0].set_xticks(xticks)
axs[0].set_xticklabels(xticks,fontsize=ticksize)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels(yticks,fontsize=ticksize)
axs[0].set_xlabel('Distance (mm)', fontsize=labelsize)
axs[0].set_ylabel('Density', fontsize=labelsize)
axs[0].set_xlim([0,max(bins)])
#axs[0].legend(['Reciprocal edges', 'All edges'],prop={'size':24})




### Now for the model...
G, A, labels = biophysical_model(N=num_brain_nodes,
                                 N_edges=num_brain_edges_directed,
                                 L=.7,
                                 gamma=1., brain_size='brain')

centroids = G.centroids

# get in & out degree
indeg = np.array([G.in_degree()[node] for node in G])
outdeg = np.array([G.out_degree()[node] for node in G])
deg = indeg + outdeg
deg_diff = outdeg - indeg

edges = {} # distances
actualEdges = G.edges()

for edge in actualEdges:
    edges[edge] =  np.sqrt(np.sum((centroids[edge[0]] - centroids[edge[1]])**2))



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


axs[1].hist(edges.values(),bins,normed=True,facecolor=[0.8,0.1,0.15])
axs[1].plot(bins[0:len(bins)-1],proportionRecip,linewidth=3,color=[0.1,0.1,0.8],linestyle='-')
axs[1].set_xticks(xticks)
axs[1].set_xticklabels(xticks,fontsize=ticksize)
axs[1].set_yticks(yticks)
axs[1].set_yticklabels([])
axs[1].set_xlabel('Distance (mm)', fontsize=labelsize)
#axs[1].set_ylabel('Density', fontsize=28)
axs[1].set_xlim([0,10.0])
axs[1].legend(['Reciprocal edges', 'All edges'],prop={'size':legendsize})

#fig,axs = plt.subplots(1,facecolor='white')
plt.show(block=False)
