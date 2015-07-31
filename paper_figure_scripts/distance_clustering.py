from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import graph_tools.auxiliary as aux_tools
import networkx as nx
import os

from numpy import concatenate as cc

from extract.brain_graph import binary_directed as brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

import aux_random_graphs

from brain_constants import *

from network_compute import reciprocity

import color_scheme

fig,axs = plt.subplots(1,3,facecolor='white',edgecolor='white',figsize=(22,6))


#fig.set_figheight(3)
#fig.set_figwidth(6)
fig.subplots_adjust(bottom=0.15,wspace=0.3)

col = ['m','cyan',[0.8,0.1,0.1]]
# load brain graph, adjacency matrix, and labels
labelsize = 24
ticksize = 20
legendsize = 18

### Now for the model...
G, A, labels = biophysical_model(N=num_brain_nodes,
                                 N_edges=num_brain_edges_directed,
                                 L=0.7,
                                 gamma=1., brain_size=[7.,7.,7.])
centroids = G.centroids
G_PGPA = G.to_undirected()
G_PGPA.centroids = centroids

### Now for the model...
G, A, labels = biophysical_model(N=num_brain_nodes,
                                 N_edges=num_brain_edges_directed,
                                 L=2.5,
                                 gamma=1., brain_size=[7.,7.,7.])
centroids = G.centroids
G_PGPA2 = G.to_undirected()
G_PGPA2.centroids = centroids

G, A, labels = brain_graph()
G = G.to_undirected()
label_mapping = {k:labels[k] for k in range(len(labels))}
G_mouse = nx.relabel_nodes(G,label_mapping)

centroidsUncorrected = aux_random_graphs.get_coords()
# This is because the coordinates are off by a factor of 10
centroids = {k:centroidsUncorrected[k]/10. for k in centroidsUncorrected}
G_mouse.centroids = centroids

Gs = [G_mouse,G_PGPA,G_PGPA2]

for g_i,G in enumerate(Gs):
    ### First we do stuff for the brain
    # load brain graph, adjacency matrix, and labels

    centroids = G.centroids
    edges = {} # distances
    nodes = G.nodes()
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


    distances = [edges[k] for k in actualEdges]


    nodeEdgeDistance = {}
    for node in nodes:
        currentEdges = G.edges(node)
        currentDistances = []
        for edge in currentEdges:
            try:
                currentDistances.append(edges[edge])
            except KeyError:
                currentDistances.append(edges[(edge[1],edge[0])])
            
        
        nodeEdgeDistance[node] = currentDistances


    nodeList = nodeEdgeDistance.keys()
    meanEdgeDistance = [np.mean(nodeEdgeDistance[node]) for node in nodeList]

    cc_all = nx.clustering(G)
    cc = [cc_all[node] for node in nodeList]
    

    axs[g_i].scatter(meanEdgeDistance,cc,color=col[g_i])

    axs[g_i].set_xlim([0,10])
    axs[g_i].set_ylim([0,1])
    axs[g_i].set_xlabel('Mean edge length (mm)',fontsize=16)

axs[0].set_ylabel('Clustering coefficient',fontsize=16)


plt.show(block=False)
