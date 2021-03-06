from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import graph_tools.auxiliary as aux_tools
import networkx as nx
import os
import pandas as pd

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

# load brain graph, adjacency matrix, and labels
labelsize = 24
ticksize = 20
legendsize = 18

### First we do stuff for the brain
# load brain graph, adjacency matrix, and labels
G, A, labels = brain_graph()
label_mapping = {k:labels[k] for k in range(len(labels))}
G = nx.relabel_nodes(G,label_mapping)
G_shuffled = nx.configuration_model(G.degree().values())

Ls = np.linspace(0,2,21)

# Load the saved reciprocity values (this is from like 200 runs of the model,
# which takes forever to run..
data_dir=os.getenv('DBW_DATA_DIRECTORY')
modelReciprocity = pd.read_csv(data_dir+'/reciprocity2.csv',
                               index_col=0).as_matrix()

meanReciprocity = np.mean(modelReciprocity,axis=1)
stdReciprocity = np.std(modelReciprocity,axis=1)

brainReciprocity = reciprocity(A)
n_repeats = 20

# Get a mean value for the shuffled connectome
configReciprocity = np.zeros([n_repeats,1])
# First get lists of in and out deg to feed to config function
nodes = G.nodes()
inDeg = G.in_degree(); inDeg = [inDeg[node] for node in nodes]
outDeg = G.out_degree(); outDeg = [outDeg[node] for node in nodes]
# Loop over 20 (to get a good mean value and SD estimate)
for j in range(n_repeats):
    G_config = nx.directed_configuration_model(inDeg,outDeg)
    A_config = nx.adjacency_matrix(G_config)
    A_configInt = np.zeros(A_config.shape,dtype=int)

    # Was having some weird difficulties with this
    A_configInt[A_config < 0.5] = 0
    A_configInt[A_config >= 0.5] = 1

    configReciprocity[j] = reciprocity(A_configInt)


# Now plot everything
# First plot mean reciprocity and fill in +/- 1 SD
axs[1].plot(Ls[1:],meanReciprocity[1:],linewidth=2,color=color_scheme.PGPA)
axs[1].fill_between(Ls[1:],meanReciprocity[1:]-stdReciprocity[1:],
                    meanReciprocity[1:]+stdReciprocity[1:],
                    facecolor=color_scheme.PGPA,
                    alpha=0.2,antialiased=True,linewidth=3,linestyle='-',
                    edgecolor=color_scheme.PGPA)

# Plot brain reciprocity
axs[1].plot([0,2],[brainReciprocity,brainReciprocity],linewidth=4,
            color=color_scheme.ATLAS,linestyle='--')

# Plot configuration reciprocity
configX = [0,2]; configY = [np.mean(configReciprocity),
                            np.mean(configReciprocity)]
axs[1].plot(configX,configY,linewidth=2,color=color_scheme.CONFIG)
axs[1].fill_between(configX,configY-np.std(configReciprocity),
                    configY+np.std(configReciprocity),
                    facecolor=color_scheme.CONFIG,alpha=0.2,antialiased=True,
                    linewidth=3,linestyle=':', edgecolor=color_scheme.CONFIG)


# Set ticks, labels, and font sizes
xticks = np.arange(0,2.5,0.5)
yticks = [0,0.1,0.2,0.3]
axs[1].set_xlabel('L (mm)',fontsize=labelsize,color='k')
axs[1].set_ylabel('Reciprocity coefficient',fontsize=labelsize,color='k')
axs[1].set_xticks(xticks)
axs[1].set_yticks(yticks)
axs[1].set_xticklabels(xticks,fontsize=labelsize,color='k')
axs[1].set_yticklabels(yticks,fontsize=labelsize,color='k')
axs[1].set_xlim([Ls[1],2])
axs[1].set_ylim([0,0.3])
leg = axs[1].legend(['PGPA model', 'Connectome', 'Random'],
                    prop={'size':22})


##################################
# Get reciprocity
##################################

centroidsUncorrected = aux_random_graphs.get_coords()
# This is because the coordinates are off by a factor of 10
centroids = {k:centroidsUncorrected[k]/10. for k in centroidsUncorrected}

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
nonRecipNames = list( set(edges.keys()) - set(recipNames) ) # subtract recip edges from all edges

nonRecipDistances = [edges[k] for k in nonRecipNames]
recipDistances = [edges[k] for k in recipNames]

binnedRecipDistances = np.histogram(recipDistances,bins,normed=True)

dx = bins[1]-bins[0]
proportionRecip = binnedRecipDistances[0]

yticks = [0,0.15,0.3,0.45,0.6,0.75]
xticks = [0,2.5,5.0,7.5,10.0]

axs[0].hist(nonRecipDistances,bins,normed=True,facecolor='b',alpha=0.5)
axs[0].hist(recipDistances,bins,normed=True,facecolor='g',alpha=0.5)

axs[0].set_xticks(xticks)
axs[0].set_xticklabels(xticks,fontsize=labelsize)
axs[0].set_yticks(yticks)
axs[0].set_yticklabels(yticks,fontsize=labelsize)
axs[0].set_xlabel('Distance (mm)', fontsize=labelsize)
axs[0].set_ylabel('Probability density', fontsize=labelsize)
axs[0].set_xlim([0,max(bins)])
axs[0].set_title('Connectome',fontsize=28)
axs[0].legend(['Reciprocal edges', 'Nonreciprocal edges'],prop={'size':legendsize})



##################################
# Now for the biophysical model...
##################################
G, A, labels = biophysical_model(N=num_brain_nodes,
                                 N_edges=num_brain_edges_directed,
                                 L=.75, gamma=1., brain_size='brain')

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
nonRecipNames = list( set(edges.keys()) - set(recipNames) ) # subtract recip edges from all edges

nonRecipDistances = [edges[k] for k in nonRecipNames]
recipDistances = [edges[k] for k in recipNames]

binnedRecipDistances = np.histogram(recipDistances,bins,normed=True)
dx = bins[1]-bins[0]
proportionRecip = binnedRecipDistances[0]



axs[2].hist(nonRecipDistances,bins,normed=True,facecolor='b',alpha=0.5)
axs[2].hist(recipDistances,bins,normed=True,facecolor='g',alpha=0.5)
axs[2].set_xticks(xticks)
axs[2].set_xticklabels(xticks,fontsize=labelsize)
axs[2].set_yticks(yticks)
axs[2].set_yticklabels([])
axs[2].set_xlabel('Distance (mm)', fontsize=labelsize)
axs[2].set_ylabel('Probability density', fontsize=28)
axs[2].set_xlim([0,10.0])
axs[2].set_title('PGPA model',fontsize=28)
axs[2].legend(['Reciprocal edges', 'Nonreciprocal edges'],prop={'size':legendsize})


# Finally add labels

axs[0].text(10*0.05,0.75*0.925,'a',fontsize=labelsize,fontweight='bold')
axs[1].text(0.1+2*0.05,0.3*0.925,'b',fontsize=labelsize,fontweight='bold')
axs[2].text(10*0.05,0.75*0.925,'c',fontsize=labelsize,fontweight='bold')

plt.show(block=False)
