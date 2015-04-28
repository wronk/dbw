from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import graph_tools.auxiliary as aux_tools

from numpy import concatenate as cc

from extract.brain_graph import binary_directed as brain_graph
from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model
from network_plot.change_settings import set_all_text_fontsizes as set_fontsize

import aux_random_graphs

from brain_constants import *


if __name__ == "__main__":

    # load brain graph, adjacency matrix, and labels
    G_brain, A_brain, labels_brain = brain_graph()
    G_model, A_model, labels_model = biophysical_model(N=num_brain_nodes,
                                 N_edges=num_brain_edges_directed,
                                 L=.76,
                                 gamma=1.)
    Gs = [G_brain,G_model]
    As = [A_brain,A_model]
    allLabels = [labels_brain,labels_model]


    fig,axs = plt.subplots(1,ncols=2,facecolor='white')
    
    for g,G in enumerate(Gs):
        A = As[g]
        labels = allLabels[g]
        
        # First the brain
        if g ==0:
            brainCentroidsPre = aux_random_graphs.get_coords()
            brainCentroids = {i:brainCentroidsPre[i]/10 for i in brainCentroidsPre}
            
        else:
            brainCentroidValues = G.centroids
            brainCentroids = {k:brainCentroidValues[k] for k in range(len(brainCentroidValues))}
            labels = np.arange(len(brainCentroids))
        
        # get in & out degree
        brainInDeg = np.array([G.in_degree()[node] for node in G])
        brainOutDeg = np.array([G.out_degree()[node] for node in G])
        brainDeg = brainInDeg + brainOutDeg
        brainDegDiff = brainInDeg - brainOutDeg
        
        brainEdges = {} # distances
        
        for i in brainCentroids:
            for j in brainCentroids:
                current = tuple(sorted([i,j]))
                brainEdges[current] = np.sqrt(np.sum((brainCentroids[i] - brainCentroids[j])**2))
                
                
                
        bins = np.linspace(0,13,41)
        binnedBrainDistances = np.histogram(brainEdges.values(),bins,normed=True)

        A2 = A.copy()
        np.fill_diagonal(A2,False)
        brainRecip = A2 * A2.T
        
        i,j = np.where(brainRecip)
        brainCentroidNames = labels
        brainRecipNames = [tuple(sorted([brainCentroidNames[i[k]],brainCentroidNames[j[k]]])) for k in range(len(i))]
        
        recipDistances = [brainEdges[k] for k in brainRecipNames]
        
        binnedBrainRecipDistances = np.histogram(recipDistances,bins,normed=True)
        
        proportionBrainRecip = binnedBrainRecipDistances[0]
        
        yticks = np.round(np.linspace(0,0.7,7),decimals=2)
        xticks = np.round(np.linspace(0,np.max(bins),6),decimals=2)

        
        axs[g].hist(brainEdges.values(),bins,normed=True,facecolor=[0.8,0.1,0.1])
        axs[g].plot(bins[0:len(bins)-1],proportionBrainRecip,linewidth=3,color=[0.1,0.1,0.75])
        axs[g].set_xticks(xticks)
        axs[g].set_xticklabels(xticks,fontsize=24)
        axs[g].set_yticks(yticks)
        axs[g].set_yticklabels(yticks,fontsize=24)
        axs[g].set_xlabel('Distance (mm)', fontsize=28)
        axs[g].set_ylabel('Density', fontsize=28)
        

        leg =axs[g].legend(['Reciprocal edges', 'All edges'],prop={'size':24})
        
    plt.show(block=False)


### We need to set the loop so that it plots two subplots
        ### ALso just tidy up the plots, etc... Pretty much done
