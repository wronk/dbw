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

labelsize=11
ticksize=10
legendsize=8

def dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

if __name__ == "__main__":
    G,_,model_centroids = biophysical_model()


    centroidsUncorrected = aux_random_graphs.get_coords()
    # This is because the coordinates are off by a factor of 10
    centroids = [centroidsUncorrected[k]/10. for k in centroidsUncorrected]

    inter_node_distances = [dist(edge1,edge2)
                            for edge1 in centroids for edge2 in centroids if not all(edge1 == edge2)]

    model_distances = [model_centroids[edge1][edge2] for edge1 in G.nodes() for edge2 in G.nodes() if edge1 != edge2]

    fig,axs = plt.subplots(1,facecolor='white',figsize=(3.5,2.75),dpi=200.)
    fig.subplots_adjust(bottom=0.15,left=0.15)

    bins = np.linspace(0,13,51)
    axs.hist(inter_node_distances,bins,facecolor=color_scheme.ATLAS,normed=True)
    model_distances_binned,_ = np.histogram(model_distances,bins,normed=True)
    model_bins = bins[0:-1]+(bins[1]-bins[0])/2
    axs.plot(model_bins,model_distances_binned,'-',c='k',lw=3)
    axs.set_xlim([0,13])
    
    xticks = [0,4,8,12]
    yticks = np.arange(0,0.3,0.05)
    axs.set_ylabel('Probability',fontsize=labelsize)
    axs.set_xlabel('Distance (mm)',fontsize=labelsize)
    axs.set_xticks(xticks); axs.set_yticks(yticks)
    axs.set_xticklabels(xticks,fontsize=ticksize)
    axs.set_yticklabels(yticks,fontsize=ticksize)
    leg=axs.legend(['7mm$^3$ cube','Mouse brain'],prop={'size':legendsize})
    fig.subplots_adjust(bottom=0.2,left=0.2)
    plt.show(block=False)


