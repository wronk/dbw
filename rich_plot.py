import pdb
"""
Created on Tue Aug 26 22:26:31 2014

@author: rkp
"""

import operator
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_node_btwn(G,bins=20):
    """Plot the node-betweenness distributions.
    
    Args:
        G: networkx graph object"""
    
    # Get node-betweenness dictionary
    node_btwn_dict = nx.betweenness_centrality(G)
    # Determine histogram bins and bin labels
    node_btwn_vec = np.array(node_btwn_dict.values())

    # Sort node-betweenness dictionary by node-betweenness values
    node_btwn_dict_sorted = sorted(node_btwn_dict.iteritems(),
                                   key=operator.itemgetter(1),reverse=True)
    node_btwn_vec_sorted = [item[1] for item in node_btwn_dict_sorted]
    node_btwn_labels_sorted = [item[0] for item in node_btwn_dict_sorted]

    # Open figure & axes
    fig,axs = plt.subplots(2,1)
    # Plot histogram
    axs[0].hist(node_btwn_vec,bins)
    axs[0].set_ylabel('Probability')
    axs[0].set_xlabel('Node-betweenness')
    
    # Plot sorted node between values
    axs[1].scatter(np.arange(len(node_btwn_vec_sorted)),
                   node_btwn_vec_sorted,s=20,c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Node-betweenness')
    
def plot_edge_btwn(G,bins=20):
    """Plot the edge-betweenness distributions.
    
    Args:
        G: networkx graph object"""
    
    # Get node-betweenness dictionary
    edge_btwn_dict = nx.edge_betweenness_centrality(G)
    # Determine histogram bins and bin labels
    edge_btwn_vec = np.array(edge_btwn_dict.values())

    # Sort edge-betweenness dictionary by node-betweenness values
    edge_btwn_dict_sorted = sorted(edge_btwn_dict.iteritems(),
                                   key=operator.itemgetter(1),reverse=True)
    edge_btwn_vec_sorted = [item[1] for item in edge_btwn_dict_sorted]
    edge_btwn_labels_sorted = ['%s->%s'%(item[0][0],item[0][1]) for item in edge_btwn_dict_sorted]

    # Open figure & axes
    fig,axs = plt.subplots(2,1)
    # Plot histogram
    axs[0].hist(edge_btwn_vec,bins)
    axs[0].set_ylabel('Probability')
    axs[0].set_xlabel('Edge-betweenness')
    
    # Plot sorted node between values
    axs[1].scatter(np.arange(len(edge_btwn_vec_sorted)),
                   edge_btwn_vec_sorted,s=20,c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Edge-betweenness')