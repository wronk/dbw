import pdb
"""
Created on Tue Aug 26 22:26:31 2014

@author: rkp
"""

import operator
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def plot_connected_components(G):
    """Plot a distribution of the connected component sizes for a graph.
    
    Args:
        G: A networkx graph object.
    Returns:
        figure handle & axis."""
    
    # Calculate list of connected component sizes
    cc_sizes = [len(nodes) for nodes in nx.connected_components(G)]
    # Sort connected component sizes & plot them
    cc_sizes_sorted = sorted(cc_sizes,reverse=True)
    
    # Open plots
    fig, ax = plt.subplots(1,1)
    ax.scatter(np.arange(len(cc_sizes_sorted)),cc_sizes_sorted,s=20,c='r')
    
    return fig
    
def plot_node_btwn(G,bins=20):
    """Plot the node-betweenness distributions.
    
    Args:
        G: networkx graph object
    Returns:
        figure handle & axes array."""
    
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
    axs[0].set_ylabel('Occurrences')
    axs[0].set_xlabel('Node-betweenness')
    
    # Plot sorted node between values
    axs[1].scatter(np.arange(len(node_btwn_vec_sorted)),
                   node_btwn_vec_sorted,s=20,c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Node-betweenness')
    
    return fig
    
def plot_edge_btwn(G,bins=20):
    """Plot the edge-betweenness distributions.
    
    Args:
        G: networkx graph object
    Returns:
        figure handle & axes array."""
    
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
    axs[0].set_ylabel('Occurrences')
    axs[0].set_xlabel('Edge-betweenness')
    
    # Plot sorted node between values
    axs[1].scatter(np.arange(len(edge_btwn_vec_sorted)),
                   edge_btwn_vec_sorted,s=20,c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Edge-betweenness')
    
    return fig
    
def plot_out_in_ratios(W_net,bins=20,labels=None,binarized=True):
    """Plot a distribution of output/input connection ratios for a given
    network (defined by a weight matrix W_net)"""
    
    if binarized:
        W = (W_net > 0).astype(float)
    else:
        W = W_net.copy()
    if labels is None:
        labels = np.arange(W.shape[0])
    
    # Calculate total output & input connections for each node
    out_total = W.sum(axis=1)
    in_total = W.sum(axis=0)
    # Calculate out/in ratio
    out_in_vec = out_total.astype(float)/in_total
    
    # Make in_out_ratio dictionary
    out_in_dict = {labels[idx]:out_in_vec[idx] for idx in range(len(labels))}
    # Sort in_out_ratio
    out_in_dict_sorted = sorted(out_in_dict.iteritems(),
                                key=operator.itemgetter(1),reverse=True)
    out_in_vec_sorted = [item[1] for item in out_in_dict_sorted]
    out_in_labels_sorted = [item[0] for item in out_in_dict_sorted]
    
    # Open figure & axes
    fig, axs = plt.subplots(2,1)
    # Plot histogram
    axs[0].hist(out_in_vec,bins)
    axs[0].set_ylabel('Occurrences')
    axs[0].set_xlabel('Output/Input')
    
    # Plot sorted input/output ratios
    axs[1].scatter(np.arange(len(out_in_vec_sorted)),
                   out_in_vec_sorted,s=20,c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Output/Input')
    
    return fig,axs