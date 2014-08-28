"""
Created on Tue Aug 27 2014

@author: rkp, wronk, sidh0
"""

import operator
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def plot_connected_components(G):
    """
    Plot a distribution of the connected component sizes for a graph.

    Args:
        G: A networkx graph object.
    Returns:
        figure handle & axis.
    """

    # Calculate list of connected component sizes
    cc_sizes = [len(nodes) for nodes in nx.connected_components(G)]
    # Sort connected component sizes & plot them
    cc_sizes_sorted = sorted(cc_sizes, reverse=True)

    # Open plots
    fig, ax = plt.subplots(1, 1)
    ax.scatter(np.arange(len(cc_sizes_sorted)), cc_sizes_sorted, s=20, c='r')

    return fig


def plot_node_btwn(G, bins=20):
    """
    Plot the node-betweenness distributions.

    Args:
        G: networkx graph object
    Returns:
        figure handle & axes array.
    """

    # Get node-betweenness dictionary node_btwn_dict = nx.betweenness_centrality(G)
    # Determine histogram bins and bin labels
    node_btwn_vec = np.array(node_btwn_dict.values())

    # Sort node-betweenness dictionary by node-betweenness values
    node_btwn_dict_sorted = sorted(node_btwn_dict.iteritems(),
                                   key=operator.itemgetter(1), reverse=True)
    node_btwn_vec_sorted = [item[1] for item in node_btwn_dict_sorted]
    node_btwn_labels_sorted = [item[0] for item in node_btwn_dict_sorted]

    # Open figure & axes
    fig, axs = plt.subplots(2, 1)

    # Plot histogram
    axs[0].hist(node_btwn_vec, bins)
    axs[0].set_ylabel('Occurrences')
    axs[0].set_xlabel('Node-betweenness')

    # Plot sorted node between values
    axs[1].scatter(np.arange(len(node_btwn_vec_sorted)),
                   node_btwn_vec_sorted, s=20, c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Node-betweenness')

    return fig


def plot_edge_btwn(G, bins=20):
    """
    Plot the edge-betweenness distributions.

    Args:
        G: networkx graph object
    Returns:
        figure handle & axes array.
    """

    # Get node-betweenness dictionary
    edge_btwn_dict = nx.edge_betweenness_centrality(G)
    # Determine histogram bins and bin labels
    edge_btwn_vec = np.array(edge_btwn_dict.values())

    # Sort edge-betweenness dictionary by node-betweenness values
    edge_btwn_dict_sorted = sorted(edge_btwn_dict.iteritems(),
                                   key=operator.itemgetter(1), reverse=True)
    edge_btwn_vec_sorted = [item[1] for item in edge_btwn_dict_sorted]
    edge_btwn_labels_sorted = ['%s->%s'%(item[0][0],item[0][1])
                               for item in edge_btwn_dict_sorted]

    # Open figure & axes
    fig, axs = plt.subplots(2, 1)
    # Plot histogram
    axs[0].hist(edge_btwn_vec, bins)
    axs[0].set_ylabel('Occurrences')
    axs[0].set_xlabel('Edge-betweenness')

    # Plot sorted node between values
    axs[1].scatter(np.arange(len(edge_btwn_vec_sorted)),
                   edge_btwn_vec_sorted, s=20, c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Edge-betweenness')

    return fig


def plot_out_in_ratios(W_net, bins=20, labels=None, binarized=True):
    """
    Plot a distribution of output/input connection ratios for a given
    network (defined by a weight matrix W_net)
    """

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
    out_in_vec = out_total.astype(float) / in_total

    # Make in_out_ratio dictionary
    out_in_dict = {labels[idx]: out_in_vec[idx] for idx in range(len(labels))}
    # Sort in_out_ratio
    out_in_dict_sorted = sorted(out_in_dict.iteritems(),
                                key=operator.itemgetter(1), reverse=True)
    out_in_vec_sorted = [item[1] for item in out_in_dict_sorted]
    out_in_labels_sorted = [item[0] for item in out_in_dict_sorted]

    # Open figure & axes
    fig, axs = plt.subplots(2, 1)
    # Plot histogram
    axs[0].hist(out_in_vec, bins)
    axs[0].set_ylabel('Occurrences')
    axs[0].set_xlabel('Output/Input')

    # Plot sorted input/output ratios
    axs[1].scatter(np.arange(len(out_in_vec_sorted)),
                   out_in_vec_sorted, s=20, c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Output/Input')

    return fig, axs

def plot_clustering_coeff_pdf(coeffs, bins=np.linspace(0., 0.25, 150)):
    '''
    Plot clustering coefficient probability density function

    Parameters
    ----------
    coeffs : array
        matrix of clustering coefficients
    bins : array | list
        bin edges for histogram

    Returns
    --------
    fig : fig
        figure object of distribution histogram for plotting
    '''

    #TODO Need to normalize coefficients?

    # Constuct figure
    #fig, (ax0, ax1) = plt.subplots(ncols=1)
    fig = plt.figure()

    # Plot coefficients according to bins
    plt.hist(coeffs.flatten(), bins, fc='g', alpha=.8, normed=True)
    plt.title('Clustering Coefficient PDF')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Probability')

    return fig


def plot_clustering_coeff_ranked(coeffs, names, num_ranked=10):
    '''
    Plot clustering coefficient ranked by maximum value

    Parameters
    ----------
    coeffs : list | array
        matrix of clustering coefficients

    names : list
        labels corresponding to brain area acronyms

    num_ranked : int
        number of ranked brain areas to show

    Returns
    --------
    fig : fig
        figure object of distribution histogram for plotting
    '''

    # Graph params width = 0.5
    xpos = np.arange(num_ranked)
    width = 0.8

    # Constuct figure
    fig = plt.figure()

    sorted_tups = sorted(zip(coeffs, names), key=lambda tup: tup[0],
                         reverse=True)[:num_ranked]

    # Plot top ranked coefficients according to bins
    plt.bar(xpos, [w for w, _ in sorted_tups], fc='green',
            width=width, alpha=.8)
    plt.xticks(xpos + width / 2., [n for _, n in sorted_tups])

    plt.title('Ranked Clustering Coefficients')
    plt.xlabel('Region')
    plt.ylabel('Clustering Coefficient')

    return fig


def connection_strength(W, bins=10):
    '''
    Generate figure/axis and plots a histogram of connection strength

    Parameters
    ----------
    W : 2-D array
        weight matrix

    Returns
    --------
    fig, ax : fig, ax
        plotting objects showing distribution of connection strengths
    '''

    W_new = W[W > 0]

    fig, ax = plt.subplots(1, 1)
    binnedW, bins, patches = plt.hist(W_new, bins, facecolor='red', alpha=0.5)

    ax.set_xlabel('Weight value')
    ax.set_ylabel('Frequency')
    ax.set_title('Connection strength')
    plt.show()

    return fig, ax


def shortest_path_distribution(W):
    '''
    Generate figure/axis and plots a bar graph of shortest path distribution

    Parameters
    ----------
    W : 2-D array
        weight matrix

    Returns
    --------
    fig, ax : fig, ax
        plotting objects showing distribution of shortest paths
    '''

    if type(W) != '<type \'numpy.ndarray\'>':
        W = np.array(W)

    W_new = W[W > 0]
    uniques = np.unique(W_new)
    int_uniques = [int(entry) for entry in uniques]
    counts = []
    for j in range(len(uniques)):
        current = uniques[j]
        counts.append(sum(W_new == current))

    fig, ax = plt.subplots(1, 1)

    ax.bar(uniques, counts)
    ax.set_xlabel('Number of nodes in shortest path')
    ax.set_ylabel('Frequency')
    ax.set_xticks(uniques + 0.4)
    ax.set_xticklabels(int_uniques)
    ax.set_title('Distribution of shortest path lengths')

    plt.show()

    return fig, ax

def plot_degree_distribution(G):
    ''' Plots the degree distribution of a graph object '''
    degrees = G.degree()
    degrees_list = [degrees[entry] for entry in degrees]
    degrees_array = np.array(degrees_list)
    uniques = np.unique(degrees_list)
    int_uniques = [int(entry) for entry in uniques]
    
    counts = []
    for j in range(len(uniques)):
        current = uniques[j]
        counts.append(sum(degrees_array == current))
       
    #deg_pdf = counts/sum(counts) 
    Fig,ax = plt.subplots(1,1)
    
    ax.bar(uniques,counts)
    ax.set_xlabel('Node degree')
    ax.set_ylabel('PDF')
    #ax.set_ylim((0,0.1))
    #ax.set_xlim((0,120))
    ax.set_title('Node degree distribution')
    
    plt.show()
    
    return Fig,ax