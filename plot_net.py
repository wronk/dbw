"""
Created on Tue Aug 27 2014

@author: rkp, wronk, sidh0
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import network_compute


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


def plot_node_btwn(G, bins=20,ranked=False):
    """
    Plot the node-betweenness distributions.

    Args:
        G: networkx graph object
    Returns:
        figure handle & axes array.
    """
    # Calculate node-betweenness
    node_btwn_dict = nx.betweenness_centrality(G)

    # Sort node-betweenness dictionary by node-betweenness values
    node_btwn_labels_sorted, node_btwn_vec_sorted = \
        network_compute.get_ranked(node_btwn_dict)

    # Open figure & axes
    if ranked:
        fig, axs = plt.subplots(2, 1,facecolor='w')

        # Plot histogram
        axs[0].hist(node_btwn_vec_sorted, bins)
        axs[0].set_ylabel('Occurrences')
        axs[0].set_xlabel('Node-betweenness')
    
        # Plot sorted node between values
        axs[1].scatter(np.arange(len(node_btwn_vec_sorted)),
                       node_btwn_vec_sorted, s=20, c='r')
        axs[1].set_xlabel('Area')
        axs[1].set_ylabel('Node-betweenness')

        return fig,axs
    else:
        fig, ax = plt.subplots(1,1,facecolor='w')
        
        # Plot histogram
        ax.hist(node_btwn_vec_sorted,bins=bins)
        ax.set_xlabel('Betweenness centrality')
        ax.set_ylabel('Occurrences')
        ax.set_title('Betweenness centrality')
        return fig,ax


def plot_edge_btwn(G, bins=20):
    """
    Plot the edge-betweenness distributions.

    Args:
        G: networkx graph object
    Returns:
        figure handle & axes array.
    """
    # Get edge-betweenness dictionary
    edge_btwn_dict = nx.edge_betweenness_centrality(G)

    # Sort edge-betweenness dictionary by edge-betweenness values
    edge_btwn_labels_sorted, edge_btwn_vec_sorted = \
        network_compute.get_ranked(edge_btwn_dict)

    # Open figure & axes
    fig, axs = plt.subplots(2, 1)
    # Plot histogram
    axs[0].hist(edge_btwn_vec_sorted, bins)
    axs[0].set_ylabel('Occurrences')
    axs[0].set_xlabel('Edge-betweenness')

    # Plot sorted node between values
    axs[1].scatter(np.arange(len(edge_btwn_vec_sorted)),
                   edge_btwn_vec_sorted, s=20, c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Edge-betweenness')

    return fig


def plot_out_in_ratios(W_net, labels=None, bins=20):
    """
    Plot a distribution of output/input connection ratios for a given
    network (defined by a weight matrix W_net)
    """

    if labels is None:
        labels = np.arange(W_net.shape[0])

    # Calculate total output & input connections for each node
    _,_,out_in_dict = network_compute.out_in(W_net, labels,binarized=False)
    # Calculate ranked output/input ratios
    out_in_labels_sorted, out_in_vec_sorted = \
        network_compute.get_ranked(out_in_dict)

    # Open figure & axes
    fig, axs = plt.subplots(2, 1)
    # Plot histogram
    axs[0].hist(out_in_vec_sorted, bins)
    axs[0].set_ylabel('Occurrences')
    axs[0].set_xlabel('Output/Input')

    # Plot sorted input/output ratios
    axs[1].scatter(np.arange(len(out_in_vec_sorted)),
                   out_in_vec_sorted, s=20, c='r')
    axs[1].set_xlabel('Area')
    axs[1].set_ylabel('Output/Input')

    return fig, axs


def plot_clustering_coeff_pdf(G, bins=np.linspace(0., 0.25, 150)):
    '''
    Plot clustering coefficient probability density function

    Parameters
    ----------
    G : networkx graph object
        graph to calculate clustering coefficients of
    bins : array | list
        bin edges for histogram

    Returns
    --------
    fig : fig
        figure object of distribution histogram for plotting
    '''

    ccoeff_dict = nx.clustering(G)
    ccoeffs = np.array(ccoeff_dict.values())
    #TODO Need to normalize coefficients?

    # Constuct figure
    #fig, (ax0, ax1) = plt.subplots(ncols=1)
    fig,axs = plt.subplots(ncols=1,facecolor='white')

    # Plot coefficients according to bins
    plt.hist(ccoeffs, bins)
    plt.title('Clustering Coefficient')
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Occurrences')

    return fig,axs


def plot_clustering_coeff_ranked(G, num_ranked=10):
    '''
    Plot clustering coefficient ranked by maximum value

    Parameters
    ----------
    G : networkx graph object
        graph to get clustering coefficients for

    num_ranked : int
        number of ranked brain areas to show

    Returns
    --------
    fig : fig
        figure object of distribution histogram for plotting
    '''

    # Get clustering coefficients
    ccoeff_dict = nx.clustering(G)

    # Graph params width = 0.5
    xpos = np.arange(num_ranked)
    width = 0.8

    # Constuct figure
    fig = plt.figure()

    sorted_tups = sorted(zip(ccoeff_dict.values(), ccoeff_dict.keys()),
                         key=lambda tup: tup[0], reverse=True)[:num_ranked]

    # Plot top ranked coefficients according to bins
    plt.bar(xpos, [w for w, _ in sorted_tups], fc='green',
            width=width, alpha=.8)
    plt.xticks(xpos + width / 2., [n for _, n in sorted_tups])

    plt.title('Ranked Clustering Coefficients')
    plt.xlabel('Region')
    plt.ylabel('Clustering Coefficient')

    return fig


def plot_connection_strength(W, bins=10):
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
    W_new = W[~(np.isinf(W) + np.isnan(W))]

    fig, ax = plt.subplots(1, 1)
    binnedW, bins, patches = plt.hist(W_new, bins, facecolor='red', alpha=0.5)

    ax.set_xlabel('Weight value')
    ax.set_ylabel('Frequency')
    ax.set_title('Connection strength')
    plt.show()

    return fig, ax


def plot_shortest_path_distribution(G):
    '''
    Generate figure/axis and plots a bar graph of shortest path distribution

    Parameters
    ----------
    G -- A graph object

    Returns
    --------
    fig, ax : fig, ax
        plotting objects showing distribution of shortest paths
    '''
    SP = nx.shortest_path_length(G)
    Names = G.nodes()

    SP_values = [SP[entry].values() for entry in SP]

    All_SP_values = [item for sublist in SP_values for item in sublist]

    uniques = np.unique(All_SP_values)
    int_uniques = [int(entry) for entry in uniques]
    counts = []
    for j in range(len(uniques)):
        current = uniques[j]
        counts.append(sum(All_SP_values == current))

    fig, ax = plt.subplots(1, 1)

    ax.bar(uniques, counts)
    ax.set_xlabel('Number of nodes in shortest path')
    ax.set_ylabel('Frequency')
    ax.set_xticks(uniques + 0.4)
    ax.set_xticklabels(int_uniques)
    ax.set_title('Distribution of shortest path lengths')

    plt.show()

    return fig, ax


def plot_degree_distribution(G,bins=np.linspace(0,140,50)):
    ''' Plots the degree distribution of a graph object '''
    degrees = G.degree().values()
    fig, ax = plt.subplots(1,1,facecolor='white')

    ax.hist(degrees,bins)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Occurrences')
    ax.set_title('Degree distribution')

    plt.show()

    return fig, ax


def line_hist(ax,G,feature,bins,**kwargs):
    """Plot a histogram of a specified feature as a line on specified axis"""
    # Get data vector to use
    if feature == 'degree':
        data_vec = G.degree().values()
    elif feature == 'ccoeff':
        data_vec = nx.clustering(G).values()
    elif feature == 'node_btwn':
        data_vec = nx.betweenness_centrality(G).values()
    
    # Calculate histogram
    cts,bins = np.histogram(data_vec,bins)
    
    # Get bin centers
    bcents = .5*(bins[:-1] + bins[1:])
    
    # Plot line
    ax.plot(bcents,cts,**kwargs)
    