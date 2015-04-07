'''
Created on Tue Aug 27 2014

@author: wronk

network_viz.py
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from os import path as op
from mpl_toolkits.mplot3d import Axes3D


def plot_3D_network(ax, node_names, node_positions, node_label_set, edges,
                    edge_label_set, node_sizes=None, node_colors=None,
                    node_alpha=None, edge_sizes=None, edge_colors=None,
                    edge_alpha=None):
    '''
    Plot clustering coefficient probability density function

    Parameters
    ----------
    ax : axis object with 3d projection
        axis object to modify
    node_names : list
        labels of nodes in graph
    node_positions : N x 3 array
        array of positions
    edges : list
        connections between nodes
    node_label_set : list of bool
        whether or not to text label the nodes
    edges : list of tuples
        edges between nodes (use graph.edges())
    edge_label_set : list of bool
        whether or not to label the edges
    node_sizes : list | None
        size of each node's marker in the graph
    node_colors : list | None
        color of each node's marker
    edge_sizes : list | None
        size of each edge's line
    edge_colors : list | None
        color of each edge's line

    '''

    node_label_offset = 0.05
    edge_label_offset = 0.05

    ###################################

    if node_sizes is None:
        node_sizes = [50] * len(node_names)
    if node_colors is None:
        node_colors = ['green'] * len(node_names)
    if node_alpha is None:
        node_alpha = [1.] * len(node_names)
    if edge_sizes is None:
        edge_sizes = [1] * len(edges)
    if edge_colors is None:
        edge_colors = ['blue'] * len(edges)
    if edge_alpha is None:
        edge_alpha = [1.] * len(edges)

    ###################################
    # Add nodes to graph
    for ni, node_pt in enumerate(node_positions):
        ax.scatter(node_pt[0], node_pt[1], node_pt[2],
                   s=node_sizes[ni], c=node_colors[ni],
                   alpha=node_alpha[ni], depthshade=False, lw=0)
        if node_label_set[ni]:
            ax.text(node_positions[ni, 0], node_positions[ni, 1],
                    node_positions[ni, 2] + node_label_offset, node_names[ni],
                    color=node_colors[ni], alpha=node_alpha[ni], ha='center')

    ###################################
    # Convert edges to list
    edge_inds = [[node_names.index(n1), node_names.index(n2)]
                 for n1, n2 in edges]

    # Generate list of 2 x 3 array corresponding to list of end edge points
    edge_positions = [np.vstack((node_positions[e[0]], node_positions[e[1]]))
                      for e in edge_inds]

    ###################################
    # Add edges to graph
    for ei, edge_pt in enumerate(edge_positions):
        ax.plot(edge_pt[:, 0], edge_pt[:, 1], edge_pt[:, 2],
                lw=edge_sizes[ei], c=edge_colors[ei],
                alpha=edge_alpha[ei], dash_capstyle='round', zorder=-1)

        # Add text edge label to graph
        if edge_label_set[ei]:
            mean_pt = np.mean(edge_pt, axis=0)
            vec_dir = edge_pt[1, :] - edge_pt[0, :]

            ax.text(mean_pt[0], mean_pt[1], mean_pt[2] + edge_label_offset,
                    node_names[edge_inds[ei][0]] + '<->' +
                    node_names[edge_inds[ei][1]], vec_dir,
                    alpha=edge_alpha[ei], color=edge_colors[ei], ha='center')

    # Cleanup
    #ax.set_xlabel('A <-> P')
    #ax.set_ylabel('L <-> M <-> L')
    #ax.set_zlabel('S <-> I')
    ax.set_axis_off()

    ax.set_title('Top Clustering Coefficients')

    return ax


def make_movie():
    import network_gen

    plt.close('all')
    plt.ion()

    save_movie = True
    save_fpath = './movie/images'

    # Set parameters
    p_th = .01  # P-value threshold
    w_th = 0  # Weight-value threshold

    # Set relative directory path
    dir_name = '../friday-harbor/linear_model'

    # Load weights & p-values
    W, P, row_labels, col_labels = network_gen.load_weights(dir_name)
    # Threshold weights according to weights & p-values
    W_net, mask = network_gen.threshold(W, P, p_th=p_th, w_th=w_th)
    # Set weights to zero if they don't satisfy threshold criteria
    W_net[W_net == -1] = 0.
    # Set diagonal weights to zero
    np.fill_diagonal(W_net, 0)

    # Put everything in a dictionary
    W_net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
                  'data': W_net}
    G = network_gen.import_weights_to_graph(W_net_dict, directed=True)

    node_names = row_labels[0:5]
    node_positions = np.random.rand(5, 3)
    node_labels = [True] * len(node_names)
    edges = [('AAA_L', 'ACAv_L'), ('AD_L', 'ACAd_L')]

    node_sizes = np.random.randint(20, high=300, size=len(node_names))
    node_colors = ['green'] * len(node_names)
    node_alpha = list(np.arange(.1, 1., 1. / len(node_names)))

    edge_sizes = np.random.randint(1, high=4, size=len(edges))
    edge_colors = ['blue'] * len(edges)
    edge_labels = [False] * len(edges)
    edge_alpha = list(np.arange(.1, 1., 1. / len(node_names)))

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d', axisbg='black')
    ax2 = fig.add_subplot(122)
    fig.set_facecolor('black')

    plot_3D_network(ax, node_names, node_positions, node_labels, edges,
                    edge_labels, node_sizes, node_colors, node_alpha,
                    edge_sizes, edge_colors, edge_alpha)

    if save_movie:
        # Png params
        elev = 20.  # elevation angle for movie
        for ai, ang in enumerate(np.arange(-270, 90, 3)):
            ax.view_init(elev=elev, azim=ang)
            plt.savefig(op.join(save_fpath, 'mov_%03i.png' % ai), ec='black',
                        fc='black', bbox_inches='tight', pad_inches=0.)
    plot_3D_network(ax, node_names, node_positions, node_labels, edges,
                    edge_labels)

    plt.draw()
    plt.show()


def plot_scatterAndMarginal(ax_scat, ax_histTop, ax_histRight, k_in, k_out,
                            bin_width, marker_size, marker_color,
                            indegree_bins=None, outdegree_bins=None):
    """
    Function to create a scatter plot with marginal histogram distributions.

    Parameters
    ==========
    ax_scat : axes object
        Axes object to plot scatter points on
    ax_histTop : axes object
        Axes object to plot histogram of points collapsed onto x axis
    ax_histRight : axes object
        Axes object to plot histogram of points collapsed onto y axis
    k_in : array or list
        In degree data (or scatter x coordinates)
    k_out : array or list
        Out degree data (or scatter y coordinates)
    bin_width : int or float
        Desired width of histogram bins
    marker_size : float
        Size of scatter points in scatter plot
    marker_color : str
        Color of scatter points and histogram
    """
    # Set font type for compatability with adobe if editting later
#    mpl.rcparams['ps.fonttype'] = 42
#    mpl.rcparams['pdf.fonttype'] = 42

    # Params
    lw = 0  # Line width around markers
    bin_width = float(bin_width)

    ############################
    # Scatter Plot
    ############################
    ax_scat.scatter(k_in, k_out, s=marker_size, lw=lw, c=marker_color)
    ax_scat.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax_scat.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax_scat.set_aspect(1.)
    #ax_scat.set_xticklabels([int(t) for t in ax_scat.get_xticks()], rotation=45., va='top')
    #ax_scat.set_yticklabels([int(t) for t in ax_scat.get_yticks()], rotation=45., ha='right')

    #Round up to nearest 10
    x_lim, y_lim = ax_scat.get_xlim(), ax_scat.get_ylim()
    max_lim = np.max((x_lim[1], y_lim[1]))
    ax_scat.set_xlim([-5, max_lim])
    ax_scat.set_ylim([-5, max_lim])
    #ax_scat.set_xlim([-5, round(x_lim[1] + 5.)])
    #ax_scat.set_ylim([-5, round(y_lim[1] + 5.)])

    ############################
    # Plot marginal histograms
    ############################

    # Remove tick labels below top hist and left of right hist
    plt.setp(ax_histTop.get_xticklabels() + ax_histRight.get_yticklabels(),
             visible=False)

    # Create identical bins for histograms
    xymax = np.max([np.max(np.fabs(k_in)), np.max(np.fabs(k_out))])
    lim = (int(xymax / bin_width) + 1) * bin_width
    bins = np.arange(-lim, lim + bin_width, bin_width)

    if indegree_bins is None:
        indegree_bins = bins
    if outdegree_bins is None:
        outdegree_bins = bins

    # Plot histograms and limit number of ticks
    ax_histTop.hist(k_in, bins=indegree_bins, orientation='vertical',
                    fc=marker_color)
    ax_histTop.yaxis.set_major_locator(plt.MaxNLocator(3))
    #ax_histTop.set_yticks(y_histTics)
    #ax_histTop.set_yticklabels([str(l) for l in y_histTics], rotation=0)

    ax_histRight.hist(k_out, bins=outdegree_bins, orientation='horizontal',
                      fc=marker_color)
    ax_histRight.xaxis.set_major_locator(plt.MaxNLocator(3))
    plt.setp(ax_histRight.xaxis.get_majorticklabels(), va='top')
    #ax_histRight.set_xticks(x_histTics)
    #ax_histRight.set_xticklabels([str(l) for l in x_histTics], rotation=0)

if __name__ == '__main__':
    #make_movie()

    from random_graph.binary_directed import biophysical_reverse_outdegree as biophysical_model
    from network_plot.change_settings import (set_all_text_fontsizes,
                                              set_all_colors)
    import brain_constants as bc
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.close('all')
    plt.ion()

    # PLOT PARAMETERS
    FIGSIZE = (16, 8)
    FONTSIZE = 20
    MARKERSIZE = 25
    BINWIDTH = 4
    MARKERCOLOR = 'cyan'
    FACECOLOR = 'black'
    LABELCOLOR = 'white'
    TICKSIZE = 2.

    # load brain graph, adjacency matrix, and labels
    G, A, D = biophysical_model(N=bc.num_brain_nodes,
                                N_edges=bc.num_brain_edges_directed, L=.75,
                                gamma=1.)

    # Get in & out degree
    indeg = np.array([G.in_degree()[node] for node in G])
    outdeg = np.array([G.out_degree()[node] for node in G])
    deg = indeg + outdeg
    deg_diff = outdeg - indeg

    # Calculate proportion in degree
    percent_indeg = indeg / deg.astype(float)

    # Create figure
    fig = plt.figure(figsize=FIGSIZE, facecolor=FACECOLOR, tight_layout=True)
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)

    # Add new axes for histograms in margins
    divider = make_axes_locatable(ax0)
    ax0_histTop = divider.append_axes('top', 1.2, pad=0.3, sharex=ax0)
    ax0_histRight = divider.append_axes('right', 1.2, pad=0.3, sharey=ax0)

    ##########################################################################
    # Call plotting function for scatter/marginal histograms (LEFT SIDE)
    plot_scatterAndMarginal(ax0, ax0_histTop, ax0_histRight, indeg, outdeg,
                            bin_width=BINWIDTH, marker_size=MARKERSIZE,
                            marker_color=MARKERCOLOR)

    ax0.set_xlabel('In-degree')
    ax0.set_ylabel('Out-degree')
    ax0_histTop.set_title('In- vs. Out-degree', fontsize=FONTSIZE + 2,
                          va='bottom')
    ##########################################################################
    # Plot percent_indeg vs. degree (RIGHT SIDE)
    ax1.scatter(deg, percent_indeg, s=MARKERSIZE, lw=0, c=MARKERCOLOR)
    ax1.set_xlabel('Total degree (in + out)')
    ax1.set_ylabel('Proportion in-degree')
    ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax1.set_yticks(np.arange(0, 1.1, .2))
    ax1.set_title('Proportion of Edges that are Incoming\nby Degree',
                  fontsize=FONTSIZE + 2, va='bottom')
    ax1.set_ylim([0., 1.05])

    ##########################################################################
    # Set background color and text size for all spines/ticks
    for temp_ax in [ax0, ax0_histRight, ax0_histTop, ax1]:
        set_all_text_fontsizes(temp_ax, FONTSIZE)
        set_all_colors(temp_ax, LABELCOLOR)
        #temp_ax.patch.set_facecolor(FACECOLOR)  # Set color of plot area
        temp_ax.tick_params(width=TICKSIZE)

    #fig.savefig('/home/wronk/Builds/fig_save.png', transparent=True)
    plt.show()
