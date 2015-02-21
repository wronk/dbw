'''
Created on Tue Aug 27 2014

@author: wronk

network_viz.py
'''
import numpy as np
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

def make_movie:
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


def plot_scatter_marginal(ax, k_in, k_out, bin_width, color):
    """
    Function to create a scatter plot with marginal histogram distributions.

    Parameters
    ==========
    ax : axes object
    k_in : array or list
        In degree.
    k_out : array or list
        Out degree
    bin_width : float
        Desired width of histogram bins.
    color : str
        Color of scatter points and histogram.

    Returns
    =======
    ax

    """

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    bin_width = float(bin_width)

    # Create scatter plot
    ax.scatter(k_in, k_out, c=color)

    # Add new histogram axes objects
    divider = make_axes_locatable(ax)
    ax_histTop = divider.append_axes('top', 1.2, pad=0.05, sharex=ax)
    ax_histRight = divider.append_axes('right', 1.2, pad=0.05, sharey=ax)

    # Remove tick labels
    plt.setp(ax_histTop.get_xticklabels() + ax_histRight.get_yticklabels(),
             visible=False)

    xymax = np.max([np.max(np.fabs(k_in)), np.max(np.fabs(k_out))])
    lim = (int(xymax / bin_width) + 1) * bin_width

    # Create bin edges
    bins = np.arange(-lim, lim + bin_width, bin_width)

    # Plot histograms
    ax_histTop.hist(k_in, bins=bins, orientation='vertical', c=color)
    ax_histRight.hist(k_out, bins=bins, orientation='horizontal', c=color)

    return ax


if __name__ == '__main__':
    #make_movie()

    import numpy as np
    import matplotlib.pyplot as plt

    from extract.brain_graph import binary_directed as brain_graph
    from network_plot.change_settings import set_all_text_fontsizes, set_all_colors

    # PLOT PARAMETERS
    FACECOLOR = 'black'
    FONTSIZE = 16
    NBINS = 15

    # load brain graph, adjacency matrix, and labels
    G, A, labels = brain_graph()

    # get in & out degree
    indeg = np.array([G.in_degree()[node] for node in G])
    outdeg = np.array([G.out_degree()[node] for node in G])
    deg = indeg + outdeg
    deg_diff = outdeg - indeg

    # calculate percent in & percent out degree
    percent_indeg = indeg / deg.astype(float)
    percent_outdeg = outdeg / deg.astype(float)

    # open figure
    fig = plt.figure(facecolor=FACECOLOR, tight_layout=True)

    ax00 = fig.add_subplot(2, 3, 1)
    ax10 = fig.add_subplot(2, 3, 4, sharex=ax00)
    ax01 = fig.add_subplot(2, 3, 2, sharey=ax00)

    # plot out vs. in-degree scatter
    ax00.scatter(indeg, outdeg, lw=0)
    ax00.set_xlabel('indegree')
    ax00.set_ylabel('outdegree')

    # plot out & in-degree distributions
    ax01.hist(outdeg, bins=NBINS, orientation='horizontal')
    ax01.set_ylabel('outdegree')
    ax01.set_xlabel('# nodes')
    ax01.set_xticks(np.arange(0, 161, 40))

    ax10.hist(indeg, bins=NBINS)
    ax10.set_xlabel('indegree')
    ax10.set_ylabel('# nodes')

    '''
# plot percent_indeg & percent_outdeg distributions
    ax11.hist(percent_indeg, bins=NBINS)
    ax11.set_xlabel('% indegree')
    ax11.set_ylabel('# nodes')
    ax11.set_xticks(np.arange(0, 1.1, .2))

# plot scatter
    ax02.scatter(deg, deg_diff, lw=0)
    ax02.set_xlabel('outdegree + indegree')
    ax02.set_ylabel('outdegree - indegree')
    ax02.set_xticks(np.arange(0, 161, 40))

# plot percent_indeg vs. degree
    ax12.scatter(deg, percent_indeg, lw=0)
    ax12.set_xlabel('outdegree + indegree')
    ax12.set_ylabel('% indegree')
    ax12.set_xticks(np.arange(0, 161, 40))
    ax12.set_yticks(np.arange(0, 1.1, .2))
    '''

    for ax in [ax00, ax01, ax10]:
        set_all_text_fontsizes(ax, FONTSIZE)
        set_all_colors(ax, 'white')


