'''
Created on Tue Aug 27 2014

@author: wronk

network_viz.py
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3D_network(ont_names, ont_positions, label_set, edges, edge_labels,
                    node_sizes=1, node_colors='g', edge_sizes=1,
                    edge_colors='b'):
    '''
    Plot clustering coefficient probability density function

    Parameters
    ----------
    ont_names : list
        matrix of clustering coefficients
    ont_positions : N x 3 array
        array of positions
    edges : list
        connections between nodes

    Returns
    --------
    fig : fig
        figure object of distribution histogram for plotting
    '''

    # Initialize figure
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax = fig.add_subplot(111, projection='3d')
    node_label_offset = 0.1
    edge_label_offset = 0.05

    # Add nodes to graph
    ax.scatter(ont_positions[:, 0], ont_positions[:, 1], ont_positions[:, 2],
               s=node_sizes, c=node_colors, depthshade=False)
    for ni, node_pt in enumerate(ont_positions):
        ax.text(ont_positions[ni, 0], ont_positions[ni, 1],
                ont_positions[ni, 2] + node_label_offset, ont_names[ni],
                color=node_colors[ni], ha='center')

    ###################################
    # Convert edges to list
    edge_inds = [[ont_names.index(n1), ont_names.index(n2)]
                 for n1, n2 in edges]
    print edge_inds

    # Generate list of 2 x 3 array corresponding to list of end edge points
    edge_positions = [np.vstack((ont_positions[e[0]], ont_positions[e[1]]))
                      for e in edge_inds]
    print edge_positions[0]

    # Add edges to graph
    for ei, edge_pt in enumerate(edge_positions):
        ax.plot(edge_pt[:, 0], edge_pt[:, 1], edge_pt[:, 2],
                lw=edge_sizes[ei], c=edge_colors[ei],
                dash_capstyle='round', zorder=-1)

        mean_pt = np.mean(edge_pt, axis=0)
        vec_dir = edge_pt[1, :] - edge_pt[0, :]
        ax.text(mean_pt[0], mean_pt[1], mean_pt[2] + edge_label_offset,
                ont_names[edge_inds[ei][0]] + '<->' + ont_names[edge_inds[ei][1]],
                vec_dir, color=edge_colors[ei])

    # Cleanup
    ax.set_xlabel('X pos')
    ax.set_ylabel('Y pos')
    ax.set_zlabel('Z pos')

    plt.show()
    return fig, ax

if __name__ == '__main__':

    import network_gen

    plt.close('all')
    plt.ion()

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
    G = network_gen.import_weights_to_graph(W_net_dict)

    ont_names = row_labels[0:5]
    ont_positions = np.random.rand(5, 3)
    ont_labels = [True] * len(ont_names)
    edges = [('AAA_L', 'ACAv_L'), ('AD_L', 'ACAd_L')]

    node_sizes = np.random.randint(20, high=300, size=len(ont_names))
    node_colors = ['g'] * len(ont_names)

    edge_sizes = np.random.randint(1, high=4, size=len(edges))
    edge_colors = ['b'] * len(edges)
    edge_labels = [True] * len(edges)

    fig, ax = plot_3D_network(ont_names, ont_positions, ont_labels, edges,
                              edge_labels, node_sizes, node_colors,
                              edge_sizes, edge_colors)

    plt.draw()
    plt.show()
