'''
Created on Tue Aug 27 2014

@author: wronk

network_viz.py
'''
import numpy as np


def plot_3D_network(ont_names, ont_positions, edges):
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

    # Initialize graph

    # Add nodes to graph

    # Add edges to graph

if __name__ == '__main__':

    import scipy.io as sio
    import os.path as op

    import networkx as nx
    import matplotlib.pyplot as plt

    import network_gen
    import plot_net
    import shortest_path
    plt.close('all')

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
