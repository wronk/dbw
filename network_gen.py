import pdb
"""
Created on Tue Aug 26 19:20:36 2014

@author: rkp

Loads weight matrix and creates network model from it according to p_value
& weight thresholding. Also includes option to create lesioned network.
"""

import networkx as nx
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from copy import deepcopy


def load_weights(dir_name):
    """Load weights into matrix."""

    # Load files
    D_W_ipsi = sio.loadmat(dir_name + '/W_ipsi.mat')
    D_W_contra = sio.loadmat(dir_name + '/W_contra.mat')
    D_PValue_ipsi = sio.loadmat(dir_name + '/PValue_ipsi.mat')
    D_PValue_contra = sio.loadmat(dir_name + '/PValue_contra.mat')

    # Make weight matrix for each side, then concatenate them
    W_L = np.concatenate([D_W_ipsi['data'], D_W_contra['data']], 1)
    W_R = np.concatenate([D_W_contra['data'], D_W_ipsi['data']], 1)
    W = np.concatenate([W_L, W_R], 0)
    # Make p_value matrix in the same manner
    P_L = np.concatenate([D_PValue_ipsi['data'], D_PValue_contra['data']], 1)
    P_R = np.concatenate([D_PValue_contra['data'], D_PValue_ipsi['data']], 1)
    P = np.concatenate([P_L, P_R], 0)

    col_labels = D_W_ipsi['col_labels']
    # Add ipsi & contra to col_labels
    col_labels_L = [label.split(' ')[0] + '_L' for label in col_labels]
    col_labels_R = [label.split(' ')[0] + '_R' for label in col_labels]
    col_labels_full = col_labels_L + col_labels_R
    row_labels_full = col_labels_full[:]

    return W, P, row_labels_full, col_labels_full


def threshold(W, P, p_th=.01, w_th=0):
    """Threshold empirical weight matrix to get network weights.

    Args:
        W: weight matrix
        P: p_value matrix
        p_th: max p-value threshold
        w_th: min weight threshold
    Returns:
        weight matrix with all sub-threshold values set to -1
    """

    W_net = deepcopy(W)

    W_mask = W <= w_th
    P_mask = P >= p_th
    mask = W_mask + P_mask

    W_net[mask] = -1

    return W_net, mask


def lesion_node(W_net, idxs):
    '''Simulate a simple node lesion in the network and pass back copy.

    Parameters
    ----------
    W_net : matrix
        matrix of weights specifying network
    idxs : list | N x 1 matrix
        node indices needing to be lesioned (set to zero in W_net)

    Returns
    -------
    W_lesion : matrix
        matrix with rows and columns deleted to reflect lesion
    '''

    # Make sure col_idxs is iterable
    assert hasattr(idxs, '__iter__'), 'Lesion idxs specified not iterable'

    # Count how many connections were lost due to the lesion
    num_cxns_lost = 0

    # Loop through all area idxs & simulate lesions by setting
    # the row & column corresponding to that index to zero
    W_lesion = deepcopy(W_net)
    for a_idx in idxs:
        # Count how many connections were lost due to the lesion
        num_cxns_lost += ((W_lesion[:, a_idx] > 0).sum() +
                          (W_lesion[a_idx, :] > 0).sum())

        W_lesion = np.delete(W_lesion, a_idx, axis=0)
        W_lesion = np.delete(W_lesion, a_idx, axis=1)
        print W_lesion.shape

    return W_lesion, num_cxns_lost


def lesion_edge(W_net, idxs):
    '''Simulate a simple edge lesion in the network and pass back copy.

    Parameters
    ----------
    W_net : matrix
        matrix of weights specifying network
    idxs : list of 2d lists | N x 2 matrix
        row/column indices needing to be lesioned (set to zero in W_net)

    Returns
    -------
    W_lesion : matrix
        matrix reflecting lesion
    '''

    W_lesion = deepcopy(W_net)

    # Make sure col_idxs is iterable
    assert hasattr(idxs, '__iter__'), 'Lesion specified not iterable'

    # Loop through all idxs & simulate lesions by setting # the cell indexed
    # to zero
    for a_idx in idxs:
        W_lesion[a_idx[0], a_idx[1]] = 0.

    # Count how many connections were lost due to the lesion
    num_cxns_lost = (W_lesion - W_net <= 0).sum()

    return W_lesion, num_cxns_lost


def import_weights_to_graph(graph_dict, directed=False):
    '''
    Convert a weight/label dict into a NetworkX graph object
    '''

    assert 'data' in graph_dict.keys(), 'data not in weight matrix'
    assert 'row_labels' in graph_dict.keys(), 'row_labels not in weight matrix'
    assert 'col_labels' in graph_dict.keys(), 'col_labels not in weight matrix'

    # Initialize the graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add nodes to graph according to names
    G.add_nodes_from(graph_dict['row_labels'])

    # Add edges to list object according to names
    edges_to_add = []
    for ri, row in enumerate(graph_dict['row_labels']):
        for ci, col in enumerate(graph_dict['col_labels']):
            if graph_dict['data'][ri, ci] > 0:
                edges_to_add.append((row, col, graph_dict['data'][ri, ci]))

    # Add list of edges to graph object
    G.add_weighted_edges_from(edges_to_add, weight='weight')
    #print G.edges()

    return G


def quick_net(p_th=.01,w_th=0,dir_name='../friday-harbor/linear_model'):
    """Quickly load and threshold the weight matrix."""

    # Load weights & p-values
    W,P,row_labels,col_labels = load_weights(dir_name)
    # Threshold weights according to weights & p-values
    W_net,mask = threshold(W,P,p_th=p_th,w_th=w_th)
    # Set weights to zero if they don't satisfy threshold criteria
    W_net[W_net==-1] = 0.
    # Set diagonal weights to zero
    np.fill_diagonal(W_net,0)
    
    return W_net,row_labels,col_labels
    

## function becomes symmetric unintentionally by nature of nondirected graph
#def import_graph_to_weights(graph, node_labels):
#    '''
#    Convert a NetworkX graph object into a weight matrix
#
#    Parameters
#    ----------
#    graph : NetworkX graph
#        graph needing conversion to a dict with weights and labels
#    node_labels : list of str
#        str names of all labels to be returned in the weight matrix
#
#    Returns
#    -------
#    weight_info : dict
#        dict containing lists 'row_labels' 'col_labels'
#        and 2-D array 'data' of size row_labels x col_labels
#    '''
#    nodes = graph.nodes()
#
#    weights = np.zeros((len(nodes), len(nodes)))
#
#    for ri, r_key in enumerate(node_labels):
#        for ci, c_key in enumerate(node_labels):
#            temp_dict = graph.get_edge_data(r_key, c_key, default={})
#            if 'weight' in temp_dict:
#                weights[ri, ci] = temp_dict['weight']
#                #weights[ci, ri] = temp_dict['weight']
#
#    return {'row_labels': node_labels, 'col_labels': node_labels,
#            'data': weights}


if __name__ == '__main__':
    
    W_net,row_labels,col_labels = quick_net()

    # Put everything in a dictionary
    W_net_dict = {'row_labels': row_labels, 'col_labels': col_labels,
                  'data': W_net}

    # Plot log-connectivity matrix
    #plt.imshow(np.log(W_net),interpolation='nearest')

    # Convert to networkx graph object
    G = import_weights_to_graph(W_net_dict)
