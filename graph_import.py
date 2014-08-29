'''
graph_import

@author wronk
08/26/2014

'''

import networkx as nx
import numpy as np


def import_weights_to_graph(weight_mat, directed=False):
    '''
    Convert a weight dict into a NetworkX graph object
    '''

    assert 'data' in weight_mat.keys(), 'data not in weight matrix'
    assert 'row_labels' in weight_mat.keys(), 'row_labels not in weight matrix'
    assert 'col_labels' in weight_mat.keys(), 'col_labels not in weight matrix'

    # Initialize the graph
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Add nodes to graph according to names
    G.add_nodes_from(weight_mat['col_labels'])

    # Add edges to list object according to names
    # Potential target for optimization
    edges_to_add = []
    for ri, row in enumerate(weight_mat['data']):
        for ci, col in enumerate(row):
            if weight_mat['data'][ri, ci] > 0:
                edges_to_add.append((weight_mat['row_labels'][ri],
                                    weight_mat['col_labels'][ci],
                                    weight_mat['data'][ri, ci]))

    # Add list of edges to graph object (row_label, col_label, 'weight')
    G.add_weighted_edges_from(edges_to_add)

    return G


# TODO can't handle undirected graphs (2x too many edges)
def import_graph_to_weights(graph, node_labels):
    '''
    Convert a NetworkX graph object into a weight matrix

    Parameters
    ----------
    graph : NetworkX graph
        graph needing conversion to a dict with weights and labels
    node_labels : list of str
        str names of all labels to be returned in the weight matrix

    Returns
    -------
    weight_info : dict
        dict containing lists 'row_labels' 'col_labels'
        and 2-D array 'data' of size row_labels x col_labels
    '''
    nodes = graph.nodes()

    weights = np.zeros((len(nodes), len(nodes)))

    for ri, r_key in enumerate(node_labels):
        for ci, c_key in enumerate(node_labels):
            temp_dict = graph.get_edge_data(r_key, c_key)
            weights[ri, ci] = temp_dict['weight']

    return {'row_labels': node_labels, 'col_labels': node_labels,
            'data': weights}
