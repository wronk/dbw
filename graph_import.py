'''

@author wronk
08/26/2014

'''

import networkx as nx


def import_weights_to_graph(weight_mat):
    '''
    Convert a weight dict into a NetworkX graph object
    '''

    assert 'data' in weight_mat.keys(), 'data not in weight matrix'
    assert 'row_labels' in weight_mat.keys(), 'row_labels not in weight matrix'
    assert 'col_labels' in weight_mat.keys(), 'col_labels not in weight matrix'

    # Initialize the graph
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

    # Add list of edges to graph object
    G.add_weighted_edges_from(edges_to_add)

    return G
