"""
Created on Tue Nov 18 15:32:52 2014

@author: rkp

Code for extracting sub-graphs of Allen mouse connectivity matrix.
"""

LINEAR_MODEL_DIRECTORY = '../../friday-harbor/linear_model'
CENTROID_DIRECTORY = '../../mouse_connectivity_data'

import numpy as np
import networkx as nx

import auxiliary as aux
import brain_graph

def cortex_binary_undirected(p_th=.01, w_th=0, data_dir=LINEAR_MODEL_DIRECTORY):
    """Extract graph corresponding to cortex only.
    
    Returns:
        NetworkX graph, adjacency matrix, row labels, column labels"""
    # Load entire graph
    G, A, labels = brain_graph.binary_undirected(p_th, w_th, data_dir)
    
    # Get mask of areas in cortex
    mask = aux.mask_specific_structures(labels, ['CTX'])
    
    # Get new labels
    labels = list(np.array(labels)[mask])
    
    # Calculate cortex-only adjacency matrix
    A = A[mask,:][:,mask]
    
    # Create cortex-only graph
    G = nx.from_numpy_matrix(A)
    
    return G, A, labels