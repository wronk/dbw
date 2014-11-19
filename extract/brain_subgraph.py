"""
Created on Tue Nov 18 15:32:52 2014

@author: rkp

Code for extracting sub-graphs of Allen mouse connectivity matrix.
"""

LINEAR_MODEL_DIRECTORY = '../../friday-harbor/linear_model'
STRUCTURE_DIRECTORY = '../../mouse_connectivity_data'

import numpy as np
import networkx as nx

import auxiliary as aux
import graph_tools.auxiliary as aux_tools
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
    
def cortex_distance_matrix(lm_dir=LINEAR_MODEL_DIRECTORY, 
                           cent_dir=STRUCTURE_DIRECTORY, in_mm=True):
    """Compute distance matrix from centroid data.
    
    Args:
        lm_dir: Directory containing linear model data
        cent_dir: Directory containing centroid data
        in_mm: Set to true to return dist matrix in mm instead of 100um units
    Returns:
        distance matrix, centroid matrix"""
    # Get labels
    _, _, labels = aux.load_W_and_P(data_dir=lm_dir)
    
    # Get mask of areas in cortex
    mask = aux.mask_specific_structures(labels, ['CTX'])
    
    # Get new labels
    labels = list(np.array(labels)[mask])
    
    # Load centroids
    centroids = aux.load_centroids(labels, data_dir=cent_dir, in_mm=in_mm)
    # Compute distance matrix
    dist_mat = aux_tools.dist_mat(centroids)
    
    return dist_mat, labels